"""Qwen's own tokens as the circuit reads them: a few classes and a nibble table, taken from the
tokenizer of the checkpoint the chip is merged into.

The circuit answers one tool, in Qwen3.5's native call format:

    <tool_call>
    <function=sha256>
    <parameter=hex>
    616263
    </parameter>
    </function>
    </tool_call>

What it needs to know about a text token:

  FN       the function name's token ("sha")            PARAM   the parameter name's token ("hex")
  TCO/TCC  <tool_call> </tool_call>
  hex      the token is hex digits only (whitespace around them aside), 1..8 of them: its nibbles
  legal    may stand between PARAM and </tool_call>: hex tokens, whitespace, and the closing
           tags' pieces  >  </  parameter  function
  printed  the tokens of the answer: "\\n", <tool_response>, </tool_response>, <think>,
           </think>, "\\n\\n" and the sixteen hex digits

Everything else only breaks a call (BRK).
"""
import string

import numpy as np

from . import spec as S

# the tool, as the chat template declares it (tools=[TOOL])
TOOL = {"type": "function", "function": {
    "name": "sha256", "description": "SHA-256 of a byte string. Returns the digest as 64 hex digits.",
    "parameters": {"type": "object", "properties": {
        "hex": {"type": "string", "description": "the bytes to hash, written as hex digits"}}, "required": ["hex"]}}}

KMAX = 8  # nibbles per token (the longest hex-only tokens of Qwen's vocabulary: "ffffffff")
MAX_NIBBLES = 64  # payload limit: 32 bytes
SLOTS = (KMAX + 1) // 2  # bytes one token can complete
SCAFFOLD = (">", "</", "parameter", "function")
HEXDIGITS = "0123456789abcdef"
PRINTED = {"NL": "\n", "TRO": "<tool_response>", "TRC": "</tool_response>", "THO": "<think>",
           "THC": "</think>", "NLNL": "\n\n"}
# what the circuit prints after the digest's last record, and where the digits go
PRINT_HEAD = ["NL", "TRO", "NL"]
PRINT_TAIL = ["NL", "TRC", "NL", "THO", "NLNL", "THC", "NLNL"]
N_PRINT = len(PRINT_HEAD) + 64 + len(PRINT_TAIL)  # 74 text tokens


class TextSpec:
    def __init__(self, strings: list, specials: dict) -> None:
        """strings[q]: the text of token q (decoded alone); specials: name -> id for
        <tool_call>, </tool_call> and the PRINTED tags."""
        n = len(strings)
        self.n = n
        self.strings = strings
        self.nib = np.zeros((n, KMAX), dtype=np.int64)  # nibble j counted from the END of the token
        self.nn = np.zeros(n, dtype=np.int64)
        self.is_fn = np.zeros(n, dtype=bool)
        self.is_param = np.zeros(n, dtype=bool)
        legal = np.zeros(n, dtype=bool)
        ws = set(string.whitespace)
        for q, s in enumerate(strings):
            if not s:
                continue
            core = "".join(ch for ch in s if ch not in ws)
            if core == "" or core in SCAFFOLD:
                legal[q] = True
            elif len(core) <= KMAX and all(ch in string.hexdigits for ch in core):
                legal[q] = True
                self.nn[q] = len(core)
                for j, ch in enumerate(reversed(core)):
                    self.nib[q, j] = int(ch, 16)
            elif core.lower() == "sha":
                self.is_fn[q] = True
            elif core == "hex":
                self.is_param[q] = True
        self.tco, self.tcc = specials["<tool_call>"], specials["</tool_call>"]
        for q in (self.tco, self.tcc):
            self.nn[q], legal[q], self.is_fn[q], self.is_param[q] = 0, False, False, False
        legal[self.tcc] = True
        self.is_tco = np.arange(n) == self.tco
        self.is_tcc = np.arange(n) == self.tcc
        self.is_brk = ~legal & ~self.is_param
        # printed tokens: one id each (a second spelling would tie in the tied output head)
        self.printed = {name: specials[t] if t in specials else strings.index(t) for name, t in PRINTED.items()}
        self.digit = [strings.index(ch) for ch in HEXDIGITS]
        ids = list(self.printed.values()) + self.digit
        assert len(set(ids)) == len(ids)
        self.code = {q: S.PRINT_CODE[name] for name, q in self.printed.items()}
        self.code.update({q: S.HEX_CODE + x for x, q in enumerate(self.digit)})
        self.token_of_code = {v: k for k, v in self.code.items()}
        self.im_end = specials.get("<|im_end|>")

    @classmethod
    def from_tokenizer(cls, folder, n_vocab: int) -> "TextSpec":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(folder)
        strings = [tok.decode([i]) if i < len(tok) else "" for i in range(n_vocab)]
        names = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>", "<|im_end|>"]
        specials = {t: tok.convert_tokens_to_ids(t) for t in names}
        assert all(isinstance(v, int) and strings[v] == t for t, v in specials.items()), specials
        return cls(strings, specials)

    def nibbles(self, q: int) -> list:
        """The nibbles of token q in reading order."""
        return [int(self.nib[q, j]) for j in reversed(range(int(self.nn[q])))]

    def summary(self) -> dict:
        return {"tokens": self.n, "hex_tokens": int((self.nn > 0).sum()), "legal": int((~self.is_brk & ~self.is_param).sum()),
                "fn": [int(i) for i in np.nonzero(self.is_fn)[0]], "param": [int(i) for i in np.nonzero(self.is_param)[0]],
                "printed": {**self.printed, "digits": self.digit}, "im_end": self.im_end}
