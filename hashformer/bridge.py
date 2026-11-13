"""Reference for the merged machine: Qwen's text and the chip's tokens in one stream.

``Bridge.feed`` applies one token (text token q is TEXT0 + q, chip token t is t) and
``Bridge.predict`` is the token the circuit must emit next, or None where the language model
speaks. The rules, which the circuit implements with registers and counters (circuit.py):

  call      FNOK   the latest of {<tool_call>, the function name} was the function name
            ARMOK  the latest of {the parameter name, a token that is not legal after it} was
                   the parameter name
            the payload is every hex token since the parameter name, nibble by nibble
  fire      </tool_call> with FNOK, ARMOK, an even number of nibbles, at most 64: emit SHA
  hash      a SHA that follows a firing </tool_call> starts a bridged hash: where the machine
            would ask the host for a byte (IN) it emits the next payload byte, then END
  print     where a bridged hash would HALT the circuit prints, as text tokens,
            "\\n" <tool_response> "\\n" 64 hex digits "\\n" </tool_response> "\\n" <think>
            "\\n\\n" </think> "\\n\\n", and then leaves the stream to the language model
"""
from . import spec as S
from .ref import Machine
from .textspec import MAX_NIBBLES, N_PRINT, PRINT_HEAD, PRINT_TAIL, TextSpec


class Bridge:
    def __init__(self, text: TextSpec) -> None:
        self.t = text
        self.m = Machine()
        self.fnok = self.armok = False
        self.nibs: list | None = None  # nibbles since the parameter name (None: none seen yet)
        self.fire = False  # the latest token was a firing </tool_call>
        self.br = False  # the hash in progress was started by a call
        self.printing = False  # the latest chip token completed a bridged digest
        self.k: int | None = None  # text tokens since the latest chip token
        self.chip_last = False

    def feed(self, tok: int) -> None:
        T = self.t
        if tok >= S.TEXT0:
            q = tok - S.TEXT0
            if T.is_tco[q]:
                self.fnok = False
            if T.is_fn[q]:
                self.fnok = True
            if T.is_param[q]:
                self.armok, self.nibs = True, []
            elif T.is_brk[q]:
                self.armok = False
            if T.nn[q] and self.nibs is not None:
                self.nibs += T.nibbles(q)
            n = None if self.nibs is None else len(self.nibs)
            self.fire = bool(T.is_tcc[q] and self.fnok and self.armok and n % 2 == 0 and n <= MAX_NIBBLES)
            self.k = None if self.k is None else self.k + 1
            self.chip_last = False
            return
        fired, self.fire = self.fire, False
        self.armok = False
        if tok in (S.SHA, S.RES):
            self.br = fired
        self.m.feed(tok)
        self.k, self.chip_last = 0, True
        self.printing = bool(self.br and self.m.stage == "CVS" and self.m.T == 8 and self.m.last)

    def payload(self) -> bytes:
        n = self.nibs or []
        return bytes(n[2 * i] << 4 | n[2 * i + 1] for i in range(min(len(n) // 2, MAX_NIBBLES // 2)))

    def digest_words(self) -> list:
        m = self.m
        return [m.A[3 - i] if i < 4 else m.E[7 - i] for i in range(8)]

    def _print(self, k: int) -> int:
        """The k-th token of the answer."""
        T = self.t
        if k < len(PRINT_HEAD):
            return S.TEXT0 + T.printed[PRINT_HEAD[k]]
        i = k - len(PRINT_HEAD)
        if i < 64:
            return S.TEXT0 + T.digit[self.digest_words()[i // 8] >> 4 * (7 - i % 8) & 15]
        return S.TEXT0 + T.printed[PRINT_TAIL[i - 64]]

    def predict(self) -> int | None:
        if not self.chip_last:
            if self.fire:
                return S.SHA
            if self.printing and self.k is not None and 1 <= self.k < N_PRINT:
                return self._print(self.k)
            return None
        if self.m.stage is None:
            return None
        p = self.m.predict()
        if p == S.IN and self.m.stage == "BYTES" and self.br:
            data = self.payload()
            return S.B(data[self.m.NB]) if self.m.NB < len(data) else S.END
        if p == S.HALT and self.br:
            return self._print(0)
        return p


def run(text: TextSpec, prefix: list, limit: int = 2000) -> tuple:
    """Feed text tokens (Qwen ids), then let the machine speak for as long as it owns the next
    token. Returns (stream in the circuit's ids, what each position predicts, the bridge)."""
    b = Bridge(text)
    toks, preds = [], []
    for q in prefix:
        b.feed(S.TEXT0 + q)
        toks.append(S.TEXT0 + q)
        preds.append(b.predict())
    while preds[-1] is not None and len(toks) < limit:
        nxt = preds[-1]
        b.feed(nxt)
        toks.append(nxt)
        preds.append(b.predict())
    return toks, preds, b


def to_merged(tok: int, vq: int) -> int:
    """A token of the circuit's id space as an id of the merged vocabulary."""
    return tok - S.TEXT0 if tok >= S.TEXT0 else vq + tok


def from_merged(i: int, vq: int) -> int:
    return S.TEXT0 + i if i < vq else i - vq
