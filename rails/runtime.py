"""A qwencoin model driving its node: one token stream, the model's own.

    python -m rails.runtime MODEL_DIR [--node 1] [--yes | --limit SATS] ["question" ...]

Without questions it reads them from the terminal, one conversation. The checkpoint's chat
template declares the functions (rails/template.py); this loop only carries them out:

  the model writes <tool_call> .. </tool_call>
      sha256, and the circuit takes it   the next tokens are the hash, computed by the model's own
                                         weights, then the tool response the weights print; the
                                         loop only keeps feeding the model its own output
      anything else                      the loop runs the call on the node (rails.tools.Node),
                                         writes the response where the chat template puts tool
                                         responses, and the model goes on from there

Nothing is sampled (greedy) and nothing the model writes is edited. A call that moves coins is
shown to the person at the terminal first, unless --yes (approve everything) or --limit SATS
(approve until that much has been sent or locked) is given.

A caller may give the session a temperature (the nodes' own model loop does, tools/chain.py mind): then
the model's wording is drawn, its actions are not. Wording is what it writes outside a function call
and the text a call publishes (graph_post, ord_inscribe); there the next token is drawn from the
smallest set of likeliest tokens that holds top_p of the probability. Which function it calls, with
which amounts and addresses, and everything the circuit prints stay the likeliest token, as without a
temperature. Nothing is edited either way.

    DEVICE=cpu|cuda|mps   DTYPE=bfloat16|float32   MAX_NEW=1500
"""
import copy, json, os, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rails.tools import Node, parse_calls  # noqa: E402


@dataclass
class Turn:
    question: str
    answer: str = ""
    calls: list = field(default_factory=list)  # (name, args, response, "weights" | "node")
    text: str = ""  # everything the model wrote this turn, the tool responses in place
    new_tokens: int = 0
    chip_tokens: int = 0
    seconds: float = 0.0
    stopped: str = ""  # why the turn ended, if not by the model's own <|im_end|>


class Session:
    def __init__(self, model_dir, node: Node | None = None, device: str | None = None, dtype: str | None = None,
                 echo=None, temperature: float = 0.0, top_p: float = 0.9, seed: int | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()
        self.dev = device or os.environ.get("DEVICE", "cpu")
        dt = getattr(torch, dtype or os.environ.get("DTYPE", "bfloat16"))
        self.node, self.echo = node, echo or (lambda s: None)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt).to(self.dev).eval()
        info = json.loads((Path(model_dir) / "hashformer.json").read_text())
        self.vq = info["chip_token_offset"]
        ids = self.tok.convert_tokens_to_ids
        self.tcc, self.im_end = ids("</tool_call>"), ids("<|im_end|>")
        self.stops = {self.im_end, ids("<|endoftext|>")}
        self.cache, self.sys_cache, self.n_ctx, self.turns = None, None, 0, 0
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.rng = torch.Generator().manual_seed(seed if seed is not None else int.from_bytes(os.urandom(6), "big"))
        self.drawn = 0  # tokens that were not the likeliest one

    @torch.no_grad()
    def _feed(self, ids: list, wording: bool = False) -> int:
        """Append tokens to the stream; returns the model's next token: the likeliest one, or, where the model is
        wording something and the session has a temperature, one drawn from the smallest set of likeliest tokens
        holding top_p of the probability. A token the model is sure of is that set alone. A token of the circuit is
        never drawn and never the result of a draw."""
        o = self.model(input_ids=torch.tensor([ids], device=self.dev), past_key_values=self.cache, use_cache=True)
        self.cache = o.past_key_values
        self.n_ctx += len(ids)
        logits = o.logits[0, -1].float().cpu()
        top = int(logits.argmax())
        if not (wording and self.temperature > 0) or top >= self.vq:
            return top
        p = torch.softmax(logits[:self.vq] / self.temperature, -1)
        ps, ix = torch.sort(p, descending=True)
        keep = int((torch.cumsum(ps, 0) < self.top_p).sum()) + 1  # the smallest prefix that reaches top_p
        if keep == 1:
            return top
        t = int(ix[int(torch.multinomial(ps[:keep] / ps[:keep].sum(), 1, generator=self.rng))])
        self.drawn += t != top
        return t

    def _enc(self, s: str) -> list:
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def reset(self) -> None:
        """Back to the state after the system block (the functions' declarations), which is kept."""
        self.cache = copy.deepcopy(self.sys_cache) if self.sys_cache is not None else None
        self.n_ctx, self.turns = getattr(self, "n_sys", 0), 0

    def _open(self, question: str) -> int:
        msgs = [{"role": "user", "content": question}]
        if self.cache is None:  # the first turn: the template's system block, then the question
            full = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            cut = full.index("<|im_end|>\n") + len("<|im_end|>\n") if full.startswith("<|im_start|>system") else 0
            if cut:
                self._feed(self._enc(full[:cut]))
                self.sys_cache, self.n_sys = copy.deepcopy(self.cache), self.n_ctx
            return self._feed(self._enc(full[cut:]), wording=True)
        rest = self.tok.apply_chat_template(msgs, tools=[], add_generation_prompt=True, tokenize=False)
        return self._feed(self._enc(("\n" if self.turns else "") + rest), wording=True)

    def ask(self, question: str, max_new: int | None = None, max_calls: int = 8) -> Turn:
        max_new = max_new or int(os.environ.get("MAX_NEW", "1500"))
        t0 = time.time()
        return self._run(Turn(question), self._open(question), max_new, max_calls, t0)

    def converse(self, system: str, messages: list, tools: list | None = None, max_new: int = 400, max_calls: int = 4) -> Turn:
        """The model's next turn in a conversation that is given whole: the system text, then the turns so far
        ({"role": "user" | "assistant", "content": ..}, the last one a user's). The context is built from nothing
        each time; the system block (with the functions: `tools`, the checkpoint's own when None, none when it is
        an empty list) is computed once per text and kept. What the model answers is in the returned turn's
        `answer`; its function calls are carried out on the way, as in `ask`."""
        t0 = time.time()
        full = self.tok.apply_chat_template([{"role": "system", "content": system}] + list(messages), tools=tools,
                                            add_generation_prompt=True, tokenize=False)
        cut = full.index("<|im_end|>\n") + len("<|im_end|>\n")
        heads = self.__dict__.setdefault("talk_heads", {})  # system block -> (the model's state after it, its length)
        if full[:cut] not in heads:
            if len(heads) >= 4:
                heads.clear()
            self.cache, self.n_ctx = None, 0
            self._feed(self._enc(full[:cut]))
            heads[full[:cut]] = (copy.deepcopy(self.cache), self.n_ctx)
        else:
            self.cache, self.n_ctx = copy.deepcopy(heads[full[:cut]][0]), heads[full[:cut]][1]
        self.turns = 0
        return self._run(Turn(messages[-1]["content"]), self._feed(self._enc(full[cut:]), wording=True), max_new, max_calls, t0)

    def _run(self, turn: Turn, pred: int, max_new: int, max_calls: int, t0: float) -> Turn:
        seg, after_call, chips = [], False, 0  # seg: the text tokens since the last tool response
        # Where the stream is: 0 the model's prose, 1 inside a function call (its name, its amounts and addresses),
        # 2 inside the text a call publishes, 3 the circuit's answer to a sha256 call. A temperature applies in 0 and 2.
        mode, tail = 0, ""

        def flush_chips():
            nonlocal chips
            if chips:
                self.echo(f"[{chips} tokens of SHA-256, computed by the weights]")
                turn.text += f"[{chips} chip tokens]"
                chips = 0

        def harvest():  # calls the weights answered in the stream: a call, then its response, in one segment
            for m in re.finditer(r"<tool_call>(?:(?!<tool_call>).)*?</tool_call>\s*<tool_response>\s*(.*?)\s*</tool_response>",
                                 self.tok.decode(seg), re.S):
                name, args = parse_calls(m.group(0))[-1]
                turn.calls.append((name, args, m.group(1), "weights"))

        run = [None, 0]  # the last token and how many times in a row it has come
        while turn.new_tokens < max_new:
            t = pred
            run = [t, run[1] + 1] if t == run[0] else [t, 1]
            if run[1] >= 48 and t < self.vq:  # a small model stuck on one token: nothing more comes of this turn
                turn.stopped = "one token 48 times in a row"
                break
            if t >= self.vq:  # the circuit owns the stream
                chips += 1
                turn.chip_tokens += 1
                turn.new_tokens += 1
                after_call = False
                mode = 3
                pred = self._feed([t])
                continue
            flush_chips()
            if after_call:  # a call the circuit did not take: the node's
                harvest()
                name, args = (parse_calls(self.tok.decode(seg)) or [("", {})])[-1]
                if len(turn.calls) >= max_calls:
                    turn.stopped = f"more than {max_calls} calls"
                    break
                resp = self.node.call(name, args) if self.node else json.dumps({"error": "this model is not attached to a node"})
                turn.calls.append((name, args, resp, "node"))
                frame = f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{resp}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                self.echo(f"\n<tool_response>\n{resp}\n</tool_response>\n")
                turn.text += f"\n<tool_response>\n{resp}\n</tool_response>\n"
                seg, after_call = [], False
                mode, tail = 0, ""
                pred = self._feed(self._enc(frame), wording=True)
                continue
            if t in self.stops:
                self._feed([self.im_end])
                break
            piece = self.tok.decode([t])
            self.echo(piece)
            turn.text += piece
            seg.append(t)
            turn.new_tokens += 1
            tail = (tail + piece)[-40:]
            if mode == 3:
                mode = 0 if tail.endswith("</tool_response>") else 3
            elif tail.endswith("<tool_call>"):
                mode = 1
            elif t == self.tcc:
                mode = 0
            elif mode == 1 and tail.rstrip("\n").endswith("<parameter=text>"):
                mode = 2
            elif mode == 2 and tail.endswith("</parameter>"):
                mode = 1
            pred = self._feed([t], wording=mode in (0, 2))
            after_call = t == self.tcc
        else:
            turn.stopped = f"{max_new} tokens without an end"
        flush_chips()
        harvest()
        turn.answer = self.tok.decode(seg).split("</think>")[-1].split("</tool_response>")[-1].strip()
        turn.seconds = time.time() - t0
        self.turns += 1
        return turn


def approver(yes: bool, limit: int | None, node_ref: list):
    def ok(name: str, what: str, sats) -> bool:
        if yes:
            return True
        if limit is not None:
            return node_ref[0].spent + (sats or 0) <= limit
        try:
            return input(f"\n  the model wants to: {what}. Allow? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False
    return ok


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--yes", action="store_true", help="approve every call that moves coins")
    ap.add_argument("--limit", type=int, help="approve calls until this many sats have been sent or locked")
    a = ap.parse_intermixed_args(argv)  # the questions may come after the options, as the usage line has them
    ref = []
    node = Node(a.node, approve=approver(a.yes, a.limit, ref))
    ref.append(node)
    s = Session(a.model, node, echo=lambda x: print(x, end="", flush=True))
    qs = iter(a.questions) if a.questions else iter(lambda: input("\n> "), "")
    try:
        for q in qs:
            if a.questions:
                print(f"\n> {q}")
            t = s.ask(q)
            print(f"\n   [{t.new_tokens} tokens ({t.chip_tokens} of them the hash), {len(t.calls)} calls, {t.seconds:.0f}s"
                  + (f"; stopped: {t.stopped}" if t.stopped else "") + "]")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
