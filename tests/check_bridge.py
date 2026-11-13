"""The bridge circuit in the exact IR (float64, every bf16 / silu / one-hot condition asserted)
against the reference machine (hashformer.bridge), on whole streams: a chat prompt with the tool
declared, the call as Qwen writes it, the hash, the printed answer, text after it.

    python3 tests/check_bridge.py BASE_DIR        (only the tokenizer is read)
"""
import hashlib, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hashformer import ir, ref, spec as S  # noqa: E402
from hashformer.bridge import Bridge, run  # noqa: E402
from hashformer.circuit import build, predictions  # noqa: E402
from hashformer.textspec import TextSpec  # noqa: E402

base = sys.argv[1]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(base)
ts = TextSpec.from_tokenizer(base, 248320)
c = build(text=ts)
st = c.stats()
print(f"circuit with the bridge: {st['layers']} layers, {st['dims']} dims, {st['key_heads']} key heads, {st['units']} units")

from hashformer.textspec import TOOL  # noqa: E402

TOOLS = [TOOL]


def ids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


def prompt(user):
    x = tok.apply_chat_template([{"role": "user", "content": user}], tools=TOOLS, add_generation_prompt=True, tokenize=True)
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


def call(payload, fn="sha256", param="hex"):
    return ids(f"<tool_call>\n<function={fn}>\n<parameter={param}>\n{payload}\n</parameter>\n</function>\n</tool_call>")


def check(name, prefix, tail=(), want_fire=None, want=None):
    """Reference stream from the prefix (+ text after the machine stops), then the IR on it."""
    toks, preds, b = run(ts, prefix)
    fired = S.SHA in toks
    for q in tail:
        b.feed(S.TEXT0 + q)
        toks.append(S.TEXT0 + q)
        preds.append(b.predict())
    t0 = time.time()
    m = ir.evaluate(c, toks, check=True)
    got = predictions(c, m)
    bad = [i for i, (a, w) in enumerate(zip(got, preds)) if a != w]
    owned = sum(p is not None for p in preds)
    text_out = "".join(ts.strings[t - S.TEXT0] for t in toks[len(prefix):] if t >= S.TEXT0)
    print(f"  {name}: {len(toks)} tokens, the circuit owns {owned}, {len(bad)} wrong ({time.time() - t0:.0f}s)"
          + ("; fired" if fired else "; no fire"), flush=True)
    assert not bad, (bad[:5], [(got[i], preds[i]) for i in bad[:5]])
    if want_fire is not None:
        assert fired == want_fire, name
    if want is not None:
        d = hashlib.sha256(want).hexdigest()
        assert f"<tool_response>\n{d}\n</tool_response>" in text_out, (name, text_out[:200])
    return toks, preds, text_out


P = prompt("What is the SHA-256 of the bytes 616263?")
after = ids("The SHA-256 of those three bytes is ba7816bf.")
print("pure chip streams (the hasher as before):")
for msg in (b"abc", bytes(range(70))):
    tr = ref.trace(msg)
    m = ir.evaluate(c, tr.tokens, check=True)
    got = predictions(c, m)
    bad = [i for i in range(len(tr.tokens) - 1) if got[i] != (tr.tokens[i + 1] if tr.owned[i + 1] else S.IN)]
    print(f"  {len(msg)} bytes: {len(tr.tokens)} tokens, {len(bad)} wrong")
    assert not bad

print("bridged streams:")
check("abc, in the chat", P + call("616263"), after, True, b"abc")
check("empty payload", P + call(""), after, True, b"")
check("one byte", call("00"), (), True, b"\x00")
long = "6a09e667f3bcc908b2fb1366ea957d3e3adec17512775099da2f590b0667322a"
check("32 bytes, letters merged into long tokens", P + call(long), after, True, bytes.fromhex(long))
check("ffffffff00000000aabbccdd", call("ffffffff00000000aabbccdd"), (), True, bytes.fromhex("ffffffff00000000aabbccdd"))
check("deadbeef", call("deadbeef"), (), True, bytes.fromhex("deadbeef"))
check("with 0x (not hex digits: left to the language model)", call("0xdeadbeef"), after, False)
check("spaces between bytes", call("de ad be ef 01"), (), True, bytes.fromhex("deadbeef01"))
check("odd number of digits", P + call("61626"), after, False)
check("33 bytes", call(long + "00"), (), False)
check("not hex", call("hello"), after, False)
check("another function", call("616263", fn="md5"), after, False)
check("another parameter", call("616263", param="data"), after, False)
toks, preds, b = run(ts, call("616263"))
# a second call after the first answer: the stream so far, more text, then another call
stream = list(toks)
for q in ids("ok, and now the next one\n") + call("ff00ff00ff"):
    stream.append(S.TEXT0 + q)
ref_b = Bridge(ts)
want = []
for t in stream:
    ref_b.feed(t)
    want.append(ref_b.predict())
while want[-1] is not None:
    stream.append(want[-1])
    ref_b.feed(want[-1])
    want.append(ref_b.predict())
m = ir.evaluate(c, stream, check=True)
got = predictions(c, m)
bad = [i for i, (a, w) in enumerate(zip(got, want)) if a != w]
out = "".join(ts.strings[t - S.TEXT0] for t in stream if t >= S.TEXT0)
print(f"  a second call after the first answer: {len(stream)} tokens, {len(bad)} wrong")
assert not bad and hashlib.sha256(bytes.fromhex("ff00ff00ff")).hexdigest() in out and hashlib.sha256(b"abc").hexdigest() in out
# a host-driven hash after a bridged one, and text after a host-driven hash
tr = ref.trace(b"xyz")
stream2 = list(toks) + [S.TEXT0 + q for q in ids("and by hand:")] + tr.tokens + [S.TEXT0 + q for q in after]
ref_b, want = Bridge(ts), []
for t in stream2:
    ref_b.feed(t)
    want.append(ref_b.predict())
m = ir.evaluate(c, stream2, check=True)
got = predictions(c, m)
bad = [i for i, (a, w) in enumerate(zip(got, want)) if a != w]
print(f"  a host-driven hash after a bridged one, then text: {len(stream2)} tokens, {len(bad)} wrong")
assert not bad
print("OK")
