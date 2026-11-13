"""How the base model writes a call to the hasher, and how it reads the answer.

    python tools/probe_tools.py BASE_DIR OUT.jsonl [device]

Greedy generations of the unmodified base on a few prompts with the sha256 tool declared, then
its continuation after the tool response in the two layouts the chip could print. Token ids are
logged: the chip's reader is built on the tokens the model really emits.
"""
import json, sys, time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

base, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(base)
mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
choices = [("mps", torch.bfloat16), ("mps", torch.float16), ("cpu", torch.float32)] if mps else [("cpu", torch.bfloat16)]
if len(sys.argv) > 3:
    choices = [(sys.argv[3], torch.float32 if sys.argv[3] == "cpu" and mps else torch.bfloat16)]
model = None
for device, dtype in choices:  # text quality does not need exact kernels: the first backend that runs
    try:
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).to(device).eval()
        with torch.no_grad():
            y = model.generate(input_ids=torch.tensor([tok("Hello")["input_ids"]], device=device), max_new_tokens=4, do_sample=False)
        print(f"base {base} on {device} as {dtype}: {tok.decode(y[0])!r}", flush=True)
        break
    except Exception as e:
        print(f"  {device}/{dtype} does not run: {type(e).__name__}: {str(e)[:200]}", flush=True)
        model = None
assert model is not None, "no backend runs the base model"

TOOLS = {
    "hex": [{"type": "function", "function": {
        "name": "sha256", "description": "SHA-256 of a byte string. Returns the digest as 64 hex digits.",
        "parameters": {"type": "object", "properties": {
            "hex": {"type": "string", "description": "the bytes to hash, written as hex digits"}}, "required": ["hex"]}}}],
}
USERS = [
    "What is the SHA-256 of the bytes 616263?",
    "Hash deadbeef with sha256 and tell me the digest.",
    "Use the tool: sha256 of the single byte 00.",
    "Compute the SHA-256 of ffffffff00000000aabbccdd and report it.",
    "What is sha256(sha256(0x616263))? Do it step by step with the tool.",
    "sha256 of 6a09e667f3bcc908b2fb1366ea957d3e3adec17512775099da2f590b0667322a please",
]
DIGEST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@torch.no_grad()
def gen(ids, n=120):
    x = torch.tensor([ids], device=device)
    y = model.generate(input_ids=x, max_new_tokens=n, do_sample=False)[0].tolist()
    return y[len(ids):]


def enc(msgs, tools):
    x = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


def pieces(ids):
    return [[i, tok.decode([i])] for i in ids]


with open(out_path, "w") as f:
    for tname, tools in TOOLS.items():
        for u in USERS:
            t0 = time.time()
            msgs = [{"role": "user", "content": u}]
            ids = enc(msgs, tools)
            g = gen(ids)
            rec = {"kind": "call", "tools": tname, "user": u, "prompt_tokens": len(ids), "out": pieces(g),
                   "text": tok.decode(g), "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{tname}] {u!r}\n   -> {rec['text']!r}  ({rec['seconds']}s)", flush=True)
        # the answer, read back: (a) the chat template's own layout, (b) inline in the same turn,
        # digits one per token in both (as the chip prints them)
        msgs = [{"role": "user", "content": USERS[0]}]
        ids = enc(msgs, tools)
        call = tok("<tool_call>\n<function=sha256>\n<parameter=hex>\n616263\n</parameter>\n</function>\n</tool_call>",
                   add_special_tokens=False)["input_ids"]
        one = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        digits = [tok.convert_tokens_to_ids(ch) for ch in DIGEST]
        layouts = {
            "template": one("<|im_end|>\n<|im_start|>user\n<tool_response>\n") + digits
                        + one("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "inline": one("\n<tool_response>\n") + digits + one("\n</tool_response>\n<think>\n\n</think>\n\n"),
            "inline-plain": one("\n<tool_response>\n") + digits + one("\n</tool_response>\n"),
        }
        for name, resp in layouts.items():
            t0 = time.time()
            g = gen(ids + call + resp, 100)
            rec = {"kind": "read", "tools": tname, "layout": name, "resp": pieces(resp), "out": pieces(g),
                   "text": tok.decode(g), "has_digest": DIGEST in tok.decode(g).replace(" ", ""),
                   "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{tname}] after the response ({name}) -> {rec['text']!r}  digest repeated: {rec['has_digest']}", flush=True)
print("done", flush=True)
