"""The merge as it is now against the checkpoint the long checks were run on (dist/qwen08-sha),
and a sharded checkpoint against a single-file one.

    python3 tests/check_remerge.py        -> logs/remerge.log

Qwen3.5-0.8B is merged again, written as 1 GB shards. Every tensor is then compared with the
verified checkpoint's: the bf16 ones must be the same bytes, the ones now kept in the base's
float32 must round to what was stored before. Then transformers loads the shards and its logits on
a prompt with a tool call must equal the single-file model's exactly.
"""
import json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QWEN_SHARD_GB"] = "1"
from hashformer import merge as M  # noqa: E402

REF, NEW = ROOT / "dist" / "qwen08-sha", ROOT / "work" / "qwen08-remerge"
LOG = open(ROOT / "logs" / "remerge.log", "w")


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def tensors(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.safetensors")):
        hdr, off = M.st_header(p)
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (p, off + v["data_offsets"][0], v)
    return out


def read(entry):
    p, off, v = entry
    dt = {"BF16": np.int16, "F32": np.float32}[v["dtype"]]
    return np.array(np.memmap(p, dtype=dt, mode="r", offset=off, shape=tuple(v["shape"]) or (1,)))


t0 = time.time()
shutil.rmtree(NEW, ignore_errors=True)
M.merge(ROOT / "base" / "Qwen3.5-0.8B", NEW, log=say, tools=False)
info = json.loads((NEW / "hashformer.json").read_text())
assert len(info["weight_files"]) > 1 and (NEW / "model.safetensors.index.json").exists(), info["weight_files"]
index = json.loads((NEW / "model.safetensors.index.json").read_text())
a, b = tensors(REF), tensors(NEW)
assert set(a) == set(b) == set(index["weight_map"]), set(a) ^ set(b)
same = f32 = 0
for k in sorted(a):
    x, y = read(a[k]), read(b[k])
    if b[k][2]["dtype"] == "BF16":
        assert a[k][2]["dtype"] == "BF16" and np.array_equal(x, y), k
        same += 1
    else:  # kept as the base's float32 now; the verified checkpoint held its bf16 rounding
        assert np.array_equal(torch.from_numpy(y).to(torch.bfloat16).view(torch.int16).numpy(), x), k
        f32 += 1
say(f"re-merged 0.8B in {len(info['weight_files'])} shards: {same} bf16 tensors are byte for byte the verified checkpoint's; "
    f"{f32} tensors are now the base's own float32 and round to what the verified checkpoint held ({time.time() - t0:.0f}s)")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from hashformer.textspec import TOOL  # noqa: E402

tok = AutoTokenizer.from_pretrained(REF)
ids = tok.apply_chat_template([{"role": "user", "content": "What is the SHA-256 of the bytes 616263?"}], tools=[TOOL], add_generation_prompt=True, tokenize=True)
ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
ids += tok("<tool_call>\n<function=sha256>\n<parameter=hex>\n616263\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False)["input_ids"]
out = []
for d in (REF, NEW):
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        out.append(m(input_ids=torch.tensor([ids])).logits[0].float())
    del m
assert torch.equal(out[0], out[1]), float((out[0] - out[1]).abs().max())
nxt = int(out[1][-1].argmax())
assert nxt == info["controls"]["SHA"], nxt
say(f"transformers loads the shards: logits at all {len(ids)} positions (prompt and tool call) equal the single-file model's exactly; "
    f"after </tool_call> the next token is the circuit's SHA ({time.time() - t0:.0f}s)")
shutil.rmtree(NEW)
say("OK")
