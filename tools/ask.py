"""Talk to a merged checkpoint: a chat turn with the sha256 tool declared, one plain generate().

    python3 tools/ask.py MERGED_DIR "What is the SHA-256 of the bytes deadbeef?"
    DEVICE=cpu|mps  DTYPE=bfloat16|float32  MAX_NEW=900  NOTOOL=1 (do not declare the tool)

Text tokens are printed as they come; a run of chip tokens (the hash, computed by the same
weights between </tool_call> and <tool_response>) is printed as its length. Nothing here parses
the call or computes anything: the script only decodes what the model emits, and afterwards
checks the printed digest against hashlib.
"""
import hashlib, json, os, re, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.utils import logging

logging.set_verbosity_error()  # the reference-kernel notices would land in the middle of the stream

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hashformer.textspec import TOOL  # noqa: E402

model_dir, question = sys.argv[1], sys.argv[2]
dev = os.environ.get("DEVICE", "cpu")
dtype = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
vq = json.loads((Path(model_dir) / "hashformer.json").read_text())["chip_token_offset"]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(dev).eval()


class Show(BaseStreamer):
    def __init__(self, skip):
        self.skip, self.chip, self.text, self.t_chip = skip, 0, [], 0.0

    def put(self, value):
        for i in value.flatten().tolist():
            if self.skip:
                self.skip -= 1
                continue
            if i >= vq:
                if not self.chip:
                    self.t0 = time.time()
                self.chip += 1
                if self.chip % 50 == 0:
                    print(".", end="", flush=True)
                continue
            self.flush_chip()
            s = tok.decode([i])
            self.text.append(s)
            print(s, end="", flush=True)

    def flush_chip(self):
        if self.chip:
            self.t_chip = time.time() - self.t0
            print(f"[{self.chip} chip tokens, {self.t_chip:.0f}s]", end="", flush=True)
            self.text.append(f"[{self.chip} chip tokens]")
            self.chip = 0

    def end(self):
        self.flush_chip()
        print(flush=True)


x = tok.apply_chat_template([{"role": "user", "content": question}], tools=None if os.environ.get("NOTOOL") else [TOOL],
                            add_generation_prompt=True, tokenize=True)
ids = list(x["input_ids"]) if hasattr(x, "keys") else list(x)
show = Show(len(ids))
t0 = time.time()
with torch.no_grad():
    out = model.generate(torch.tensor([ids], device=dev), max_new_tokens=int(os.environ.get("MAX_NEW", "900")),
                         do_sample=False, streamer=show)
n = out.shape[1] - len(ids)
print(f"-- {n} tokens in {time.time() - t0:.0f}s on {dev}")
text = "".join(show.text)
for payload, digest in re.findall(r"<parameter=hex>\s*([0-9a-fA-F\s]*?)\s*</parameter>.*?<tool_response>\s*([0-9a-f]{64})\s*</tool_response>", text, re.S):
    want = hashlib.sha256(bytes.fromhex("".join(payload.split()))).hexdigest()
    print(f"-- sha256({''.join(payload.split()) or '(empty)'}) printed by the weights {'==' if want == digest else '!='} hashlib")
