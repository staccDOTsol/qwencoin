"""Clean room: load the saved checkpoint with stock transformers and hash with plain generate()."""
import hashlib, sys, time, torch
from transformers import AutoModelForCausalLM
torch.set_num_threads(2)
path = sys.argv[1] if len(sys.argv) > 1 else "dist/qwensha"
SHA, END, HALT, V, F = 257, 256, 261, 512, 66048
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
print(type(model).__name__, f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)
def sha256(msg: bytes) -> bytes:
    assert len(msg) < 56  # one block: the model's own preamble, then the bytes, then generate
    with torch.no_grad():
        pre = model.generate(torch.tensor([[SHA]]), max_new_tokens=21, do_sample=False)[0].tolist()
        out = model.generate(torch.tensor([pre + list(msg) + [END]]), max_new_tokens=600,
                             do_sample=False, eos_token_id=HALT)[0].tolist()
    assert out[-1] == HALT
    pay = [(t - V) & 0xFFFF for t in out[-17:-1]]
    return b"".join(p.to_bytes(2, "big") for p in pay)
for msg in [b"abc", b"hello qwencoin"]:
    t0 = time.time(); d = sha256(msg)
    print(msg, d.hex(), d == hashlib.sha256(msg).digest(), f"{time.time()-t0:.0f}s", flush=True)
