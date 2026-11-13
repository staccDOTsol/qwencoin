"""The chain's proof of work, recomputed by the stock checkpoint.

Loads dist/qwensha with plain transformers (no code from this repo), takes block headers from a
running node, and double-hashes each one with ``generate()``. The digest the model prints must be
the block hash the node reports (which the node computed in its executor), must equal hashlib's,
and must meet the header's own target.

    python3 tests/check_hf_chain.py [heights...]        default: 0 and the tip of node 1
"""
import hashlib, json, os, subprocess, sys, time
import torch
from transformers import AutoModelForCausalLM

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
END, SHA, IN, HALT, V = 256, 257, 260, 261, 512  # hashformer.json: controls
CLI = ["bin/qwencoin-cli", "-datadir=run/n1"]
model = AutoModelForCausalLM.from_pretrained("dist/qwensha", dtype=torch.bfloat16).eval()
count = [0]


def sha256(msg: bytes) -> bytes:
    """The host sends SHA, then answers each IN with the bytes of the block the model is waiting
    for (or with the tail and END). Everything else is the model's own greedy output."""
    toks, i = [SHA], 0
    while True:
        with torch.no_grad():
            out = model.generate(torch.tensor([toks]), max_new_tokens=2000, do_sample=False,
                                 eos_token_id=[IN, HALT])[0].tolist()
        count[0] += len(out) - len(toks)
        if out[-1] == HALT:
            pay = [(t - V) & 0xFFFF for t in out[-17:-1]]
            return b"".join(p.to_bytes(2, "big") for p in pay)
        assert out[-1] == IN, out[-1]  # the model asks at the start of a block
        chunk = list(msg[i:i + 64])
        i += len(chunk)
        toks = out[:-1] + chunk + ([END] if len(chunk) < 64 else [])


def cli(*a):
    return subprocess.run(CLI + [str(x) for x in a], capture_output=True, text=True, check=True).stdout.strip()


heights = [int(a) for a in sys.argv[1:]] or [0, int(cli("getblockcount"))]
for h in heights:
    bh = cli("getblockhash", h)
    hdr = bytes.fromhex(cli("getblockheader", bh, "false"))
    bits = json.loads(cli("getblockheader", bh))["bits"]
    target = (int(bits, 16) & 0xFFFFFF) << 8 * ((int(bits, 16) >> 24) - 3)
    t0 = time.time()
    d = sha256(sha256(hdr))
    ok = d[::-1].hex() == bh and d == hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
    pow_ok = int.from_bytes(d, "little") <= target
    print(f"height {h}: generate() -> {d[::-1].hex()}  node+hashlib equal: {ok}  meets target {bits}: {pow_ok}"
          f"  [{time.time() - t0:.0f}s]", flush=True)
    assert ok and pow_ok
print(f"OK: {len(heights)} headers, {count[0]} generated tokens")
