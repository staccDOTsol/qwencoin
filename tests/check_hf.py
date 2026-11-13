"""Stock Qwen3_5ForCausalLM vs the IR: residual equality on every dim, and free-running hashes."""
import sys, time, hashlib, resource
import torch
from hashformer import ref, spec as S
from hashformer.hf import build_model, check_residual, hf_sha256

torch.set_num_threads(2)
mode = sys.argv[1] if len(sys.argv) > 1 else "residual"
dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[2] if len(sys.argv) > 2 else "bf16"]
t0 = time.time()
if mode == "residual":
    model, c, lay = build_model(dtype=dt, exit=False)
    print(f"model built {time.time()-t0:.0f}s: hidden {lay.hidden}, m {lay.m}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, nnz {model.nonzero_weights}, "
          f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB", flush=True)
    for msg in [b"abc", bytes(range(70))]:
        tr = ref.trace(msg)
        for m_ in ("prefill", "decode", "mixed"):
            t0 = time.time()
            n = check_residual(model, c, lay, tr.tokens, m_)
            print(f"len {len(msg)} {m_}: residual == IR on all {len(c.dims)-1} dims at {n} positions "
                  f"({time.time()-t0:.0f}s)", flush=True)
else:
    model, c, lay = build_model(dtype=dt, exit=True)
    print(f"model built {time.time()-t0:.0f}s", flush=True)
    for msg in [b"abc", b"", b"qwencoin: a computer is a hash is a model", bytes(range(64)), bytes(range(121))]:
        t0 = time.time()
        d, s = hf_sha256(model, msg)
        tr = ref.trace(msg)
        ok = d == hashlib.sha256(msg).digest() and s.tokens == tr.tokens and s.owned == tr.owned
        print(f"len {len(msg)}: {d.hex()} {'OK' if ok else 'MISMATCH'} {len(s.tokens)} tokens, "
              f"{sum(s.owned)} generated, {time.time()-t0:.0f}s", flush=True)
