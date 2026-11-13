"""Which kernels on this machine run the chip exactly, and how fast.

    python tools/probe_backend.py OUT.json

Builds the stand-alone SHA-256 checkpoint (the circuit alone, 328M parameters) and runs it on
every (device, dtype) this machine offers: teacher-forced over two traces (every position's
argmax against the reference machine), then free-running through the cache. Also times a plain
matmul of the merged model's size.
"""
import copy, hashlib, json, platform, sys, time, traceback

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from hashformer import ref, spec as S  # noqa: E402
from hashformer.hf import build_model  # noqa: E402

out_path = sys.argv[1]
res = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
       "torch": torch.__version__, "threads": torch.get_num_threads(),
       "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()), "runs": []}
print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)
combos = [("cpu", torch.bfloat16), ("cpu", torch.float32)]
if res["mps"]:
    combos += [("mps", torch.bfloat16), ("mps", torch.float32)]
if torch.cuda.is_available():
    res["cuda"] = {"name": torch.cuda.get_device_name(0), "gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
    combos += [("cuda", torch.bfloat16), ("cuda", torch.float32)]


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(dev, dt):
    w = torch.randn(9216, 3840, dtype=torch.float32).to(dt).to(dev)
    x1 = torch.randn(1, 3840, dtype=torch.float32).to(dt).to(dev)
    xb = torch.randn(512, 3840, dtype=torch.float32).to(dt).to(dev)
    r = {}
    for name, x, reps in (("one_token", x1, 20), ("512_tokens", xb, 3)):
        (x @ w.T).sum().item()
        sync(dev)
        t0 = time.time()
        for _ in range(reps):
            y = x @ w.T
        y.sum().item()
        sync(dev)
        dt_ = (time.time() - t0) / reps
        r[name] = {"ms": round(dt_ * 1e3, 2), "gflops": round(2 * x.shape[0] * w.numel() / dt_ / 1e9, 1)}
    return r


t0 = time.time()
model32, c, lay = build_model(dtype=torch.float32)
print(f"stand-alone checkpoint built in {time.time() - t0:.0f}s (hidden {lay.hidden})", flush=True)
msgs = [b"abc", bytes(range(70))]
traces = [ref.trace(m) for m in msgs]

for dev, dt in combos:
    name = f"{dev}/{str(dt).split('.')[-1]}"
    run = {"backend": name}
    try:
        run["matmul"] = bench(dev, dt)
        model = copy.deepcopy(model32).to(dt).to(dev)
        slow = run["matmul"]["512_tokens"]["gflops"] < 20  # no fast kernel for this dtype here
        run["reduced"] = slow
        wrong = n = 0
        t0 = time.time()
        with torch.no_grad():
            for tr in traces[:1] if slow else traces:
                ids = torch.tensor([tr.tokens], device=dev)
                pred = model(input_ids=ids).logits[0].float().argmax(-1).cpu().tolist()
                want = [tr.tokens[i + 1] if tr.owned[i + 1] else S.IN for i in range(len(tr.tokens) - 1)]
                wrong += sum(1 for a, b in zip(pred, want) if a != b)
                n += len(want)
        sync(dev)
        run["teacher_forced"] = {"positions": n, "wrong": wrong, "seconds": round(time.time() - t0, 1)}
        # free-running: the host answers IN with the message, the model does the rest
        host = [S.SHA] + [S.B(x) for x in b"abc"] + [S.END]
        limit = 60 if slow else 600
        t0 = time.time()
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[host[0]]], device=dev), use_cache=True)
            cache, pred, i, outp = o.past_key_values, int(o.logits[0, -1].float().argmax()), 1, [host[0]]
            while len(outp) < limit:
                nxt = pred
                if pred == S.IN:
                    nxt, i = host[i], i + 1
                outp.append(nxt)
                if nxt == S.HALT:
                    break
                o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
                cache, pred = o.past_key_values, int(o.logits[0, -1].float().argmax())
        sync(dev)
        pay = [(t - S.V_BASE) & 0xFFFF for t in outp[-17:-1]]
        digest = b"".join(p.to_bytes(2, "big") for p in pay)
        run["free_running"] = {"tokens": len(outp), "digest_ok": digest == hashlib.sha256(b"abc").digest(),
                               "trace_ok": outp == traces[0].tokens[:len(outp)],
                               "ms_per_token": round((time.time() - t0) / len(outp) * 1e3, 1)}
        # how a bf16 matmul accumulates here: 1.0 means in fp32 with one rounding at the end
        if dt == torch.bfloat16:
            w = torch.randn(512, 2048).to(dt).to(dev)
            x = torch.randn(64, 2048).to(dt).to(dev)
            run["bf16_matmul_equals_fp32_rounded"] = round(float(((x.float() @ w.float().T).to(dt) == (x @ w.T)).float().mean()), 4)
        del model
    except Exception as e:  # a backend that cannot run it at all is a result too
        run["error"] = f"{type(e).__name__}: {e}"[:400]
        run["where"] = traceback.format_exc().strip().splitlines()[-3:]
    print(json.dumps(run), flush=True)
    res["runs"].append(run)
    json.dump(res, open(out_path, "w"), indent=1)
print("done", flush=True)
