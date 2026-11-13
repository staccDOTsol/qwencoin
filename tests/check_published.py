"""A published checkpoint on this machine, used the way its card says: asked for a hash in plain
words through its own chat template, no code around it but the generation loop.

    python3 tests/check_published.py MODEL_DIR        DEVICE=cpu|cuda  DTYPE=bfloat16|float32

The model has to write the call itself, its weights have to compute the digest inside the token
stream, and the digest has to equal hashlib's. Nothing is compared with a base model here (a
machine that joins the network with a downloaded checkpoint has none): this is the check that
the checkpoint as published does its work on this machine's kernels.
"""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from rails.runtime import Session  # noqa: E402

model = Path(sys.argv[1])
info = json.loads((model / "hashformer.json").read_text())
print(f"== tests/check_published.py ({time.strftime('%Y-%m-%d %H:%M:%S')})")
t0 = time.time()
s = Session(model, None)
print(f"{model.name}: {info['parameters'] / 1e9:.2f}B parameters, hidden {info['hidden']}, on {s.dev} as {os.environ.get('DTYPE', 'bfloat16')}, "
      f"loaded in {time.time() - t0:.0f}s; its template declares {info.get('functions', '?')} functions", flush=True)
cases = [("616263", "What is the SHA-256 of the bytes 616263?"),
         ("000102030405060708090a0b0c0d0e0f", "What is the SHA-256 of the bytes 000102030405060708090a0b0c0d0e0f?")]
bad = 0
for payload, q in cases:
    s.reset()
    t = s.ask(q, max_new=1100, max_calls=2)
    want = hashlib.sha256(bytes.fromhex(payload)).hexdigest()
    mine = [(a, r) for n, a, r, by in t.calls if n == "sha256" and by == "weights"]
    ok = bool(mine) and "".join(mine[0][0].get("hex", "").split()) == payload and mine[0][1] == want
    bad += not ok
    print(f"  {q!r}\n     " + ("its weights printed " + mine[0][1] + (" == hashlib" if ok else f" != hashlib's {want}") if mine else
                              f"no hash was computed in the stream; it called {[(n, a) for n, a, r, by in t.calls] or 'nothing'}")
          + f"  ({t.chip_tokens} tokens of the hash machine among {t.new_tokens} new tokens, {t.seconds:.0f}s, "
          f"{1000 * t.seconds / max(t.new_tokens, 1):.0f} ms per token)\n     answer: {t.answer[:160]!r}", flush=True)
if bad:
    sys.exit(f"FAILED: {bad} of {len(cases)} hashes were not computed by the weights or differ from hashlib")
print("OK")
