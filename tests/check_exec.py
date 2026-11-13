"""The executor against the IR (every dim, every position), the reference trace and hashlib."""
import hashlib, random, time, sys
import numpy as np
from hashformer import ref, spec as S, ir, executor as X
from hashformer.circuit import build

c = build()
rng = random.Random(11)
# 1. residual equality with the IR, teacher-forced
tot = 0
for msg in [b"abc", b"", bytes(rng.getrandbits(8) for _ in range(150)), b"\xff" * 64]:
    tr = ref.trace(msg)
    want = ir.evaluate(c, tr.tokens, check=False)
    s = X.Session()
    for i, t in enumerate(tr.tokens):
        s.feed(t)
        got = s.residual()
        if not (got == want[i]).all():
            d = int(np.argmax(got != want[i])); raise SystemExit(f"dim {c.dims[d].name} differs at {i}: {got[d]} vs {want[i][d]}")
    tot += len(tr.tokens)
for st, n, blk in [([rng.getrandbits(32) for _ in range(8)], 2**48 - 1, bytes(rng.getrandbits(8) for _ in range(64)))]:
    tr = ref.trace_compress(st, n, blk)
    want = ir.evaluate(c, tr.tokens, check=False); s = X.Session()
    for i, t in enumerate(tr.tokens):
        s.feed(t); assert (s.residual() == want[i]).all(), i
    tot += len(tr.tokens)
print(f"executor residual == IR on all {len(c.dims)} dims at {tot} positions")
# 2. free-running: the executor generates every token it owns
tot = 0
for n in list(range(0, 130)) + [200, 257, 1000]:
    msg = bytes(rng.getrandbits(8) for _ in range(n))
    tr = ref.trace(msg)
    toks, owned = X.Session().run([S.SHA] + [S.B(x) for x in msg] + [S.END], S.HALT)
    assert toks == tr.tokens and owned == tr.owned, n
    assert X.sha256(msg) == hashlib.sha256(msg).digest(), n
    tot += len(toks)
print(f"free-running executor == reference trace on {tot} tokens")
# 3. drivers: iv, midstate compress / finish
assert X.iv() == S.IV
for _ in range(40):
    msg = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 700)))
    st, n = X.iv(), 0
    k = len(msg) // 64
    for i in range(k):
        st, n = X.compress(st, n, msg[64 * i:64 * i + 64]); assert n == i + 1
    assert X.finish(st, n, msg[64 * k:]) == hashlib.sha256(msg).digest()
print("midstate compress / finish == hashlib")
# 4. volume
count = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
t0 = time.time(); k0 = X.tokens(); nb = 0
for i in range(count):
    msg = rng.randbytes(rng.choice([0, 1, 31, 32, 55, 56, 63, 64, 65, 80, 119, 120, 200, 500]))
    assert X.sha256(msg) == hashlib.sha256(msg).digest(), msg.hex()
dt = time.time() - t0; k = X.tokens() - k0
print(f"{count} random messages == hashlib; {k} tokens in {dt:.1f}s = {k/dt:.0f} tok/s, {dt/k*1e6:.1f} us/token")
hdr = bytes.fromhex("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
t0 = time.time()
for _ in range(50): h = X.sha256d(hdr)
print("bitcoin genesis:", h[::-1].hex(), f"{(time.time()-t0)/50*1e3:.1f} ms per header double-hash")
