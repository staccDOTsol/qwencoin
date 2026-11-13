"""The work function (exec/qwork.c, hashformer/work.py) on the real table.

    python3 tests/check_work.py dist/qwork.bin [N=4000]

  kernel     the C kernel the node links against the exact numpy reference
  avalanche  one bit of the nonce or of the seed flipped: how much of the result changes
  collapse   distinct inputs must end in distinct states (a sign network could contract)
  whitening  the proof-of-work value SHA256(out ^ h0): bit balance, leading-byte histogram,
             and how often it falls under the chain's minimum target (1 in 256)
"""
import hashlib, struct, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hashformer import work as W  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
inf = W.info(path)
print(f"table: {inf['layers']} layers, {inf['m']} x {inf['n']}, {inf['bytes']} bytes, fnv64 {inf['fnv64']:#018x}, sha256 {inf['sha256']}")
t, ct = W.Table(path), W.CTable(path)
assert ct.fnv == inf["fnv64"]
rng = np.random.default_rng(7)


def rnd():
    return bytes(rng.integers(0, 256, 32, dtype=np.uint8)), int(rng.integers(0, 2**32))


bad = 0
for i in range(40):
    h0, nonce = rnd()
    bad += W.evaluate(t, h0, nonce) != ct.evaluate(h0, nonce)
h = bytes(32)
bad += W.evaluate(t, h, 0) != ct.evaluate(h, 0)
bad += W.evaluate(t, b"\xff" * 32, 2**32 - 1) != ct.evaluate(b"\xff" * 32, 2**32 - 1)
print(f"kernel: C against the numpy reference on 42 inputs (edge seeds included): {bad} differ")
assert bad == 0

bits = lambda b: np.unpackbits(np.frombuffer(b, dtype=np.uint8))
d_nonce, d_seed, d_state = [], [], []
for i in range(60):
    h0, nonce = rnd()
    a = ct.evaluate(h0, nonce)
    d_nonce.append(int((bits(a) != bits(ct.evaluate(h0, nonce ^ 1 << int(rng.integers(32))))).sum()))
    k = int(rng.integers(256))
    h1 = bytearray(h0); h1[k // 8] ^= 1 << k % 8
    d_seed.append(int((bits(a) != bits(ct.evaluate(bytes(h1), nonce))).sum()))
tr0, tr1 = [], []
h0, nonce = rnd()
W.evaluate(t, h0, nonce, tr0)
W.evaluate(t, h0, nonce ^ 1, tr1)
per_layer = [int((a != b).sum()) for a, b in zip(tr0, tr1)]
print(f"avalanche: one nonce bit flips {np.mean(d_nonce):.1f} of the 256 output bits (min {min(d_nonce)}), one seed bit {np.mean(d_seed):.1f} (min {min(d_seed)})")
print(f"           inside the pass, nonce and nonce^1: state bits that differ after layers 1, 2, 3, 12, 24: "
      f"{[per_layer[i] for i in (0, 1, 2, 11, 23)]} of {t.n}")
assert min(d_nonce) > 80 and min(d_seed) > 80

t0 = time.time()
h0 = hashlib.sha256(b"qwencoin work check").digest()
outs = [ct.evaluate(h0, n) for n in range(N)]
dt = (time.time() - t0) / N
pows = [hashlib.sha256(bytes(a ^ b for a, b in zip(o, h0))).digest() for o in outs]
print(f"speed: {dt * 1e3:.1f} ms per attempt on one thread ({1 / dt:.0f} attempts/s)")
print(f"collapse: {N} nonces under one seed give {len(set(outs))} distinct results")
assert len(set(outs)) == N
ob = np.stack([bits(o) for o in outs]).mean(0)
print(f"raw output bits (before whitening): balance per bit {ob.min():.3f} .. {ob.max():.3f} (binomial 3-sigma: {0.5 - 1.5 / N**0.5:.3f} .. {0.5 + 1.5 / N**0.5:.3f})")
pb = np.stack([bits(p) for p in pows]).mean(0)
lead = np.bincount([p[31] for p in pows], minlength=256)  # the node reads the value little-endian: byte 31 leads
chi = float(((lead - N / 256) ** 2 / (N / 256)).sum())
hits = sum(1 for p in pows if p[31] == 0)
print(f"whitening: bit balance of the proof-of-work value {pb.min():.3f} .. {pb.max():.3f}; leading byte chi-square {chi:.0f} on 255 degrees of freedom "
      f"(99% range 205 .. 310); under the minimum target {hits} of {N} (expected {N / 256:.1f})")
assert 180 < chi < 340
print("OK")
