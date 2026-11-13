"""Teacher-forced sweep: every padding case, long messages, midstate resume, block counters."""
import hashlib, random, time
import numpy as np
from hashformer import ref, spec as S, ir
from hashformer.circuit import build
from tests.check_ir import check_trace

c = build(); print(c.stats())
rng = random.Random(7); tot = bad = 0; t0 = time.time()
def run(tr, what):
    global tot, bad
    b, n = check_trace(c, tr); tot += n; bad += b
    if b: print("FAIL", what, b)
for n in list(range(0, 130)) + [183, 184, 185, 191, 192, 193, 255, 256, 257, 640, 1000]:
    msg = bytes(rng.getrandbits(8) for _ in range(n))
    tr = ref.trace(msg); assert tr.digest == hashlib.sha256(msg).digest()
    run(tr, f"len {n}")
for msg in [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", b"\xff" * 200, b"\x00" * 200]:
    run(ref.trace(msg), msg[:8])
print(f"fresh messages: {tot} positions, {bad} wrong, {time.time()-t0:.0f}s")
# midstate resume with large block counts (carries across every 16-bit chunk of N)
for nblk in [0, 1, 0xFFFF, 0x10000, 0xFFFFFFFF, 0x1_0000_0000, 0xFFFF_FFFF_FFFF, 2**55 - 2, 0x1234_5678_9ABC]:
    st = [rng.getrandbits(32) for _ in range(8)]
    blk = bytes(rng.getrandbits(8) for _ in range(64))
    tr = ref.trace_compress(st, nblk, blk); assert tr.nblk == nblk + 1
    run(tr, f"compress nblk={nblk:x}")
    for tail in (0, 1, 55, 56, 63):
        run(ref.trace_finish(st, nblk, blk[:tail]), f"finish nblk={nblk:x} tail={tail}")
# extreme words: all-ones / all-zero state and block (max carries)
for st, blk in [([0xFFFFFFFF] * 8, b"\xff" * 64), ([0] * 8, b"\x00" * 64), ([0x80000000] * 8, b"\x80" * 64)]:
    run(ref.trace_compress(st, 5, blk), "extreme")
print(f"total: {tot} positions, {bad} wrong, {time.time()-t0:.0f}s")
