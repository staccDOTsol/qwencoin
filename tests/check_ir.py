"""Teacher-forced check of the circuit against the reference machine."""
import sys, time, hashlib, random
import numpy as np
from hashformer import ref, spec as S, ir
from hashformer.circuit import build, predictions

def check_trace(c, tr, check=True):
    m = ir.evaluate(c, tr.tokens, check=check)
    pred = predictions(c, m)
    bad = 0
    for i in range(len(tr.tokens) - 1):
        want = tr.tokens[i + 1] if tr.owned[i + 1] else S.IN
        if pred[i] != want:
            bad += 1
            if bad <= 8:
                print(f"  pos {i}: after {S.name(tr.tokens[i])} predicted "
                      f"{S.name(pred[i]) if pred[i] is not None else None}, want {S.name(want)}")
    return bad, len(tr.tokens) - 1

if __name__ == "__main__":
    t0 = time.time(); c = build(); print("built", c.stats(), f"{time.time()-t0:.1f}s")
    msgs = [b"abc"] if len(sys.argv) < 2 else [bytes(random.Random(i).getrandbits(8) for _ in range(n)) for i, n in enumerate(map(int, sys.argv[1:]))]
    for msg in msgs:
        tr = ref.trace(msg); assert tr.digest == hashlib.sha256(msg).digest()
        t0 = time.time(); bad, n = check_trace(c, tr)
        print(f"len {len(msg)}: {n} positions, {bad} wrong, {time.time()-t0:.1f}s")
