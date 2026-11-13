"""What SHA-256 is inside the node binary: nothing but the circuit.

A native SHA-256 needs its 64 round constants. This scans the binary for them (and for the other
numbers a hasher could lean on) and says where each hit lives.

  round constants   allowed only as the top half of SHA-512's 64-bit constants (SHA-512 is native)
  initial value     allowed only in SelfTest(), which compares the model's output against it
  tagged midstates  libsecp256k1 hard-codes eleven precomputed SHA-256 states; the fork removes them

    python3 tests/check_binary.py [bin/qwencoind]
    python3 tests/check_binary.py --ord [bin/ord]

ord also links a TLS stack and the `sha2` crate, which carry their own SHA-256. With --ord the
scan attributes every round constant to the symbol that holds it and fails if any is in
rust-bitcoin's hashes or in libsecp256k1: those two hash and sign everything ord does on chain.
"""
import struct, subprocess, sys

ORD = "--ord" in sys.argv
args = [a for a in sys.argv[1:] if a != "--ord"]
path = args[0] if args else ("bin/ord" if ORD else "bin/qwencoind")
b = open(path, "rb").read()

K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
     0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
     0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
     0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
     0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
     0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
     0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
MIDSTATES = {  # first word of each state libsecp256k1 hard-codes upstream
    "BIP0340/nonce": 0x46615b35, "BIP0340/aux": 0x24dd3219, "BIP0340/challenge": 0x9cecba11,
    "secp256k1_ellswift_encode": 0xd1a6524b, "secp256k1_ellswift_create": 0xd29e1bf5,
    "bip324_ellswift_xonly_ecdh": 0x8c12d730, "MuSig/aux": 0xa19e884b, "MuSig/nonce": 0x07101b64,
    "MuSig/noncecoef": 0x2c7d5a45, "KeyAgg list": 0xb399d5e0, "KeyAgg coefficient": 0x6ef02c5a}


def hits(word):
    out, i = [], b.find(struct.pack("<I", word))
    while i >= 0:
        out.append(i)
        i = b.find(struct.pack("<I", word), i + 1)
    return out


def sections():
    rows = []
    for line in subprocess.run(["readelf", "-S", "-W", path], capture_output=True, text=True).stdout.splitlines():
        f = line.replace("[", " ").replace("]", " ").split()
        if len(f) >= 6 and f[0].isdigit() and f[1].startswith("."):
            rows.append((f[1], int(f[3], 16), int(f[4], 16), int(f[5], 16)))  # name, addr, offset, size
    return rows


def symbol_at(off, secs, syms):
    for name, addr, o, size in secs:
        if o <= off < o + size:
            va = addr + off - o
            best = max((s for s in syms if s[0] <= va), default=None)
            return f"{name}: {best[1]}" if best else name
    return "?"


secs = sections()
nm = subprocess.run(["nm", "-n", "-C", "--defined-only", path], capture_output=True, text=True).stdout
syms = [(int(l.split()[0], 16), " ".join(l.split()[2:])) for l in nm.splitlines() if len(l.split()) >= 3]

# 1. round constants: every hit must be the high half of one of SHA-512's 64-bit constants
def icbrt(n):
    x = 1 << -(-n.bit_length() // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            return x
        x = y


primes = [p for p in range(2, 410) if all(p % q for q in range(2, int(p ** 0.5) + 1))][:80]
K512 = [icbrt(p << 192) & (2 ** 64 - 1) for p in primes]  # frac(cbrt(p)) * 2^64
assert [k >> 32 for k in K512[:64]] == K  # SHA-256's constants are the top halves of the first 64
loose = []
for k, k512 in zip(K, K512):
    for off in hits(k):
        if b[off - 4:off + 4] != struct.pack("<Q", k512):
            loose.append((hex(k), hex(off), symbol_at(off, secs, syms)))
print(f"{path}")
if ORD:
    by = {}
    for _, _, where in loose:
        name = where.split(": ", 1)[-1]
        name = "K256 (ring / aws-lc: TLS)" if name.startswith("K256") else "unnamed constant table (the sha2 crate's, one per version)" if name.startswith("anon.") else name
        by[name] = by.get(name, 0) + 1
    print("  SHA-256 round constants, by the symbol that holds them:")
    for name, n in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d} x {name[:90]}")
    chain = [n for n in by if "bitcoin_hashes" in n or "secp256k1" in n]
    found = {t: len(hits(w)) for t, w in MIDSTATES.items() if hits(w)}
    print(f"  in rust-bitcoin's hashes or libsecp256k1: {len(chain)}", chain if chain else "")
    print(f"  hard-coded tagged-hash midstates: {len(found)}", found if found else "")
    print("  executor linked:", any(" qsha_compress" in l for l in nm.splitlines()))
    ok = not chain and not found and any(" qsha_compress" in l for l in nm.splitlines())
    print("OK: ord's chain hashing and signing have no native SHA-256" if ok else "FOUND NATIVE SHA-256 ON THE CHAIN PATH")
    sys.exit(0 if ok else 1)
print(f"  SHA-256 round constants outside SHA-512's constants: {len(loose)}", loose[:4] if loose else "")

# 2. the initial value
where = {}
for off in hits(IV[0]):
    if b[off - 4:off] == struct.pack("<I", 0xf3bcc908):
        kind = "SHA-512 initial value (64-bit)"
    elif list(struct.unpack("<8I", b[off:off + 32])) == IV:
        kind = "8-word table in " + symbol_at(off, secs, syms)
    else:
        kind = "immediate in " + symbol_at(off, secs, syms)
    where[kind] = where.get(kind, 0) + 1
print("  SHA-256 initial value:")
for kind, n in where.items():
    print(f"    {n} x {kind}")

# 3. libsecp256k1's precomputed tagged-hash midstates
found = {t: len(hits(w)) for t, w in MIDSTATES.items() if hits(w)}
print(f"  hard-coded tagged-hash midstates: {len(found)}", found if found else "")

ok = not loose and not found and all("SelfTest" in k or "SHA-512" in k for k in where)
print("OK" if ok else "FOUND NATIVE SHA-256 MATERIAL")
sys.exit(0 if ok else 1)
