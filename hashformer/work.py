"""Proof of work bound to the weights: the chain's work function is a pass through the MLP
matrices of Qwen3.5-0.8B (exec/qwork.h has the definition; exec/qwork.c is what the node runs).

    python -m hashformer.work build BASE_DIR qwork.bin     the table, from the model's checkpoint (needs numpy only)
    python -m hashformer.work info qwork.bin               its checksums (the chain's parameters commit to it)

Why a block's validity then depends on the model. To try one nonce a miner evaluates
264,241,152 multiply-adds over the model's own trained weights (every MLP weight of every
layer, at 16-bit fixed point), and a validator does the same once per header. No hash rate in
SHA-256 helps: the two SHA-256 evaluations per attempt are a rounding error next to the pass.
What the rule cannot do is tell which program ran: any kernel that computes the same integer
sums gets the same answer, so faster matmul hardware mines faster. It cannot be skipped, and it
cannot be done without the weights.

This module is the reference (numpy, exact integers) and the table builder.
"""
import hashlib, struct, sys
from pathlib import Path

import numpy as np

SCALE = 4096  # fixed point: a weight w is rint(4096 w)
MAGIC = b"QWRK1\0\0\0"
MODEL = "Qwen/Qwen3.5-0.8B"
# the table the qwencoin chain commits to (kernel/chainparams.cpp: consensus.qworkChecksum)
CHAIN_FNV64 = 0x33A548A33925D201
CHAIN_SHA256 = "8f53884f7727e80b86f3ddf288d3c3f0f998f1c63b8cd4f0bdad182ddd3f2c03"
M64 = (1 << 64) - 1


def quantize(w: np.ndarray) -> np.ndarray:
    """bf16 weights (as float32) to int16: round half to even, clamp."""
    return np.clip(np.rint(w.astype(np.float64) * SCALE), -32767, 32767).astype("<i2")


def _checkpoint(base_dir):
    """(text config, name -> float32 array) of a Qwen3.5 checkpoint's text model. numpy only: a
    node needs the table, not a deep-learning framework."""
    import json

    d = Path(base_dir)
    cfg = json.loads((d / "config.json").read_text())
    where = {}
    for p in sorted(d.glob("*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k.startswith("model.language_model."):
                where["model." + k[len("model.language_model."):]] = (p, 8 + n + v["data_offsets"][0], v)
    assert where, f"no Qwen3.5 text model in {d}"

    def get(name) -> np.ndarray:
        p, off, v = where[name]
        if v["dtype"] == "BF16":  # a bf16 value is the top half of the float32 with the same value
            raw = np.memmap(p, dtype="<u2", mode="r", offset=off, shape=tuple(v["shape"]))
            return (raw.astype(np.uint32) << 16).view(np.float32)
        assert v["dtype"] == "F32", v["dtype"]
        return np.array(np.memmap(p, dtype="<f4", mode="r", offset=off, shape=tuple(v["shape"])))

    return cfg.get("text_config", cfg), get


def build(base_dir, out_path, log=print) -> dict:
    cfg, get = _checkpoint(base_dir)
    L, n, m = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
    assert n % 64 == 0
    worst = 0
    with open(out_path, "wb") as f:
        f.write(MAGIC + struct.pack("<IIII", L, n, m, 0))
        for l in range(L):
            for name, shape in (("gate_proj", (m, n)), ("up_proj", (m, n)), ("down_proj", (n, m))):
                w = get(f"model.layers.{l}.mlp.{name}.weight")
                assert w.shape == shape, (l, name, w.shape)
                q = quantize(w)
                worst = max(worst, int(np.abs(q.astype(np.int64)).sum(1).max()))
                f.write(q.tobytes())
    assert worst < 2**31, "a row's sum could overflow 32 bits"
    r = info(out_path)
    r["largest_row_sum"] = worst
    log(f"{out_path}: {L} layers of {m} x {n}, {r['bytes'] / 2**20:.0f} MB, fnv64 {r['fnv64']:#018x}, sha256 {r['sha256']}")
    return r


def fnv64(buf) -> int:
    """FNV-1a over little-endian 64-bit words (the checksum the node computes on start)."""
    h = 0xCBF29CE484222325
    w = np.frombuffer(buf, dtype="<u8")
    P = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        hh = np.uint64(h)
        for i in range(0, len(w), 1 << 20):  # numpy has no fold: chunks through a tiny loop in C would be nicer
            for v in w[i:i + (1 << 20)].tolist():
                hh = np.uint64((int(hh) ^ v) * 0x100000001B3 & M64)
    return int(hh)


def info(path) -> dict:
    """Checksums of a table file. The FNV comes from the C kernel when it is built (seconds, not minutes)."""
    data = Path(path).read_bytes()
    L, n, m, _ = struct.unpack("<IIII", data[8:24])
    assert data[:8] == MAGIC and len(data) == 24 + L * 3 * n * m * 2
    try:
        f = clib().qwork_fnv64(data, len(data))
    except OSError:
        f = fnv64(data)
    return {"layers": L, "n": n, "m": m, "bytes": len(data), "fnv64": f, "sha256": hashlib.sha256(data).hexdigest()}


class Table:
    def __init__(self, path) -> None:
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")
        self.L, self.n, self.m, _ = struct.unpack("<IIII", bytes(self.raw[8:24]))
        assert bytes(self.raw[:8]) == MAGIC
        self.w = np.memmap(path, dtype="<i2", mode="r", offset=24)

    def layer(self, l: int):
        k = self.n * self.m
        a = self.w[l * 3 * k:(l + 1) * 3 * k]
        return (a[:k].reshape(self.m, self.n), a[k:2 * k].reshape(self.m, self.n), a[2 * k:].reshape(self.n, self.m))


def _rotl(x, k):
    return (x << k | x >> (64 - k)) & M64


def _mix64(z):
    z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9 & M64
    z = (z ^ z >> 27) * 0x94D049BB133111EB & M64
    return z ^ z >> 31


def expand(h0: bytes, nonce: int, n: int = 1024) -> np.ndarray:
    """The start state: n values of +-1 from the seed and the nonce (xoshiro256** from a mixed seed)."""
    w = struct.unpack("<4Q", h0)
    a = w[0] ^ (nonce << 32 | nonce)
    s = [_mix64(a), _mix64((w[1] + a) & M64), _mix64(w[2] ^ _rotl(a, 21)), _mix64((w[3] + _rotl(a, 42)) & M64)]
    if not any(s):
        s[0] = 1
    x = np.empty(n, dtype=np.int32)
    for i in range(n // 64):
        r = _rotl(s[1] * 5 & M64, 7) * 9 & M64
        t = s[1] << 17 & M64
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = _rotl(s[3], 45)
        x[64 * i:64 * i + 64] = [1 if r >> b & 1 else -1 for b in range(64)]
    return x


def sgn(v: np.ndarray) -> np.ndarray:
    return np.where(v > 0, 1, -1).astype(np.int32)


def evaluate(t: Table, h0: bytes, nonce: int, trace: list | None = None) -> bytes:
    """The pass, in exact integers. Returns the 32 folded bytes."""
    x = expand(h0, nonce, t.n)
    for l in range(t.L):
        G, U, D = t.layer(l)
        a = sgn(G.astype(np.int32) @ x) * sgn(U.astype(np.int32) @ x)
        x = sgn(D.astype(np.int32) @ a)
        if trace is not None:
            trace.append(x.copy())
    bits = np.packbits((x > 0).astype(np.uint8), bitorder="little")
    return bytes(np.bitwise_xor.reduce(bits.reshape(-1, 32), axis=0))


def pow_hash(t, header: bytes, sha256=None, kernel=None) -> bytes:
    """The proof-of-work value of an 80-byte header. sha256: the caller's SHA-256 (hashlib by
    default; the node's is the model). kernel: evaluate (numpy) or the C library's."""
    sha256 = sha256 or (lambda b: hashlib.sha256(b).digest())
    nonce = struct.unpack("<I", header[76:80])[0]
    h0 = sha256(sha256(header[:76] + bytes(4)))
    out = (kernel or evaluate)(t, h0, nonce)
    return sha256(bytes(a ^ b for a, b in zip(out, h0)))


# ------------------------------------------------------------------ the C kernel, for cross-checks and mining tools
_clib = None


def clib():
    global _clib
    if _clib is None:
        import ctypes, shutil, subprocess
        root = Path(__file__).resolve().parent.parent / "exec"
        so, src = root / "libqwork.so", root / "qwork.c"
        if not so.exists() or (so.stat().st_mtime < src.stat().st_mtime and shutil.which("gcc")):
            subprocess.run(["gcc", "-O3", "-fPIC", "-shared", "-o", str(so), str(src)], check=True)
        L = ctypes.CDLL(str(so))
        L.qwork_fnv64.restype = ctypes.c_uint64
        L.qwork_fnv64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        L.qwork_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        L.qwork_eval.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p]
        _clib = L
    return _clib


class CTable:
    """The table as the node's kernel maps it."""

    def __init__(self, path) -> None:
        import ctypes

        class T(ctypes.Structure):
            _fields_ = [("layers", ctypes.c_int), ("n", ctypes.c_int), ("m", ctypes.c_int), ("data", ctypes.c_void_p),
                        ("bytes", ctypes.c_size_t), ("fnv", ctypes.c_uint64), ("map", ctypes.c_void_p)]
        self.t = T()
        rc = clib().qwork_open(str(path).encode(), ctypes.byref(self.t))
        if rc:
            raise OSError(f"qwork_open({path}) = {rc}")
        self.ref = ctypes.byref(self.t)
        self.fnv, self.n = self.t.fnv, self.t.n

    def evaluate(self, h0: bytes, nonce: int) -> bytes:
        import ctypes
        out = ctypes.create_string_buffer(32)
        clib().qwork_eval(self.ref, h0, nonce, out)
        return out.raw


def c_kernel(ct: CTable, h0: bytes, nonce: int) -> bytes:
    return ct.evaluate(h0, nonce)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "build":
        r = build(sys.argv[2], sys.argv[3])
        if (r["fnv64"], r["sha256"]) != (CHAIN_FNV64, CHAIN_SHA256):
            sys.exit(f"this is not the chain's table (fnv64 {CHAIN_FNV64:#018x}, sha256 {CHAIN_SHA256}): was it built from {MODEL}?")
        print("this is the chain's table")
    elif len(sys.argv) >= 3 and sys.argv[1] == "info":
        r = info(sys.argv[2])
        print(r, "(the chain's table)" if (r["fnv64"], r["sha256"]) == (CHAIN_FNV64, CHAIN_SHA256) else "(NOT the chain's table)")
    else:
        sys.exit(__doc__)
