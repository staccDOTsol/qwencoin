"""ctypes binding of the executor (exec/qsha.c)."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import spec as S

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "exec" / "libqsha.so"


def build_lib(force: bool = False) -> Path:
    """Export the circuit and compile the executor."""
    hdr = ROOT / "exec" / "qsha_circuit.h"
    if force or not hdr.exists():
        from .export import export

        hdr.write_text(export())
    src = ROOT / "exec" / "qsha.c"
    stale = not LIB.exists() or LIB.stat().st_mtime < max(src.stat().st_mtime, hdr.stat().st_mtime)
    if force or (stale and (shutil.which("gcc") or not LIB.exists())):  # a shipped library serves where there is no compiler
        subprocess.run(["gcc", "-O3", "-fPIC", "-shared", "-pthread", "-o", str(LIB), str(src), "-lm"],
                       check=True)
    return LIB


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(build_lib()))
        L.qsha_new.restype = ctypes.c_void_p
        L.qsha_free.argtypes = [ctypes.c_void_p]
        L.qsha_reset.argtypes = [ctypes.c_void_p]
        L.qsha_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.qsha_feed.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qsha_residual.argtypes = [ctypes.c_void_p]
        L.qsha_residual.restype = ctypes.POINTER(ctypes.c_double)
        L.qsha_sha256.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        L.qsha_sha256.restype = ctypes.c_long
        L.qsha_iv.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        L.qsha_compress.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64),
                                    ctypes.c_char_p]
        L.qsha_compress.restype = ctypes.c_long
        L.qsha_finish.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64, ctypes.c_char_p,
                                  ctypes.c_size_t, ctypes.c_char_p]
        L.qsha_finish.restype = ctypes.c_long
        L.qsha_tokens.restype = ctypes.c_uint64
        _lib = L
    return _lib


class Session:
    """The model's recurrent state; feed a token, get the model's next token."""

    def __init__(self) -> None:
        self.L = lib()
        self.p = self.L.qsha_new()
        self.n = self.L.qsha_ndims()

    def __del__(self) -> None:
        if getattr(self, "p", None):
            self.L.qsha_free(self.p)

    def feed(self, tok: int):
        t = self.L.qsha_feed(self.p, tok)
        return None if t < 0 else t

    def residual(self) -> np.ndarray:
        return np.ctypeslib.as_array(self.L.qsha_residual(self.p), (self.n,)).copy()

    def run(self, host: list, until: int) -> tuple:
        """Free-run: the host answers IN with its tokens. Returns (tokens, owned)."""
        toks, owned, i = [host[0]], [False], 1
        pred = self.feed(host[0])
        while True:
            if pred == S.IN:
                tok, mine = host[i], False
                i += 1
            else:
                tok, mine = pred, True
            toks.append(tok)
            owned.append(mine)
            if mine and tok == until and i >= len(host):
                return toks, owned
            pred = self.feed(tok)


def sha256(msg: bytes) -> bytes:
    out = ctypes.create_string_buffer(32)
    n = lib().qsha_sha256(msg, len(msg), out)
    if n < 0:
        raise RuntimeError("the model left the SHA-256 trace")
    return out.raw


def sha256d(msg: bytes) -> bytes:
    return sha256(sha256(msg))


def iv() -> list:
    st = (ctypes.c_uint32 * 8)()
    lib().qsha_iv(st)
    return list(st)


def compress(state: list, nblk: int, block: bytes) -> tuple:
    st = (ctypes.c_uint32 * 8)(*state)
    n = ctypes.c_uint64(nblk)
    if lib().qsha_compress(st, ctypes.byref(n), block) < 0:
        raise RuntimeError("the model left the SHA-256 trace")
    return list(st), n.value


def finish(state: list, nblk: int, tail: bytes) -> bytes:
    st = (ctypes.c_uint32 * 8)(*state)
    out = ctypes.create_string_buffer(32)
    if lib().qsha_finish(st, nblk, tail, len(tail), out) < 0:
        raise RuntimeError("the model left the SHA-256 trace")
    return out.raw


def tokens() -> int:
    return lib().qsha_tokens()
