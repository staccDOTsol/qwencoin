"""SHA-256 for the graph tooling, computed by the model.

The transaction builder is Bitcoin's own Python test framework, which calls hashlib.sha256 for
txids, sighashes, tagged hashes and template hashes. ``install`` swaps that for an object with
the same interface that buffers bytes and hands them to the compiled weights (the same executor
the node links), so the agents' hashing is in-weights too. Everything else in hashlib passes
through untouched.
"""
import hashlib as _hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hashformer import executor as X  # noqa: E402


class ModelSha256:
    name, digest_size, block_size = "sha256", 32, 64

    def __init__(self, data: bytes = b"") -> None:
        self.state, self.nblk, self.buf = None, 0, b""  # state None: nothing compressed yet
        if data:
            self.update(data)

    def update(self, data: bytes) -> None:
        self.buf += bytes(data)
        while len(self.buf) >= 64:
            if self.state is None:
                self.state = X.iv()  # the IV is the model's own
            self.state, self.nblk = X.compress(self.state, self.nblk, self.buf[:64])
            self.buf = self.buf[64:]

    def digest(self) -> bytes:
        if self.state is None:
            return X.sha256(self.buf)  # one run from the SHA token
        return X.finish(self.state, self.nblk, self.buf)  # the model pads and prints

    def hexdigest(self) -> str:
        return self.digest().hex()

    def copy(self) -> "ModelSha256":
        c = ModelSha256()
        c.state, c.nblk, c.buf = (None if self.state is None else list(self.state)), self.nblk, self.buf
        return c


class _Shim:
    """hashlib with sha256 replaced by the model."""

    sha256 = ModelSha256

    def __getattr__(self, name):
        return getattr(_hashlib, name)

    def new(self, name, data=b"", **kw):
        return ModelSha256(data) if name.lower() in ("sha256", "sha-256") else _hashlib.new(name, data, **kw)


shim = _Shim()


def sha256(data: bytes) -> bytes:
    return ModelSha256(data).digest()


def install(*modules) -> None:
    """Make these modules' ``hashlib`` the model-backed one."""
    for m in modules:
        if hasattr(m, "hashlib"):
            m.hashlib = shim
