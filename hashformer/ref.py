"""Reference model of the SHA-256 machine: the executable spec the circuit is checked against.

``Machine.feed`` applies one token to the state exactly as the DeltaNet heads do (same counters,
same rows, same lags); ``Machine.predict`` is the token the circuit must emit next, or IN where
the host speaks. ``trace`` drives it: the host only ever supplies message bytes and END (or, for
a resume, the midstate records). Nothing here is used at run time by the model or the executor.
"""

from dataclasses import dataclass, field

from . import spec as S
from .spec import B, F, V

M32 = 0xFFFFFFFF


def rotr(x: int, n: int) -> int:
    return (x >> n | x << (32 - n)) & M32


def big_sigma0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def big_sigma1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def small_sigma0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def small_sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
def ch(e, f, g): return (e & f) ^ (~e & g & M32)
def maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


STAGES = ["INIT", "RESM", "BYTES", "EARLY", "LATE", "CVS", "NS"]
STAGE_TOKEN = {S.SHA: "INIT", S.RES: "RESM", S.BLK: "BYTES", S.R0: "EARLY", S.R16: "LATE",
               S.RC: "CVS", S.RN: "NS"}
T_RESET = set(STAGE_TOKEN)  # tokens that reset the record counter
NB_RESET = {S.SHA, S.RES, S.BLK}  # tokens that reset the byte counter


@dataclass
class Machine:
    stage: str | None = None
    T: int = 0  # records completed since the stage began
    J: int = 0  # V tokens of the current record so far
    NB: int = 0  # bytes of the current block so far
    ended: bool = False  # END seen in this message
    last: bool = False  # the current block is the one that carries the length
    N: int = 0  # blocks completed (the N register)
    LEN: int = 0  # the message's bit length, captured at END
    A: list = field(default_factory=lambda: [0] * 68)  # A[t] at index t + 4 (index 0..3: d,c,b,a)
    E: list = field(default_factory=lambda: [0] * 68)  # E[t] at index t + 4 (index 0..3: h,g,f,e)
    W: list = field(default_factory=lambda: [0] * 64)
    lag: list = field(default_factory=list)  # payloads of the last 4 tokens, newest first

    # ------------------------------------------------------------------ state update
    def feed(self, tok: int) -> None:
        fam, p = S.family(tok)
        t_prev, stage_prev = self.T, self.stage
        if tok in T_RESET:
            self.T = 0
        elif fam == "F":
            self.T += 1
        self.J = self.J + 1 if fam == "V" else 0
        is_byte = fam == "B" or tok == S.END
        if tok in NB_RESET:
            self.NB = 0
        elif is_byte:
            self.NB += 1
        byte = 0x80 if tok == S.END else p
        self.lag = [byte if is_byte else p] + self.lag[:3]

        if tok in STAGE_TOKEN:
            self.stage = STAGE_TOKEN[tok]
        if tok in (S.SHA, S.RES):
            self.ended = self.last = False
        if tok == S.BLK:
            self.last = self.ended
        if tok == S.END:
            self.ended = True
            self.last = self.NB <= 56
            self.LEN = (self.N << 9 | (self.NB - 1) << 3) & (2**64 - 1)

        lg = self.lag + [0] * 4
        word = (lg[1] << 16 | lg[0]) & M32
        if is_byte and self.NB % 4 == 0 and self.NB:
            self.W[self.NB // 4 - 1] = lg[3] << 24 | lg[2] << 16 | lg[1] << 8 | lg[0]
        if fam == "V":
            if self.stage == "LATE" and self.J == 2:
                self.W[self.T + 16] = word
            if self.stage == "EARLY" and self.J == 2:
                self.E[self.T + 4] = word
            if self.stage == "LATE" and self.J == 4:
                self.E[self.T + 20] = word
        if fam == "F":
            if stage_prev == "EARLY":
                self.A[t_prev + 4] = word
            elif stage_prev == "LATE":
                self.A[t_prev + 20] = word
            elif stage_prev in ("CVS", "INIT", "RESM") and t_prev < 8:
                if t_prev < 4:
                    self.A[3 - t_prev] = word
                else:
                    self.E[7 - t_prev] = word
            elif stage_prev in ("INIT", "RESM") and t_prev == 8 or stage_prev == "NS":
                self.N = lg[3] << 48 | lg[2] << 32 | lg[1] << 16 | lg[0]

    # ------------------------------------------------------------------ next token
    def _half(self, value: int, hi: bool, final: bool) -> int:
        p = value >> 16 & 0xFFFF if hi else value & 0xFFFF
        return F(p) if final else V(p)

    def predict(self) -> int:
        st, T, J = self.stage, self.T, self.J
        if st == "INIT":
            if T < 8:
                return self._half(S.IV[T], J == 0, J == 1)
            if T == 8:
                return F(0) if J == 3 else V(0)
            return S.BLK
        if st == "RESM":
            return S.IN if T < 9 else S.BLK
        if st == "BYTES":
            if self.NB == 64:
                return S.R0
            if not self.ended:
                return S.IN
            if self.last and self.NB >= 56:
                return B(self.LEN >> 8 * (63 - self.NB))
            return B(0)
        if st in ("EARLY", "LATE"):
            if st == "EARLY" and T == 16:
                return S.R16
            if st == "LATE" and T == 48:
                return S.RC
            t = T if st == "EARLY" else T + 16
            A, E, W = self.A, self.E, self.W
            n_w = 0 if st == "EARLY" else 2  # V tokens before the E half words
            if J < n_w:
                w = (small_sigma1(W[t - 2]) + W[t - 7] + small_sigma0(W[t - 15]) + W[t - 16]) & M32
                return self._half(w, J == 0, False)
            d, h = A[t], E[t]
            e1, e2, e3 = E[t + 3], E[t + 2], E[t + 1]
            t1 = h + big_sigma1(e1) + ch(e1, e2, e3) + S.K[t] + W[t]
            if J < n_w + 2:
                return self._half((d + t1) & M32, J == n_w, False)
            a1, a2, a3 = A[t + 3], A[t + 2], A[t + 1]
            a = (t1 + big_sigma0(a1) + maj(a1, a2, a3)) & M32
            return self._half(a, J == n_w + 2, J == n_w + 3)
        if st == "CVS":
            if T == 8:
                return S.HALT if self.last else S.RN
            ring, i = (self.A, T) if T < 4 else (self.E, T - 4)
            return self._half((ring[3 - i] + ring[67 - i]) & M32, J == 0, J == 1)
        if st == "NS":
            if T == 1:
                return S.BLK
            n = (self.N + 1) & (2**64 - 1)
            p = n >> 16 * (3 - J) & 0xFFFF
            return F(p) if J == 3 else V(p)
        raise RuntimeError("no stage: the stream must begin with SHA or RES")


@dataclass
class Trace:
    tokens: list[int]
    owned: list[bool]  # owned[i]: token i was emitted by the model (else supplied by the host)
    digest: bytes | None = None
    state: list[int] | None = None  # the last chaining value the model emitted
    nblk: int | None = None  # blocks completed, when the trace ends at a block boundary


def _drive(m: Machine, host: list[int], until: int, limit: int = 10**9) -> Trace:
    """Run until the model emits ``until``. The host answers IN with its tokens, in order."""
    toks, owned, i = [], [], 0
    first = host[0]
    toks.append(first)
    owned.append(False)
    m.feed(first)
    i = 1
    while len(toks) < limit:
        p = m.predict()
        if p == S.IN:
            if i >= len(host):
                raise RuntimeError("the model asks for input but the host has none")
            tok, mine = host[i], False
            i += 1
        else:
            tok, mine = p, True
        toks.append(tok)
        owned.append(mine)
        m.feed(tok)
        if mine and tok == until and i >= len(host):
            break
    return Trace(toks, owned)


def _cv(tokens: list[int], end: int) -> list[int]:
    """The 8 chaining words spelled by the 16 value tokens that end at index ``end``."""
    pay = [S.family(t)[1] for t in tokens[end - 16:end]]
    return [pay[2 * i] << 16 | pay[2 * i + 1] for i in range(8)]


def trace(msg: bytes) -> Trace:
    """SHA-256 of ``msg`` from scratch: the host sends SHA, the bytes and END."""
    m = Machine()
    tr = _drive(m, [S.SHA] + [B(x) for x in msg] + [S.END], S.HALT)
    tr.state = _cv(tr.tokens, len(tr.tokens) - 1)
    tr.digest = b"".join(w.to_bytes(4, "big") for w in tr.state)
    return tr


def resume_host(state: list[int], nblk: int) -> list[int]:
    host = [S.RES]
    for w in state:
        host += [V(w >> 16), F(w)]
    host += [V(nblk >> 48), V(nblk >> 32), V(nblk >> 16), F(nblk)]
    return host


def trace_compress(state: list[int], nblk: int, block: bytes) -> Trace:
    """One compression from a midstate: RES, the chaining value, the block count, 64 bytes; the
    model answers with the rounds, the new chaining value and the new block count."""
    assert len(block) == 64
    m = Machine()
    host = resume_host(state, nblk) + [B(x) for x in block]
    tr = _drive(m, host, S.BLK)
    # the stream ends ... RC cv x8 RN n BLK
    tr.state = _cv(tr.tokens, len(tr.tokens) - 6)
    tr.nblk = m.N
    return tr


def trace_finish(state: list[int], nblk: int, tail: bytes) -> Trace:
    """Finish a message from a midstate: the remaining < 64 bytes and END; the model pads."""
    assert len(tail) < 64
    m = Machine()
    host = resume_host(state, nblk) + [B(x) for x in tail] + [S.END]
    tr = _drive(m, host, S.HALT)
    tr.state = _cv(tr.tokens, len(tr.tokens) - 1)
    tr.digest = b"".join(w.to_bytes(4, "big") for w in tr.state)
    return tr


def sha256(msg: bytes) -> bytes:
    return trace(msg).digest
