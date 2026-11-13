"""Circuit IR for stock Qwen3.5 layers: SwiGLU units and Gated DeltaNet heads.

Every value is a scalar dim of the residual stream; linear forms over dims (``Expr``) are free,
because each sublayer starts with a linear projection. A circuit is a list of two node types,
one per sublayer kind of a Qwen3.5 linear-attention layer:

``FFN``   out = sum of units ``coef * a * relu(b)``. In the weights a unit is
          ``silu(K b) * (a / K)``; silu is an exact relu at every value the circuit produces
          (0, >= 18, or <= -104).
``Head``  one Gated DeltaNet key head and its value heads. q / k / v channels are linear forms
          through the layer's 4-tap causal conv and silu. A one-hot key with beta = 1 makes the
          delta rule an exact overwrite of one state row, so a head is a table of rows; a
          one-hot or two-hot query reads rows back (two-hot: their sum). ``a`` picks the decay
          per token (keep / halve / clear), ``beta`` gates the write, ``z`` gates each output
          column. Rows carry a ballast in two columns that dominates the head's output RMSNorm,
          so a read comes back as an exactly known rescaling of the row.

A chip token carries a large ballast in a few residual dims, which makes every RMSNorm on it a
division by a known power of two; the emitter folds that into the weights. ``ONE`` is that
ballast seen through the norm. All of this follows the numerics qwenputer established for
Qwen3.5; the graph, the evaluator and the gadgets here are written for this circuit.

``evaluate`` runs tokens through the graph in exact float64 and, with ``check``, asserts that
every tensor the real layers materialize is a bf16 value (so bf16 hardware computes exactly
these numbers), that every silu input is in the exact range, and that keys, queries, decays and
write gates have the shapes the delta rule needs.
"""

from dataclasses import dataclass, field

import numpy as np

ONE = 0  # dim 0: the normed ballast, 1 on chip tokens and 0 elsewhere
K = 128.0  # silu gate scale
A_REF = 2.0**24  # ballast of a state row (in REF_COLS)
A_BIT = 2.0**7  # a set bit in a state row
REF_COLS = (0, 1)
HD = 128  # key / value head dim of Qwen3.5
DECAY = {0.0: 1.0, 110.0: 0.5, 238.0: 0.0}  # a -> exp(g), with dt_bias = -110 and A_log = 0
OUT_W = 2.0**7  # out_proj weight that turns a one-hot read of a set bit into 1
RESIDUAL_MAX = 2.0**28  # larger residual values would move the norms


class Expr:
    """A linear form over dims: {dim: coefficient}."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict | None = None) -> None:
        self.terms = terms or {}

    def __add__(self, o):
        o = as_expr(o)
        t = dict(self.terms)
        for d, c in o.terms.items():
            v = t.get(d, 0) + c
            if v:
                t[d] = v
            else:
                t.pop(d, None)
        return Expr(t)

    __radd__ = __add__

    def __neg__(self):
        return Expr({d: -c for d, c in self.terms.items()})

    def __sub__(self, o):
        return self + (-as_expr(o))

    def __rsub__(self, o):
        return as_expr(o) + (-self)

    def __mul__(self, s):
        return Expr({d: c * s for d, c in self.terms.items()} if s else {})

    __rmul__ = __mul__

    def key(self) -> tuple:
        return tuple(sorted(self.terms.items()))


def as_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    return Expr({ONE: x} if x else {})


def esum(xs) -> Expr:
    out: dict = {}
    for x in xs:
        for d, c in as_expr(x).terms.items():
            out[d] = out.get(d, 0) + c
    return Expr({d: c for d, c in out.items() if c})


@dataclass
class Dim:
    name: str
    kind: str  # "one" | "token" | "ffn" | "head"
    level: int  # sublayer after which the value exists (0: the embedding)


@dataclass
class FFN:
    name: str
    terms: list  # (coef, a: Expr, b: Expr)
    out: int
    level: int
    into: bool = False  # adds onto an existing dim


@dataclass
class Chan:
    expr: Expr
    taps: tuple = (1.0, 0.0, 0.0, 0.0)  # weights on x_t, x_{t-1}, x_{t-2}, x_{t-3}


@dataclass
class VHead:
    v: dict  # value column -> Chan (columns REF_COLS hold the ballast)
    a: Expr  # decay selector (see DECAY)
    z: dict  # value column -> output gate pre-activation
    out: list  # (value column, residual dim, weight)
    beta: Expr | None = None  # write gate: 1, or an integer <= 0; None: always 1


@dataclass
class Head:
    name: str
    q: dict  # state row -> Chan
    k: dict
    vheads: list
    level: int
    tiny: float = 0.0  # outputs below this may round (a decay counter far past its range)


@dataclass
class Target:
    """Per-layer capacity of the host architecture (Qwen3.5-4B's linear-attention layers)."""

    key_heads: int = 16
    value_heads: int = 32
    intermediate: int = 9216


class Graph:
    def __init__(self, target: Target | None = None) -> None:
        self.target = target or Target()
        self.dims: list[Dim] = [Dim("one", "one", 0)]
        self.features: dict[int, object] = {}  # dim -> fn(token id array) -> values
        self.text_features: dict[int, np.ndarray] = {}  # dim -> value per text token (merged circuits)
        self.n_text = 0  # text tokens the circuit reads (0: a circuit of chip tokens only)
        self.nodes: list = []
        self.one = Expr({ONE: 1})
        self.units: dict[int, int] = {}
        self.heads: dict[int, int] = {}

    # --- dims
    def _dim(self, name: str, kind: str, level: int) -> int:
        self.dims.append(Dim(name, kind, level))
        return len(self.dims) - 1

    def feature(self, name: str, fn=None, text=None) -> Expr:
        """A dim set by the embedding: fn maps an array of chip token ids to values (None: 0 on
        chip tokens); text is its value on every text token (None: 0)."""
        d = self._dim(name, "token", 0)
        if fn is not None:
            self.features[d] = fn
        if text is not None:
            self.text_features[d] = np.asarray(text, dtype=np.float64)
        return Expr({d: 1})

    def level(self, *exprs) -> int:
        return max((self.dims[d].level for e in exprs for d in e.terms), default=0)

    def _place(self, lvl: int, odd: bool, need: int, used: dict, cap: int) -> int:
        sub = lvl + 1
        while (sub % 2 == 1) != odd or used.get(sub, 0) + need > cap:
            sub += 1
        used[sub] = used.get(sub, 0) + need
        return sub

    # --- nodes
    def ffn(self, terms, name: str = "ffn", into: Expr | None = None, min_level: int = 0) -> Expr:
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c]
        lvl = max(self.level(*(e for _, a, b in terms for e in (a, b))), min_level)
        if into is not None:
            (d,) = into.terms
            lvl = max(lvl, self.dims[d].level)
        sub = self._place(lvl, False, len(terms), self.units, self.target.intermediate)
        if into is not None:
            self.dims[d].level = sub
        else:
            d = self._dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, d, sub, into is not None))
        return Expr({d: 1})

    def head(self, name: str, q: dict, k: dict, vheads: list, tiny: float = 0.0,
             min_level: int = 0) -> list:
        """vheads: dicts with v {col: Chan}, a, z {col: Expr}, outs [(col, weight)] and
        optionally beta. Returns, per value head, the list of its output dims (one per out)."""
        r = self.target.value_heads // self.target.key_heads
        assert 1 <= len(vheads) <= r, (name, len(vheads))

        def chans(ch):
            return {i: c if isinstance(c, Chan) else Chan(as_expr(c)) for i, c in ch.items()}

        q, k = chans(q), chans(k)
        exprs = [c.expr for ch in (q, k) for c in ch.values()]
        built = []
        for x in vheads:
            v = chans(x["v"])
            a = as_expr(x.get("a", 0))
            z = {i: as_expr(e) for i, e in x["z"].items()}
            beta = None if x.get("beta") is None else as_expr(x["beta"])
            exprs += [c.expr for c in v.values()] + [a, *z.values()] + ([] if beta is None else [beta])
            built.append((v, a, z, beta, x["outs"], x.get("into")))
        lvl = max(self.level(*exprs), min_level)
        sub = self._place(lvl, True, 1, self.heads, self.target.key_heads)
        vhs, ret = [], []
        for i, (v, a, z, beta, outs, into) in enumerate(built):
            if into is None:  # one new dim per distinct name in outs: (col, weight, name)
                names, dims = {}, []
                for o in outs:
                    nm = o[2] if len(o) > 2 else f"{name}.v{i}[{o[0]}]"
                    if nm not in names:
                        names[nm] = self._dim(nm, "head", sub)
                    dims.append(names[nm])
            else:
                dims = []
                for e in into:
                    (d,) = e.terms
                    self.dims[d].level = max(self.dims[d].level, sub)
                    dims.append(d)
            vhs.append(VHead(v, a, z, [(o[0], d, o[1]) for o, d in zip(outs, dims)], beta))
            seen, uniq = set(), []
            for d in dims:
                if d not in seen:
                    seen.add(d)
                    uniq.append(Expr({d: 1}))
            ret.append(uniq)
        self.nodes.append(Head(name, q, k, vhs, sub, tiny))
        return ret

    def n_layers(self) -> int:
        return (max((n.level for n in self.nodes), default=0) + 1) // 2

    def stats(self) -> dict:
        units = {}
        for n in self.nodes:
            if isinstance(n, FFN):
                s = units.setdefault(n.level, set())
                s.update((a.key(), b.key()) for _, a, b in n.terms)
        heads = [n for n in self.nodes if isinstance(n, Head)]
        return {
            "layers": self.n_layers(),
            "dims": len(self.dims),
            "key_heads": len(heads),
            "value_heads": sum(len(h.vheads) for h in heads),
            "units": sum(len(s) for s in units.values()),
            "units_per_mlp": {(lv // 2): len(s) for lv, s in sorted(units.items())},
            "heads_per_attn": {(lv + 1) // 2: c for lv, c in sorted(self.heads.items())},
        }


# --- exactness -------------------------------------------------------------------------


class ExactnessError(AssertionError):
    pass


def bf16(x) -> np.ndarray:
    """Round to the nearest bf16 value (ties to even), as float64."""
    f = np.asarray(x, dtype=np.float32).copy()
    u = f.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return f.astype(np.float64)


def is_bf16(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return (bf16(x) == x) & np.isfinite(x)


def silu_ok(x) -> np.ndarray:
    """silu(x) == relu(x) bit for bit, in fp32 and bf16."""
    return (x == 0) | (x >= 18) | (x <= -104)


def require(ok, what: str, x) -> None:
    ok = np.asarray(ok)
    if not ok.all():
        x = np.asarray(x)
        t = np.unravel_index(int(np.argmin(ok)), ok.shape)
        raise ExactnessError(f"{what}: bad value {x[t] if x.shape == ok.shape else x!r} at {t}")


# --- evaluation ------------------------------------------------------------------------


def _compile(e: Expr):
    idx = np.fromiter(e.terms.keys(), dtype=np.int64, count=len(e.terms))
    coef = np.array([float(c) for c in e.terms.values()])
    return idx, coef


def ev(e: Expr, m: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(m.shape[0])
    idx, coef = _compile(e)
    return m[:, idx] @ coef


def conv(c: Chan, m: np.ndarray, check: bool) -> np.ndarray:
    """silu(conv(projection)): the q / k / v channel as the layer computes it."""
    x = ev(c.expr, m)
    if check:
        require(is_bf16(x), "channel projection", x)
    out = np.zeros(len(x))
    for lag, w in enumerate(c.taps):
        if w:
            out[lag:] += w * x[: len(x) - lag]
    if check:
        require(is_bf16(out) & silu_ok(out), "conv output (silu input)", out)
    return np.maximum(out, 0.0)


def features(g: Graph, toks: np.ndarray, check: bool = True) -> np.ndarray:
    """Residual after the embedding: token features and ONE. Chip tokens are 0 .. VOCAB-1; in a
    merged circuit text token q is TEXT0 + q and carries ONE too; anything else is a zero row."""
    from . import spec as S

    n = len(toks)
    m = np.zeros((n, len(g.dims)))
    chip = (toks >= 0) & (toks < S.VOCAB)
    text = (toks >= S.TEXT0) & (toks < S.TEXT0 + g.n_text)
    m[:, ONE] = chip | text
    ids = np.where(chip, toks, 0)
    qs = np.where(text, toks - S.TEXT0, 0)
    for d in set(g.features) | set(g.text_features):
        col = np.zeros(n)
        if d in g.features:
            col = np.where(chip, np.asarray(g.features[d](ids), dtype=np.float64), 0.0)
        if d in g.text_features:
            col = np.where(text, g.text_features[d][qs], col)
        if check:
            require(is_bf16(col), f"token feature {g.dims[d].name}", col)
        m[:, d] = col
    return m


def evaluate(g: Graph, tokens, check: bool = True, upto: int | None = None) -> np.ndarray:
    """Residual stream after every node: (n_tokens, n_dims), exact in float64."""
    toks = np.asarray(tokens, dtype=np.int64)
    m = features(g, toks, check)
    chip = m[:, ONE] > 0
    for node in g.nodes:
        if upto is not None and node.level > upto:
            continue
        try:
            if isinstance(node, FFN):
                eval_ffn(node, m, check)
            else:
                eval_head(node, m, chip, check)
        except ExactnessError as e:
            raise ExactnessError(f"{node.name} (sublayer {node.level}): {e}") from None
    if check and m.shape[0]:
        big = np.abs(m[:, 1:]).max(0)
        if big.max() > RESIDUAL_MAX:
            d = int(big.argmax()) + 1
            raise ExactnessError(f"{g.dims[d].name}: residual value {big[d - 1]!r} too large")
    return m


def eval_ffn(node: FFN, m: np.ndarray, check: bool) -> None:
    ideal = np.zeros(m.shape[0])
    hw = np.zeros(m.shape[0])
    for c, a, b in node.terms:
        bv, av = ev(b, m), ev(a, m)
        ideal += float(c) * av * np.maximum(bv, 0.0)
        if check:
            gate = bf16(K * bv)
            require(silu_ok(gate), "gate outside silu's exact range", gate)
            up = bf16(av / K)
            hw += float(c) * bf16(np.maximum(gate, 0.0) * up)
    if check:
        require(bf16(hw) == ideal, "bf16 output differs from exact", hw - ideal)
    if node.into:
        ideal = m[:, node.out] + ideal
        if check:
            require(is_bf16(ideal), "residual add", ideal)
    m[:, node.out] = ideal


def eval_head(node: Head, m: np.ndarray, chip: np.ndarray, check: bool) -> None:
    n = m.shape[0]
    qi, ki = list(node.q), list(node.k)
    q = np.stack([conv(node.q[s], m, check) for s in qi], 1) if qi else np.zeros((n, 0))
    k = np.stack([conv(node.k[s], m, check) for s in ki], 1) if ki else np.zeros((n, 0))
    if check:
        nz = (k > 0).sum(1)
        require(nz <= 1, "key must be one-hot", nz)
        require(~((nz > 0) & ~chip), "key on a non-chip token", nz)
        qn = (q > 0).sum(1)
        require(qn <= 2, "query must be one-hot or two-hot", qn)
        qmax = q.max(1) if q.shape[1] else np.zeros(n)
        require(((q == 0) | (q == qmax[:, None])).all(1), "query entries must be equal", qmax)
    has_key = k.any(1) if k.shape[1] else np.zeros(n, dtype=bool)
    key_row = np.array([ki[j] for j in k.argmax(1)]) if k.shape[1] else np.zeros(n, dtype=int)
    has_q = q.any(1) if q.shape[1] else np.zeros(n, dtype=bool)
    qrows = [[qi[j] for j in np.nonzero(q[t])[0]] if has_q[t] else [] for t in range(n)]
    ncol = HD
    for vh in node.vheads:
        v = np.zeros((n, ncol))
        for col, c in vh.v.items():
            v[:, col] = conv(c, m, check)
        a = ev(vh.a, m)
        z = {col: ev(e, m) for col, e in vh.z.items()}
        beta = np.ones(n) if vh.beta is None else ev(vh.beta, m)
        writes = has_key & (beta > 0)
        if check:
            require(is_bf16(a) & np.isin(a, list(DECAY)), "decay selector a", a)
            require((beta == 1.0) | ((beta <= 0) & (beta == np.round(beta))), "write gate beta", beta)
            for col, zz in z.items():
                require(is_bf16(zz) & silu_ok(zz), f"gate z[{col}]", zz)
            written = v[writes]
            if len(written):
                data = np.delete(written, REF_COLS, 1)
                require(np.isin(data, [0.0, A_BIT]), "state row data column", data)
                ref = written[:, list(REF_COLS)]
                require((ref == 0) | (ref == A_REF), "state row ballast column", ref)
        decay = np.vectorize(DECAY.get, otypes=[float])(a) if n else np.zeros(0)
        state: dict = {}
        out = np.zeros((n, ncol))
        for t in np.nonzero((decay != 1.0) | writes | has_q)[0]:
            if decay[t] == 0.0:
                state.clear()
            elif decay[t] == 0.5:
                state = {s: row * 0.5 for s, row in state.items()}
            if writes[t]:
                state[key_row[t]] = v[t].copy()
            if has_q[t]:
                o = np.zeros(ncol)
                for s in qrows[t]:
                    if s in state:
                        o += state[s]
                ref = (o[list(REF_COLS)] ** 2).sum()
                if ref == 0:
                    if check:
                        require(o == 0, "read without ballast", o)
                    continue
                if check:
                    data = (np.delete(o, REF_COLS) ** 2).sum()
                    require(np.array([data <= ref * 2.0**-26]), "data not absorbed by ballast",
                            np.array([data / ref]))
                out[t] = o / np.sqrt(ref / ncol)
        for col, d, w in vh.out:
            zz = z.get(col, np.zeros(n))
            val = out[:, col] * np.maximum(zz, 0.0) * w
            if check:
                small = np.abs(out[:, col]) < node.tiny
                require(is_bf16(out[:, col]) | small, f"normed column {col}", out[:, col])
                require(is_bf16(val) | small, f"head output column {col}", val)
            m[:, d] += val
        if check:
            for d in {d for _, d, _ in vh.out}:
                require(is_bf16(m[:, d]) | (np.abs(m[:, d]) < node.tiny), "head output sum", m[:, d])
