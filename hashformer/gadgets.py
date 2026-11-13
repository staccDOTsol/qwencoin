"""Exact gadgets on the IR.

Unit terms are ``(coef, a, b)`` meaning ``coef * a * relu(b)``. Every positive gate value is a
small integer (so ``K * b`` is a bf16 value and silu is an exact relu); the negative side may be
anything at or below -1, since ``silu(K * b)`` is then exactly 0.

Values are 0/1 forms (bits, one-hots, flags) or small integer forms (nibble sums up to 255).
"""

from .ir import A_BIT, A_REF, OUT_W, REF_COLS, Chan, Expr, Graph, K, as_expr, esum


def from_bits(bits) -> Expr:
    return esum(b * (1 << i) for i, b in enumerate(bits))


class Circ(Graph):
    # --- unit terms ------------------------------------------------------------------
    def t_and(self, lits, coef=1, a=None) -> list:
        """coef * a * AND(lits) for 0/1 literals. The gate is 4 * (sum - n) + 2: +2 when all are
        set and <= -2 otherwise, so a false AND is exactly 0 even with fp32 noise on a literal."""
        a = self.one if a is None else a
        return [(coef / 2, a, 4 * esum(lits) - (4 * len(lits) - 2))]

    def t_lut(self, z, lo: int, hi: int, f, coef=1, lits=()) -> list:
        """coef * f(z) (AND lits) for an integer form z in [lo, hi]: f as a piecewise-linear
        function through its integer points, one unit relu(z - j) per change of slope."""
        n = len(lits)
        off = (1 << (hi - lo).bit_length()) * (esum(lits) - n) if n else 0
        terms = []
        if f(lo):
            terms += self.t_and(list(lits), coef * f(lo)) if n else [(coef * f(lo), self.one, self.one)]
        prev = 0
        for j in range(lo, hi):
            slope = f(j + 1) - f(j)
            if slope != prev:
                terms.append((coef * (slope - prev), self.one, z - j + off))
            prev = slope
        return terms

    def t_xor(self, bits, coef=1) -> list:
        """coef * (XOR of 2 or 3 bits): the parity of their sum."""
        s = esum(bits)
        terms = [(coef, self.one, s), (-2 * coef, self.one, s - 1)]
        if len(bits) == 3:
            terms.append((2 * coef, self.one, s - 2))
        return terms

    # --- nodes -----------------------------------------------------------------------
    def node(self, terms, name: str) -> Expr:
        return self.ffn(terms, name) if terms else Expr()

    def and_(self, lits, name: str) -> Expr:
        return self.ffn(self.t_and(lits), name)

    def or_and(self, cases, name: str) -> Expr:
        """OR of mutually exclusive ANDs (each case a list of literals)."""
        return self.ffn([t for lits in cases for t in self.t_and(lits)], name)

    # --- heads -----------------------------------------------------------------------
    def sel(self, c, taps=(1.0, 0.0, 0.0, 0.0)) -> Chan:
        """Key / query channel for a 0/1 form c (<= 0 when false): K when set, <= -K otherwise,
        so a false entry is exactly 0 after silu."""
        return Chan(as_expr(c) * (2 * K) - self.one * K, taps)

    def table(self, name: str, keys: dict, queries: dict, vheads: list, key_taps=(1.0, 0, 0, 0),
              tiny: float = 0.0) -> list:
        """A head used as tables of 0/1 rows. keys / queries: row -> 0/1 form. Each value head:
        data {col: bit form or Chan of a bit form}, ref (1 where a written row carries the
        ballast), beta (write gate), a (decay), gates {col: gate form} or gate (all columns),
        outs [(col, weight, name)] or None (every data column, weight OUT_W)."""
        specs = []
        for x in vheads:
            v = {}
            for col, e in x["data"].items():
                v[col] = Chan(e.expr * A_BIT, e.taps) if isinstance(e, Chan) else Chan(as_expr(e) * A_BIT)
            for col in REF_COLS:
                v[col] = Chan(as_expr(x["ref"]) * A_REF)
            outs = x.get("outs")
            if outs is None:
                outs = [(col, OUT_W, f"{name}[{col}]") for col in sorted(x["data"])]
            gates = x.get("gates") or {col: x.get("gate", self.one) for col, *_ in outs}
            specs.append({"v": v, "a": x.get("a", 0), "z": {col: K * as_expr(g) for col, g in gates.items()},
                          "outs": outs, "beta": x.get("beta")})
        return self.head(name, {s: self.sel(e) for s, e in queries.items()},
                         {s: self.sel(e, key_taps) for s, e in keys.items()}, specs, tiny=tiny)

    def counter(self, name: str, reset, step) -> Expr:
        """x = 2^-k at every non-reset token, k = the step tokens since the latest reset token;
        0 at reset tokens. Row 0 holds a bit that halves at every step; row 1 is a fresh ballast."""
        one = self.one
        ((x,),) = self.table(name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                             [{"data": {2: reset}, "ref": one - reset,
                               "a": 110 * as_expr(step) + 238 * as_expr(reset)}], tiny=2.0**-90)
        return x

    def thermo(self, x, reset, levels, name: str, lits=()) -> dict:
        """{m: [k > m] AND lits} from a counter output x = 2^-k (all 0 at reset tokens)."""
        one, n = self.one, len(lits)
        off = 16 * (esum(lits) - n) if n else 0
        return {m: self.ffn([(1, one - reset, 7 * one - (8 * 2**m) * x + off),
                             (-1, one - reset, 6 * one - (8 * 2**m) * x + off)], f"{name}>{m}")
                for m in levels}


def eq(th: dict, flag, v: int) -> Expr:
    """[flag AND k == v] from a thermometer gated by flag."""
    lo = flag if v == 0 else th[v - 1]
    return as_expr(lo) - th[v] if v in th else as_expr(lo)
