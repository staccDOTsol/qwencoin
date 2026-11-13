"""Export the circuit's live structure for the executor (qsha.c).

    python -m hashformer.export exec/qsha_circuit.h

The executor runs the same arithmetic as the layers, restricted to what is not zero: each MLP
unit as two sparse dot products and a relu, each head as its channels, one row write and a
one- or two-row read with the ballast norm. It is checked against the IR on every dim at every
position (tests/check_exec.py), and the IR is checked against the stock HF modules.

A chip token's features are linear in its payload bits within a family, so the embedding is
exported as a base row per family plus one delta row per payload bit (verified for every token).
"""

import sys

import numpy as np

from . import spec as S
from .circuit import build
from .emit import mlp_units
from .ir import DECAY, FFN, HD, ONE, Expr, Head, features


class Pool:
    """Flat storage of linear forms: (offset, length) into index / coefficient arrays."""

    def __init__(self) -> None:
        self.idx: list[int] = []
        self.coef: list[float] = []
        self.memo: dict = {}

    def add(self, e) -> tuple:
        key = e.key()
        if key not in self.memo:
            self.memo[key] = (len(self.idx), len(e.terms))
            for d, c in sorted(e.terms.items()):
                self.idx.append(d)
                self.coef.append(float(c))
        return self.memo[key]


def one_lag(taps) -> int:
    nz = [i for i, w in enumerate(taps) if w]
    assert len(nz) == 1 and taps[nz[0]] == 1.0, taps
    return nz[0]


def feature_rows(c) -> tuple:
    """Sparse rows: per family a base and per-bit deltas, then one row per control token."""
    fam = [("B", S.B_BASE, 8), ("V", S.V_BASE, 16), ("F", S.F_BASE, 16)]
    full = features(c, np.arange(S.VOCAB), check=False)
    rows, layout = [], []

    def sparse(v):
        nz = np.nonzero(v)[0]
        return [(int(d), float(v[d])) for d in nz if d != ONE]  # the executor sets ONE itself

    for name, base, nbits in fam:
        b0 = full[base]
        deltas = np.stack([full[base + (1 << i)] - b0 for i in range(nbits)])
        p = np.arange(1 << nbits)
        bits = (p[:, None] >> np.arange(nbits)[None, :]) & 1
        assert (b0[None, :] + bits @ deltas == full[base:base + (1 << nbits)]).all(), name
        layout.append((len(rows), nbits))
        rows.append(sparse(b0))
        rows += [sparse(d) for d in deltas]
    ctl0 = len(rows)
    rows += [sparse(full[S.CTRL_BASE + i]) for i in range(len(S.CONTROL_NAMES))]
    return rows, layout, ctl0


def export(c=None) -> str:
    c = c or build()
    pool = Pool()
    into_level = {}  # dim -> levels at which an FFN adds onto it
    for n in c.nodes:
        if isinstance(n, FFN) and n.into:
            into_level.setdefault(n.out, []).append(n.level)

    subs = []  # (kind, first, count) per sublayer in order; kind 0: heads, 1: mlp
    units, outs_dim, outs_coef, bases, ucst = [], [], [], [], []
    heads, vheads, chans, zexprs, vouts = [], [], [], [], []
    state_cells = row_cells = 0
    top = max(n.level for n in c.nodes)
    for level in range(1, top + 1):
        if level % 2 == 0:
            us, os_ = mlp_units(c, level)
            first, bfirst, bmap = len(units), len(bases), {}
            for (a, b), o in zip(us, os_):
                # gate = base + const: units that threshold the same sum share its dot product
                base = Expr({d: v for d, v in b.terms.items() if d != ONE})
                if base.key() not in bmap:
                    bmap[base.key()] = len(bases) - bfirst
                    bases.append(pool.add(base))
                ao, an = pool.add(a)
                oo = len(outs_dim)
                for d, coef in sorted(o.items()):
                    outs_dim.append(d)
                    outs_coef.append(coef)
                units.append((bmap[base.key()], ao, an, oo, len(o)))
                ucst.append(float(b.terms.get(ONE, 0)))
            if len(units) > first:
                subs.append((1, first, len(units) - first, bfirst, len(bases) - bfirst))
            continue
        first = len(heads)
        for n in c.nodes:
            if not (isinstance(n, Head) and n.level == level):
                continue

            def chan_list(ch):
                off = len(chans)
                for slot, cc in sorted(ch.items()):
                    lg = one_lag(cc.taps)
                    if lg:  # a lagged read sees the final residual of an earlier token
                        for d in cc.expr.terms:
                            assert all(lv < level for lv in into_level.get(d, [])), c.dims[d].name
                    chans.append((slot, *pool.add(cc.expr), lg))
                return off, len(ch)

            qo, qn = chan_list(n.q)
            ko, kn = chan_list(n.k)
            nrows = max(list(n.q) + list(n.k)) + 1
            vfirst = len(vheads)
            for vh in n.vheads:
                vo, vn = chan_list(vh.v)
                ncols = max(vh.v) + 1
                zs, zmap = len(zexprs), {}
                oo = len(vouts)
                for col, d, w in vh.out:
                    zk = vh.z[col].key()
                    if zk not in zmap:
                        zmap[zk] = len(zexprs) - zs
                        zexprs.append(pool.add(vh.z[col]))
                    vouts.append((col, d, float(w), zmap[zk]))
                assert len(zexprs) - zs <= 16
                a = pool.add(vh.a) if vh.a.terms else (0, 0)
                beta = pool.add(vh.beta) if vh.beta is not None else (-1, 0)
                vheads.append((vo, vn, *a, *beta, zs, len(zexprs) - zs, oo, len(vh.out), nrows,
                               ncols, state_cells, row_cells))
                state_cells += nrows * ncols
                row_cells += nrows
            heads.append((qo, qn, ko, kn, vfirst, len(n.vheads)))
        if len(heads) > first:
            subs.append((0, first, len(heads) - first, 0, 0))

    rows, layout, ctl0 = feature_rows(c)
    f_off, f_idx, f_val = [], [], []
    for r in rows:
        f_off.append(len(f_idx))
        f_idx += [d for d, _ in r]
        f_val += [v for _, v in r]
    f_off.append(len(f_idx))

    def arr(ctype, name, vals, fmt="{}"):
        body = ",".join(fmt.format(v) for v in vals)
        return f"static const {ctype} {name}[{max(len(vals), 1)}] = {{{body}}};\n"

    def flat(tuples):
        return [x for t in tuples for x in t]

    s = c.stats()
    out = "/* Generated by hashformer.export from the compiled SHA-256 circuit. Do not edit. */\n"
    out += f"/* {s['layers']} layers, {s['dims']} dims, {s['key_heads']} key heads, " \
           f"{s['value_heads']} value heads, {len(units)} MLP units */\n"
    for k, v in [("NDIMS", len(c.dims)), ("NCODE", len(c.keep)), ("NSUB", len(subs)),
                 ("NUNITS", len(units)), ("NHEADS", len(heads)), ("NVHEADS", len(vheads)),
                 ("STATE", state_cells), ("NROWS", sum(v[10] for v in vheads)),
                 ("CTL0", ctl0), ("VOCAB", S.VOCAB), ("NCTL", len(S.CONTROL_NAMES)),
                 ("MAXCOLS", max(v[11] for v in vheads)), ("MAXBASE", max(x[4] for x in subs))]:
        out += f"#define QS_{k} {v}\n"
    for nm, tok in S.CONTROLS.items():
        out += f"#define QS_TOK_{nm} {tok}\n"
    out += f"#define QS_V_BASE {S.V_BASE}\n#define QS_F_BASE {S.F_BASE}\n"
    out += f"#define QS_DECAY_HALF {[k for k, v in DECAY.items() if v == 0.5][0]}\n"
    out += f"#define QS_DECAY_CLEAR {[k for k, v in DECAY.items() if v == 0.0][0]}\n"
    out += f"#define QS_HD {HD}\n"
    out += arr("int", "qs_code", c.keep)
    out += arr("int", "qs_eidx", pool.idx)
    out += arr("double", "qs_ecoef", pool.coef, "{!r}")
    out += arr("int", "qs_sub", flat(subs))
    out += arr("int", "qs_unit", flat(units))
    out += arr("double", "qs_ucst", ucst, "{!r}")
    out += arr("int", "qs_base", flat(bases))
    out += arr("int", "qs_odim", outs_dim)
    out += arr("double", "qs_ocoef", outs_coef, "{!r}")
    out += arr("int", "qs_head", flat(heads))
    out += arr("int", "qs_vhead", flat(vheads))
    out += arr("int", "qs_chan", flat(chans))
    out += arr("int", "qs_zexpr", flat(zexprs))
    out += arr("int", "qs_vout", flat((col, d, z) for col, d, _, z in vouts))
    out += arr("double", "qs_voutw", [w for _, _, w, _ in vouts], "{!r}")
    out += arr("int", "qs_foff", f_off)
    out += arr("int", "qs_fidx", f_idx)
    out += arr("double", "qs_fval", f_val, "{!r}")
    out += arr("int", "qs_fam", flat(layout))
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "exec/qsha_circuit.h"
    text = export()
    with open(path, "w") as f:
        f.write(text)
    print(f"{path}: {len(text)} bytes")


if __name__ == "__main__":
    main()
