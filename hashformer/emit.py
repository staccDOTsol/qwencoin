"""Turn the circuit graph into the weights of stock Qwen3.5 layers.

Residual layout of a chip token: [ballast x m] [IR dims 1..n-1] [unused]. Chip tokens carry
B = 2^40 in every ballast dim. With m * B^2 / hidden a power of 4, every RMSNorm on a chip token
divides by exactly B * sqrt(m / hidden), so a residual value x reaches each projection as x * c
with c = sqrt(hidden / m) / B, a power of two; projection weights carry 1 / c. The IR constant
ONE is the ballast seen through the norm (B * c).

No weight is learned. Each one is a coefficient of the circuit times a power of two, and each
is a bf16 value (checked), so the checkpoint is exact in the dtype Qwen3.5 ships in.
"""

import math
from dataclasses import dataclass

import torch

from .ir import FFN, HD, ONE, Expr, Graph, Head, K


@dataclass
class Layout:
    hidden: int
    m: int  # ballast dims
    log2_ballast: int = 40
    off: int = 0  # first chip dim (a merged model keeps the host's dims below it)

    @property
    def ballast(self) -> float:
        return 2.0**self.log2_ballast

    @property
    def scale(self) -> float:
        """c: what a residual value 1 on a chip token becomes after the layer norm."""
        log4 = math.log(self.m * self.ballast**2 / self.hidden, 4)
        assert abs(log4 - round(log4)) < 1e-12, "m * B^2 / hidden must be a power of 4"
        return 2.0 ** -round(log4)

    def dim(self, d: int) -> int:
        return self.off + self.m + d - 1


def pick_layout(n_dims: int) -> Layout:
    """Smallest hidden = m * 4^k (a multiple of 64) holding m >= 3 ballast dims and the IR dims."""
    best = None
    for m in range(3, 16):
        k = 0
        while m * 4**k < m + n_dims - 1 or (m * 4**k) % 64:
            k += 1
        if best is None or m * 4**k < best.hidden:
            best = Layout(hidden=m * 4**k, m=m)
    return best


def bf16_parts(x: float, n: int) -> list:
    """x as a sum of at most n bf16 values."""
    parts = []
    for _ in range(n):
        if x == 0:
            break
        p = torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).double().item()
        parts.append(p)
        x -= p
    if x != 0:
        raise ValueError(f"constant needs more than {n} bf16 parts")
    return parts


def row(e: Expr, lay: Layout, out: torch.Tensor, w: float = 1.0) -> None:
    """Add w * e to a projection row. A constant rides on the ballast dims, split so that each
    weight is a bf16 value (the matmul adds the parts exactly)."""
    c = lay.scale
    for d, coef in e.terms.items():
        if d == ONE:
            for i, p in enumerate(bf16_parts(w * float(coef) / (lay.ballast * c), lay.m)):
                out[lay.off + i] += p
        else:
            out[lay.dim(d)] += w * float(coef) / c


def check_bf16(name: str, t: torch.Tensor) -> None:
    b = t.to(torch.bfloat16).to(t.dtype)
    bad = ((b != t) | ~torch.isfinite(b)).nonzero()
    if len(bad):
        i = tuple(bad[0].tolist())
        raise ValueError(f"{name}: weight {t[i].item()!r} at {i} is not a bf16 value")


def mlp_units(g: Graph, level: int) -> tuple:
    """The distinct units (a, b) of an MLP sublayer and, per unit, its (output dim, coef) list.
    Terms with the same a and b are one hidden unit feeding several outputs."""
    index, units, outs = {}, [], []
    for n in g.nodes:
        if isinstance(n, FFN) and n.level == level:
            for coef, a, b in n.terms:
                key = (a.key(), b.key())
                if key not in index:
                    index[key] = len(units)
                    units.append((a, b))
                    outs.append({})
                o = outs[index[key]]
                o[n.out] = o.get(n.out, 0.0) + float(coef)
    return units, outs


def emit_layers(g: Graph, lay: Layout, tc: dict, prefix: str = "model.", first: int = 0,
                keep: list | None = None, ballast_gate: Expr | None = None, clear: tuple = ()) -> dict:
    """State dict of the chip layers. With ``keep``, the last MLP also clears every chip dim
    except those (and removes the ballast): the exit, after which only the output code is left.

    For a merged model (merge.py): the ballast is removed only where the 0/1 form
    ``ballast_gate`` is set (on text tokens: the host's layers then see their usual input, and
    are muted on the ballast that chip tokens keep), and ``clear`` lists extra residual columns
    the exit zeroes, each with the 0/1 form that says where."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    inter, ksz = tc["intermediate_size"], tc["linear_conv_kernel_dim"]
    r = nv // nk
    f32 = torch.float32
    one = Expr({ONE: 1})
    n_layers = g.n_layers()
    out = {}
    for L in range(n_layers):
        p = f"{prefix}layers.{first + L}."
        qkv = torch.zeros(2 * nk * HD + nv * HD, h, dtype=f32)
        convw = torch.zeros(2 * nk * HD + nv * HD, 1, ksz, dtype=f32)
        zw = torch.zeros(nv * HD, h, dtype=f32)
        bw = torch.zeros(nv, h, dtype=f32)
        aw = torch.zeros(nv, h, dtype=f32)
        ow = torch.zeros(h, nv * HD, dtype=f32)
        heads = [n for n in g.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        # a key head of the target carries r value heads; a head of the circuit with more value
        # heads than that takes several key heads with the same key and query
        slots = [(n, n.vheads[i:i + r]) for n in heads for i in range(0, len(n.vheads), r)]
        assert len(slots) <= nk, (L, len(slots), nk)
        for j, (n, vheads) in enumerate(slots):
            for base, chans in ((j * HD, n.q), (nk * HD + j * HD, n.k)):
                for i, ch in chans.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lg, w in enumerate(ch.taps):
                        convw[base + i, 0, ksz - 1 - lg] = w
            for x, vh in enumerate(vheads):
                vi = j * r + x  # value head x of key head j
                base = 2 * nk * HD + vi * HD
                for i, ch in vh.v.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lg, w in enumerate(ch.taps):
                        convw[base + i, 0, ksz - 1 - lg] = w
                # beta = sigmoid(128) = 1, or sigmoid(256 g - 128): exactly 1 or exactly 0
                row(Expr({ONE: 128}) if vh.beta is None else 256 * vh.beta - 128 * one, lay, bw[vi])
                row(vh.a, lay, aw[vi])
                for col, e in vh.z.items():
                    row(e, lay, zw[vi * HD + col])
                for col, d, w in vh.out:
                    ow[lay.dim(d), vi * HD + col] += w
        units, outs = mlp_units(g, 2 * L + 2)
        units = [(a, b, o) for (a, b), o in zip(units, outs)]
        if keep is not None and L == n_layers - 1:
            units += [(Expr({d: 1}), one, {d: -1.0}) for d in range(1, len(g.dims)) if d not in keep]
        last = keep is not None and L == n_layers - 1
        n_exit = lay.m if last else 0
        assert len(units) + n_exit + (len(clear) if last else 0) <= inter, (L, len(units), inter)
        gate = torch.zeros(inter, h, dtype=f32)
        up = torch.zeros(inter, h, dtype=f32)
        down = torch.zeros(h, inter, dtype=f32)
        for u, (a, b, o) in enumerate(units):
            row(b, lay, gate[u], K)
            row(a, lay, up[u], 1 / K)
            for d, coef in o.items():
                down[lay.dim(d), u] = coef
        for i in range(n_exit):  # the ballast goes too (where ballast_gate says so)
            u = len(units) + i
            row(one if ballast_gate is None else ballast_gate, lay, gate[u], K)
            up[u, lay.off + i] = (lay.ballast / K) / (lay.ballast * lay.scale)
            down[lay.off + i, u] = -1.0
        for i, (col, where) in enumerate(clear if last else ()):  # x -> x - relu(where) * x
            u = len(units) + n_exit + i
            row(where, lay, gate[u], K)
            up[u, col] = (1 / K) / lay.scale
            down[col, u] = -1.0
        layer = {
            "input_layernorm.weight": torch.zeros(h, dtype=f32),
            "post_attention_layernorm.weight": torch.zeros(h, dtype=f32),
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_z.weight": zw,
            "linear_attn.in_proj_a.weight": aw,
            "linear_attn.in_proj_b.weight": bw,
            "linear_attn.conv1d.weight": convw,
            "linear_attn.dt_bias": torch.full((nv,), -110.0, dtype=f32),
            "linear_attn.A_log": torch.zeros(nv, dtype=f32),
            "linear_attn.norm.weight": torch.ones(HD, dtype=f32),
            "linear_attn.out_proj.weight": ow,
            "mlp.gate_proj.weight": gate,
            "mlp.up_proj.weight": up,
            "mlp.down_proj.weight": down,
        }
        for name, t in layer.items():
            check_bf16(p + name, t)
            out[p + name] = t
    return out


def embedding(g: Graph, lay: Layout, n_vocab: int, n_chip: int, rows: tuple | None = None) -> torch.Tensor:
    """Embedding rows: chip token t is row t (ballast plus its features); other rows are zero.
    ``rows = (a, b)``: only rows a .. b-1."""
    import numpy as np

    a, b = rows or (0, n_vocab)
    e = torch.zeros(b - a, lay.hidden, dtype=torch.float32)
    n = max(0, min(b, n_chip) - a)
    e[:n, lay.off:lay.off + lay.m] = lay.ballast
    ids = np.arange(a, a + n)
    for d, fn in g.features.items():
        e[:n, lay.dim(d)] = torch.tensor(np.asarray(fn(ids), dtype=np.float32))
    check_bf16("embedding", e)
    return e


def nonzeros(sd: dict, prefix: str = "model.layers.") -> int:
    return sum(int((t != 0).sum()) for k, t in sd.items()
               if k.startswith(prefix) and "norm" not in k and "dt_bias" not in k)
