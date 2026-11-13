"""Merge the SHA-256 chip into a Qwen3.5 checkpoint: one stock ``Qwen3_5ForCausalLM`` that is
the language model and the hasher, and in which the language model calls the hasher itself.

    python -m hashformer.merge BASE_DIR --out DIR        BASE_DIR: a dense Qwen3.5 checkpoint (0.8B .. 27B)

Layout (qwenputer's "chip first, then Qwen", rebuilt for this circuit):

  residual   [ Qwen's dims | ballast x m | circuit dims | U ]      width H' = m * 4^k
  layers     the 5 chip layers, then Qwen's layers
  vocabulary Qwen's tokens, then the chip tokens (chip token t is id V_qwen + t)

Every token carries the ballast through the chip layers, so every RMSNorm there is an exact
division by a power of two, on text as on chip tokens. A text token is Qwen's embedding row in
Qwen's dims plus the few features the circuit reads from text (textspec.py); a chip token is
zero in Qwen's dims. Nothing is trained, and no weight of Qwen's is changed except as listed:

  chip layers on text   Their weights are zero on Qwen's dims: Qwen's embedding passes them bit
                        for bit. The circuit reads the text (the tool call) and writes only
                        its own dims. The exit clears those dims and, on a text token, the
                        ballast: Qwen's layers receive their usual input.
  Qwen's layers on chip A chip token keeps the ballast. Three of Qwen's gates are open at 0 and
                        get a column on the ballast that sets them to -1024 (the mute): the
                        DeltaNet write gate and decay (Qwen's state passes through a trace
                        untouched) and the attention output gate. Chip positions still are
                        keys of Qwen's attention; a column on the ballast gives each of them a
                        key pointing against the head's mean text query (the anti-sink, fitted
                        on the base), so they do not soak up attention.
  who speaks            Where the circuit owns the next token it leaves an output code of size
                        2^-20; elsewhere the code is exactly 0. The final norm passes Qwen's
                        dims, the code (gain 2^52) and U (gain 2^84). After a chip token the
                        head sees only the code: the token whose code agrees on all 18 bits
                        scores 72, any other 64 or less, every text token without a code 0.
                        After a text token the circuit owns (a firing </tool_call>, the answer
                        being printed) the code outweighs Qwen's logits by nine orders of
                        magnitude. After any other text token every chip row scores
                        -2^7 * U: far below any text logit.
  the output head       Tied to the embedding where the base ties it (0.8B, 2B, 4B). Where the base
                        has an output matrix of its own (9B, 27B) the merged checkpoint has one
                        too: the base's rows in Qwen's dims, the same code and owner columns.
  the wider norm        RMSNorm now averages over H' dims instead of Qwen's H. Qwen's norm gains
                        are multiplied by sqrt(H / H'), rounded to bf16. This is the only place
                        where text is not bit-identical to the base; tests/check_merge.py
                        measures it.

The writer streams: no tensor of the merged checkpoint is held beyond the one being written.
"""
import argparse, gc, json, math, os, shutil, struct
from pathlib import Path

import numpy as np
import torch

from . import spec as S
from .circuit import build
from .emit import Layout, check_bf16, embedding, emit_layers
from .ir import Target
from .textspec import MAX_NIBBLES, TOOL, TextSpec

MARK = 2.0**-60  # U on a text token
OWN = 2.0**7  # -U on a chip token, cleared by the exit
CODE_GAIN = 2.0**52
OWN_GAIN = 2.0**84
MUTE = -1024.0
SINK_NORM = 16.0
HEAD = "lm_head.weight"

# text the anti-sink keys are fitted on (tests/check_merge.py measures on other text)
CALIBRATION = [
    "A hash function maps data of any size to a value of fixed size. A cryptographic hash function is also hard "
    "to invert and hard to find collisions for, which is what makes it useful for commitments and signatures.",
    "import hashlib\n\ndef double_sha256(data: bytes) -> bytes:\n    return hashlib.sha256(hashlib.sha256(data).digest()).digest()\n\n"
    "print(double_sha256(b'hello').hex())\n",
    "She had not expected the letter to arrive before the snow did. It lay on the mat, a little damp at one corner, "
    "addressed in her brother's careful capitals, and for a while she only looked at it.",
    "| block | bits | nonce |\n|---|---|---|\n| 0 | 0x2000ffff | 141 |\n| 1 | 0x2000ffff | 9 |\n| 2 | 0x2000ffff | 388 |\n",
    "Q: What is the capital of Australia?\nA: Canberra.\n\nQ: Who wrote Middlemarch?\nA: George Eliot.\n\nQ: What is 17 times 23?\nA: 391.\n",
    "Les nœuds se parlent par TCP. Chaque nœud vérifie les blocs qu'il reçoit et ne relaie que ceux qui sont valides ; "
    "la chaîne la plus lourde en travail l'emporte.",
]
CALIBRATION_CHATS = [
    ("Explain in two sentences what a Merkle tree is.", False),
    ("What is the SHA-256 of the bytes 00ff10?", True),
    ("Hash cafebabe for me and say what you get.", True),
    ("Write a haiku about a river in winter.", False),
]


def ram_available() -> float:
    """Bytes of memory this process can still have (Linux), or a generous guess."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return 1024.0 * int(line.split()[1])
    except OSError:
        pass
    return 2.0**36


# ------------------------------------------------------------------ safetensors, streamed
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Base:
    """The text model inside a Qwen3.5 checkpoint, read tensor by tensor (memory-mapped)."""

    def __init__(self, folder):
        self.dir = Path(folder)
        cfg = json.loads((self.dir / "config.json").read_text())
        self.cfg = cfg.get("text_config", cfg)
        self.where = {}
        self.bytes = 0  # the text model's size on disk
        for p in sorted(self.dir.glob("*.safetensors")):
            hdr, off = st_header(p)
            for k, v in hdr.items():
                if k.startswith("model.language_model."):
                    self.where["model." + k[len("model.language_model."):]] = (p, off, v)
                elif k == "lm_head.weight":  # the larger checkpoints (9B, 27B) have their own output matrix
                    self.where[HEAD] = (p, off, v)
                else:
                    continue
                self.bytes += v["data_offsets"][1] - v["data_offsets"][0]
        assert "model.embed_tokens.weight" in self.where, "no text model in " + str(folder)
        # tied: the output head is the embedding matrix (0.8B, 2B, 4B). The flag is in the text
        # config or, where that has none (9B, 27B), at the top.
        flags = [f for f in (self.cfg.get("tie_word_embeddings"), cfg.get("tie_word_embeddings")) if f is not None]
        self.tied = bool(flags[0]) if flags else HEAD not in self.where
        assert self.tied or HEAD in self.where, "the checkpoint unties its head but holds no lm_head.weight"

    def dtype(self, name) -> str:
        return self.where[name][2]["dtype"]

    def names(self):
        return list(self.where)

    def shape(self, name):
        return self.where[name][2]["shape"]

    def get(self, name, rows=None) -> torch.Tensor:
        """A tensor (or a slice of its rows) as float32."""
        p, off, v = self.where[name]
        dt = {"BF16": np.int16, "F32": np.float32, "F16": np.float16}[v["dtype"]]
        shape = tuple(v["shape"]) or (1,)
        m = np.memmap(p, dtype=dt, mode="r", offset=off + v["data_offsets"][0], shape=shape)
        a = np.array(m if rows is None else m[rows[0]:rows[1]])
        t = torch.from_numpy(a)
        return (t.view(torch.bfloat16) if v["dtype"] == "BF16" else t).float().reshape(a.shape if v["shape"] else ())


class Writer:
    """Writes a safetensors file from (name, shape, chunks) without holding it in memory."""

    def __init__(self):
        self.items = []

    def add(self, name, shape, chunks, dtype="BF16"):
        """chunks: a callable returning an iterator of float tensors that concatenate (dim 0)
        to the tensor. Stored as bf16: the chip's weights are bf16 values (checked where they
        are made), Qwen's matrices are bf16 already, and the rescaled norm gains and the
        anti-sink keys are rounded. The few tensors the base stores as float32 (a DeltaNet
        layer's A_log and its head norm) are copied as float32, bit for bit."""
        assert dtype in ("BF16", "F32"), dtype
        self.items.append((name, tuple(int(x) for x in shape), chunks, dtype))

    def write(self, folder, log=print, shard_bytes: int = 5 * 2**30):
        """Writes model.safetensors, or, past shard_bytes, model-0000i-of-0000N.safetensors with
        the index transformers reads (no file of a large checkpoint is bigger than a hub accepts).
        Returns (bytes of tensor data, number of parameters, the files)."""
        folder = Path(folder)
        sized = []
        for name, shape, chunks, dtype in self.items:
            count = int(np.prod(shape)) if shape else 1
            sized.append((name, shape, chunks, dtype, count, count * (2 if dtype == "BF16" else 4)))
        shards, acc = [[]], 0
        for it in sized:
            if shards[-1] and acc + it[5] > shard_bytes:
                shards.append([])
                acc = 0
            shards[-1].append(it)
            acc += it[5]
        files = (["model.safetensors"] if len(shards) == 1 else
                 [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))])
        for old in list(folder.glob("model*.safetensors")) + [folder / "model.safetensors.index.json"]:
            old.unlink(missing_ok=True)
        total = params = done_t = 0
        for fname, shard in zip(files, shards):
            hdr, off = {"__metadata__": {"format": "pt"}}, 0
            for name, shape, _, dtype, count, n in shard:
                hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
                off += n
                params += count
            blob = json.dumps(hdr, separators=(",", ":")).encode()
            blob += b" " * (-len(blob) % 8)
            with open(folder / fname, "wb") as f:
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)
                for name, shape, chunks, dtype, count, n in shard:
                    done = 0
                    for t in chunks():
                        if dtype == "BF16":
                            b = t.to(torch.bfloat16).contiguous()
                            f.write(b.view(torch.int16).numpy().tobytes())
                        else:
                            b = t.to(torch.float32).contiguous()
                            f.write(b.numpy().tobytes())
                        done += b.numel()
                    assert done == count, (name, done, shape)
                    if done_t % 40 == 0:
                        log(f"  {done_t + 1}/{len(sized)} tensors, {(total + f.tell()) / 2**30:.1f} GB")
                    done_t += 1
            total += off
        if len(shards) > 1:
            index = {"metadata": {"total_size": total}, "weight_map": {it[0]: fname for fname, shard in zip(files, shards) for it in shard}}
            (folder / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
        return total, params, files


# ------------------------------------------------------------------ the plan
def merged_layout(hq: int, n_dims: int, extra: int = 1) -> Layout:
    """Smallest H' = m * 4^k (a multiple of 64) with room for Qwen's dims, m ballast dims, the
    circuit's dims and the owner marker."""
    best = None
    for m in range(3, 65):
        k = 0
        while m * 4**k < hq + m + n_dims - 1 + extra or (m * 4**k) % 64:
            k += 1
        if best is None or m * 4**k < best.hidden:
            best = Layout(hidden=m * 4**k, m=m, off=hq)
    return best


def plan(base: Base):
    q = base.cfg
    hq, vq = q["hidden_size"], q["vocab_size"]
    text = TextSpec.from_tokenizer(base.dir, vq)
    nk = q["linear_num_key_heads"]
    # the circuit is laid out for two value heads per key head; emit splits a head where the
    # base has one (0.8B) and checks the layer's capacity
    c = build(Target(key_heads=nk, value_heads=2 * nk, intermediate=q["intermediate_size"]), text=text)
    lay = merged_layout(hq, len(c.dims))
    n_chip = c.n_layers()
    v_chip = -(-(S.VOCAB + 1) // 64) * 64
    col_u = lay.dim(len(c.dims))
    assert col_u < lay.hidden
    cfg = dict(q)
    cfg.update(hidden_size=lay.hidden, vocab_size=vq + v_chip, num_hidden_layers=n_chip + q["num_hidden_layers"],
               layer_types=["linear_attention"] * n_chip + list(q["layer_types"]), tie_word_embeddings=base.tied,
               architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")
    info = {"machine": "hashformer sha256 merged into Qwen3.5", "base_hidden": hq, "hidden": lay.hidden,
            "ballast_dims": lay.m, "chip_layers": n_chip, "chip_token_offset": vq, "chip_vocab": S.VOCAB,
            "pad": vq + S.PAD, "controls": {k: vq + v for k, v in S.CONTROLS.items()},
            "families": {k: [vq + v[0], v[1]] if isinstance(v, (list, tuple)) else v for k, v in S.FAMILIES.items()},
            "owner_col": col_u, "code_cols": [lay.dim(d) for d in c.keep],
            "bridge": {"tool": TOOL, "max_bytes": MAX_NIBBLES // 2, **text.summary()}}
    return c, lay, cfg, info, (col_u, n_chip, v_chip)


def calibration(tok) -> list:
    out = [tok(t)["input_ids"] for t in CALIBRATION]
    for user, tools in CALIBRATION_CHATS:
        x = tok.apply_chat_template([{"role": "user", "content": user}], tools=[TOOL] if tools else None,
                                    add_generation_prompt=True, tokenize=True)
        out.append(list(x["input_ids"]) if hasattr(x, "keys") else list(x))
    return out


@torch.no_grad()
def fit_anti_sink(base_dir, device: str = "cpu", dtype=torch.bfloat16, log=print) -> dict:
    """Per full-attention layer of the base, one key per key/value head for chip positions: it
    points against the mean query of the head's group on text, on the dims RoPE leaves alone
    (so at any distance). Returns {layer: (kv heads, head dim)}, before k_norm."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    if device == "auto":  # text statistics do not need exact kernels: the first backend that runs
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        tries = [("cuda", torch.bfloat16)] if torch.cuda.is_available() else []
        tries += [("mps", torch.bfloat16), ("mps", torch.float16)] if mps else []
        # fp32 first where it fits with room to spare (every CPU has fast kernels for it): it is twice the
        # checkpoint, and while it loads the mapped file sits beside it. A base that takes most of the memory
        # that is free is read as bf16 instead: on a 62 GB machine the 9B in fp32 (40 GB) left so little
        # that the whole machine, nodes included, stopped answering.
        need32, free = 2.2 * Base(base_dir).bytes, ram_available()
        tries += ([("cpu", torch.float32)] if need32 < 0.45 * free else []) + [("cpu", torch.bfloat16)]
        log(f"anti-sink: the base is read on {tries[0][0]} as {str(tries[0][1]).split('.')[-1]} "
            f"({need32 / 2**30:.0f} GB as float32, {free / 2**30:.0f} GB of memory free)")
    else:
        tries = [(device, dtype)]
    for device, dtype in tries:
        try:
            model = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=dtype).to(device).eval()
            model.model.language_model(input_ids=torch.tensor([tok("probe")["input_ids"]], device=device))
            break
        except (Exception, MemoryError) as e:  # noqa: BLE001
            log(f"anti-sink: the base does not run on {device} as {dtype} ({type(e).__name__}); trying the next backend")
            model = None
            gc.collect()
    assert model is not None, "no backend runs the base model"
    lm, tc = model.model.language_model, model.config.text_config
    layers = {i: l.self_attn for i, l in enumerate(lm.layers) if tc.layer_types[i] == "full_attention"}
    sums, count = {i: 0 for i in layers}, 0
    hooks = [a.q_norm.register_forward_hook(
        lambda m, inp, out, i=i: sums.__setitem__(i, sums[i] + out.float().flatten(0, -3).sum(0).cpu()))
        for i, a in layers.items()]
    for ids in calibration(tok):
        lm(input_ids=torch.tensor([ids], device=device))
        count += len(ids)
    for h in hooks:
        h.remove()
    hd, nkv = tc.head_dim, tc.num_key_value_heads
    grp = tc.num_attention_heads // nkv
    rot = int(hd * tc.rope_parameters.get("partial_rotary_factor", 1.0))
    assert rot < hd, "every dim of the heads is rotated: no place for a position-free key"
    out, strength = {}, []
    for i, a in layers.items():
        mean = sums[i] / count  # (heads, head dim): the mean query after q_norm
        gain = 1.0 + a.k_norm.weight.float().cpu()
        keys = torch.zeros(nkv, hd)
        for g in range(nkv):
            u = -mean[g * grp:(g + 1) * grp].mean(0)[rot:]
            strength.append(float(u.norm() / math.sqrt(hd)))  # the mean query's size, in units of a query's rms
            keys[g, rot:] = (u / gain[rot:]) / (u / gain[rot:]).norm() * SINK_NORM
        out[i] = keys
    log(f"anti-sink: {len(layers)} attention layers x {nkv} key heads fitted on {count} text tokens ({device}, {dtype}); "
        f"mean query / query rms {min(strength):.2f} .. {max(strength):.2f}")
    del model, lm, layers, hooks
    gc.collect()
    return out


def merge(base_dir, out_dir, log=print, device: str = "cpu", dtype=torch.bfloat16, anti_sink: bool = True,
          tools: bool = True):
    base = Base(base_dir)
    c, lay, cfg, info, (col_u, n_chip, v_chip) = plan(base)
    q = base.cfg
    hq, vq, h = q["hidden_size"], q["vocab_size"], lay.hidden
    s = math.sqrt(hq / h)
    one_normed = lay.ballast * lay.scale  # the ballast after any RMSNorm on a chip token
    assert q["linear_key_head_dim"] == 128 and q["linear_value_head_dim"] == 128
    st = c.stats()
    log(f"base: hidden {hq}, {q['num_hidden_layers']} layers, vocab {vq}, "
        f"{q['linear_num_key_heads']} x {q['linear_num_value_heads'] // q['linear_num_key_heads']} DeltaNet heads, "
        f"output head {'tied to the embedding' if base.tied else 'its own matrix'}")
    log(f"circuit: {st['dims']} dims, {st['key_heads']} key heads, {st['units']} units; reads "
        f"{info['bridge']['hex_tokens']} hex tokens of the base's vocabulary")
    log(f"merged: hidden {h} (ballast {lay.m}, norm scale 2^{int(math.log2(lay.scale))}), "
        f"{n_chip} + {q['num_hidden_layers']} layers, vocab {vq + v_chip}")
    sinks = fit_anti_sink(base_dir, device, dtype, log) if anti_sink else None

    tc = dict(cfg)
    chip = emit_layers(c, lay, tc, keep=c.keep, ballast_gate=c.isText, clear=((col_u, c.one - c.isText),))
    W = Writer()

    # --- the embedding. Where the base ties its output head to it, this matrix is the head too.
    # Where the base has a head of its own (9B, 27B), the merged head is written separately: the
    # base's output row in Qwen's dims, and of the chip's dims only what the final norm lets
    # through to the head: the code and the owner marker.
    head_cols = [lay.dim(d) for d in c.keep] + [col_u]

    def embed_rows(src="model.embed_tokens.weight", head=False):
        def f():
            tf = {lay.dim(d): torch.tensor(t, dtype=torch.float32) for d, t in c.text_features.items()}
            if head:
                tf = {col: t for col, t in tf.items() if col in head_cols}
            for a in range(0, vq, 16384):
                b = min(vq, a + 16384)
                e = torch.zeros(b - a, h)
                e[:, :hq] = base.get(src, (a, b))
                if not head:
                    e[:, lay.off:lay.off + lay.m] = lay.ballast
                for col, t in tf.items():
                    e[:, col] = t[a:b]
                e[:, col_u] = MARK
                check_bf16("text rows of " + src, e)
                yield e
            free = range(S.CTRL_BASE + len(S.CONTROL_NAMES), S.V_BASE)  # ids no token of the machine has
            for a in range(0, v_chip, 16384):
                e = embedding(c, lay, v_chip, S.VOCAB, (a, min(v_chip, a + 16384)))
                if a <= free.start < a + 16384:  # their codes are the printed text tokens' codes here: zero rows
                    e[free.start - a:free.stop - a] = 0
                e[:, col_u] = -OWN  # PAD and the unused rows too: no row of the chip block can follow text
                if head:
                    keep = torch.zeros(h, dtype=torch.bool)
                    keep[head_cols] = True
                    e[:, ~keep] = 0
                yield e
        return f

    W.add("model.embed_tokens.weight", (vq + v_chip, h), embed_rows())
    if not base.tied:
        assert tuple(base.shape(HEAD)) == (vq, hq), base.shape(HEAD)
        W.add(HEAD, (vq + v_chip, h), embed_rows(HEAD, head=True))

    # --- the chip layers
    for name, t in chip.items():
        W.add(name, t.shape, (lambda t=t: iter([t])))

    # --- Qwen's layers, widened and muted on chip tokens
    def norm_gain(name):
        def f():
            w = base.get(name).double()
            out = torch.zeros(h, dtype=torch.float64)
            out[:hq] = (1 + w) * s - 1
            yield out.float()
        return f

    def cols(name, mute_rows=None, ballast_col=None):
        def f():
            w = base.get(name)
            out = torch.zeros(w.shape[0], h)
            out[:, :hq] = w
            if mute_rows is not None:
                out[mute_rows, lay.off] = MUTE / one_normed
            if ballast_col is not None:
                out[:, lay.off] = ballast_col / one_normed
            yield out
        return f

    def rows(name):
        def f():
            w = base.get(name)
            out = torch.zeros(h, w.shape[1])
            out[:hq] = w
            yield out
        return f

    def same(name):
        return lambda: iter([base.get(name)])

    hd2 = 2 * q["head_dim"]
    gate_rows = torch.arange(q["num_attention_heads"] * hd2).view(-1, hd2)[:, q["head_dim"]:].reshape(-1)
    for name in sorted(base.names()):
        if not name.startswith("model.layers."):
            continue  # the embedding, the head and the final norm are written on their own
        _, _, idx, rest = name.split(".", 3)
        new = f"model.layers.{n_chip + int(idx)}.{rest}"
        shape = base.shape(name)
        if rest.endswith("layernorm.weight"):
            W.add(new, (h,), norm_gain(name))
        elif rest in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
            W.add(new, (shape[0], h), cols(name, mute_rows=slice(None)))
        elif rest == "self_attn.q_proj.weight":
            W.add(new, (shape[0], h), cols(name, mute_rows=gate_rows))
        elif rest == "self_attn.k_proj.weight" and sinks is not None:
            W.add(new, (shape[0], h), cols(name, ballast_col=sinks[int(idx)].flatten()))
        elif rest.split(".")[-2] in ("in_proj_qkv", "in_proj_z", "k_proj", "v_proj", "gate_proj", "up_proj"):
            W.add(new, (shape[0], h), cols(name))
        elif rest.split(".")[-2] in ("out_proj", "o_proj", "down_proj"):
            W.add(new, (h, shape[1]), rows(name))
        else:  # conv, dt_bias, A_log, the head-level norms: as the base stores them
            W.add(new, shape, same(name), "F32" if base.dtype(name) == "F32" else "BF16")

    # --- final norm: Qwen's dims, the code, the owner marker; nothing else reaches the head
    def final_norm():
        w = base.get("model.norm.weight").double()
        out = torch.full((h,), -1.0, dtype=torch.float64)
        out[:hq] = (1 + w) * s - 1
        for d in c.keep:
            out[lay.dim(d)] = CODE_GAIN
        out[col_u] = OWN_GAIN
        yield out.float()

    W.add("model.norm.weight", (h,), final_norm)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, n_params, files = W.write(out, log, shard_bytes=int(float(os.environ.get("QWEN_SHARD_GB", "5")) * 2**30))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        if (base.dir / f).exists():
            shutil.copy(base.dir / f, out / f)
    if tools and (out / "chat_template.jinja").exists():
        # the node's functions, declared by the checkpoint itself when the caller declares none
        from rails.template import bake_dir
        from rails.tools import NAMES
        bake_dir(out)
        info["functions"] = NAMES
    stops = sorted({i for i in (q.get("eos_token_id"), info["bridge"].get("im_end")) if i is not None}) + [vq + S.HALT]
    gen = {"do_sample": False, "eos_token_id": stops, "pad_token_id": vq + S.PAD}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=1))
    info.update(norm_rescale=s, anti_sink=sinks is not None, parameters=n_params, tied_head=base.tied, weight_files=files,
                base=str(json.loads((base.dir / "config.json").read_text()).get("_name_or_path") or base.dir.name))
    (out / "hashformer.json").write_text(json.dumps(info, indent=1))
    log(f"{out}: {n / 2**30:.2f} GB in {len(files)} file{'s' if len(files) > 1 else ''}, {n_params / 1e9:.2f}B parameters")
    return out


class _BaseText:
    """The text model of a stock Qwen3.5 checkpoint behind the two names the checks use."""

    def __init__(self, full):
        self.full, self.model, self.lm_head = full, full.model.language_model, full.lm_head

    def generate(self, ids, **kw):
        return self.full.generate(input_ids=ids, **kw)


def load_base(base_dir, dtype=torch.bfloat16, device="cpu"):
    """The base checkpoint as transformers loads it (the stock image-text class; only its text
    model is used): what a merged model is compared to."""
    from transformers import AutoModelForImageTextToText

    return _BaseText(AutoModelForImageTextToText.from_pretrained(base_dir, dtype=dtype).to(device).eval())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge the SHA-256 chip into a Qwen3.5 checkpoint")
    ap.add_argument("base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="where the base runs once, to fit the anti-sink keys (auto: try the GPU)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--no-anti-sink", action="store_true")
    ap.add_argument("--no-tools", action="store_true", help="keep the base's chat template as it is")
    a = ap.parse_args()
    merge(a.base, a.out, device=a.device, dtype=getattr(torch, a.dtype), anti_sink=not a.no_anti_sink, tools=not a.no_tools)
