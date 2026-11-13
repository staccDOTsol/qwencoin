"""The untied-head path of the merge (what the 9B and 27B bases need), exercised at 0.8B scale.

    python3 tests/check_untied.py [build|weights|logits|all]        -> logs/untied.log

A copy of Qwen3.5-0.8B is given an output matrix of its own, W_out = E + 1 v^T rounded to bf16
(every row shifted by one vector v: the softmax, and so the text, is the stock model's, while as
matrices the two differ in every row), and its config unties the head the way the 9B's does (no
flag in the text config, false at the top). That checkpoint is merged like any other. Then:

  weights   the merged lm_head holds the base's lm_head in Qwen's dims and the merged embedding
            the base's embedding, bit for bit; a head row holds nothing outside Qwen's dims, the
            code columns and the owner column; the chip rows of the two matrices agree there
  loading   transformers loads the merged checkpoint as two separate matrices
  logits    on text, the merged model's raw logits against the untied base's. The shift is
            included: a head built from the embedding would be off by h.v at every position,
            and the size of h.v is printed next to the difference found

tests/check_merge.py then runs on the pair like on any merge (the pipeline does that for the
real 9B and 27B).
"""
import json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hashformer import merge as M  # noqa: E402

SRC, BASE, OUT = ROOT / "base" / "Qwen3.5-0.8B", ROOT / "work" / "base-untied", ROOT / "work" / "merged-untied"
LOG = open(ROOT / "logs" / "untied.log", "a")
TEXTS = ["The quick brown fox jumps over the lazy dog, and then it rests for a while under the old oak tree.",
         "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\nprint(gcd(1071, 462))\n"]


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def build():
    from safetensors.torch import load_file, save_file

    shutil.rmtree(BASE, ignore_errors=True)
    BASE.mkdir(parents=True)
    for f in SRC.iterdir():
        if f.suffix != ".safetensors" and f.name != "config.json":
            shutil.copy(f, BASE / f.name)
    cfg = json.loads((SRC / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    cfg["text_config"].pop("tie_word_embeddings", None)
    (BASE / "config.json").write_text(json.dumps(cfg, indent=1))
    (src,) = SRC.glob("*.safetensors")
    d = load_file(src)
    e = d["model.language_model.embed_tokens.weight"]
    g = torch.Generator().manual_seed(7)
    v = (torch.randint(0, 2, (e.shape[1],), generator=g).float() * 2 - 1) * 0.0625
    d["lm_head.weight"] = (e.float() + v).to(torch.bfloat16)
    save_file(d, BASE / src.name, metadata={"format": "pt"})
    diff = int((d["lm_head.weight"] != e).any(1).sum())
    say(f"untied base in {BASE}: lm_head = embedding + v (|v_i| = 1/16), rounded to bf16; {diff} of {e.shape[0]} rows differ from the embedding")
    del d
    shutil.rmtree(OUT, ignore_errors=True)
    M.merge(BASE, OUT, log=say)


def weights():
    base = M.Base(BASE)
    assert not base.tied
    info = json.loads((OUT / "hashformer.json").read_text())
    cfg = json.loads((OUT / "config.json").read_text())
    assert cfg["tie_word_embeddings"] is False and info["tied_head"] is False
    hq, vq, h = info["base_hidden"], info["chip_token_offset"], info["hidden"]
    hdr, where = {}, {}  # the checkpoint is past 5 GB: shards
    for p in sorted(OUT.glob("*.safetensors")):
        hd, off = M.st_header(p)
        for k, v in hd.items():
            if k != "__metadata__":
                hdr[k], where[k] = v, (p, off + v["data_offsets"][0])
    assert M.HEAD in hdr and hdr[M.HEAD]["shape"] == hdr["model.embed_tokens.weight"]["shape"] == [cfg["vocab_size"], h]

    def rows(name, a, b):
        p, off = where[name]
        m = np.memmap(p, dtype=np.int16, mode="r", offset=off, shape=tuple(hdr[name]["shape"]))
        return torch.from_numpy(np.array(m[a:b])).view(torch.bfloat16).float()

    allowed = torch.zeros(h, dtype=torch.bool)
    allowed[:hq] = True
    allowed[info["code_cols"] + [info["owner_col"]]] = True
    n = 0
    for a in range(0, vq, 32768):
        b = min(vq, a + 32768)
        hd, em = rows(M.HEAD, a, b), rows("model.embed_tokens.weight", a, b)
        assert torch.equal(hd[:, :hq], base.get(M.HEAD, (a, b))), "head rows are not the base's lm_head"
        assert torch.equal(em[:, :hq], base.get("model.embed_tokens.weight", (a, b))), "embedding rows are not the base's embedding"
        assert not hd[:, ~allowed].any(), "a head row holds something the head does not read"
        assert torch.equal(hd[:, info["code_cols"]], em[:, info["code_cols"]]) and torch.equal(hd[:, info["owner_col"]], em[:, info["owner_col"]])
        n += b - a
    total = cfg["vocab_size"]
    for a in range(vq, total, 32768):
        b = min(total, a + 32768)
        hd, em = rows(M.HEAD, a, b), rows("model.embed_tokens.weight", a, b)
        assert not hd[:, :hq].any() and not hd[:, ~allowed].any()
        assert torch.equal(hd[:, allowed], em[:, allowed]), "chip rows of the head and of the embedding disagree where the head reads"
    say(f"weights: lm_head[:, :{hq}] == the base's lm_head and embed[:, :{hq}] == the base's embedding on all {n} text rows, bit for bit; "
        f"head rows are zero outside Qwen's dims, the {len(info['code_cols'])} code columns and the owner column; "
        f"{total - vq} chip rows agree between head and embedding there")
    f32 = [k for k, v in hdr.items() if v["dtype"] == "F32"]
    say(f"float32 tensors kept as the base stores them: {len(f32)} (e.g. {f32[0] if f32 else '-'})")


@torch.no_grad()
def logits():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    ids = [tok(t)["input_ids"] for t in TEXTS]
    t0 = time.time()
    b = M.load_base(BASE)
    assert b.lm_head.weight is not b.model.embed_tokens.weight
    want, shift = [], []
    for x in ids:
        hs = b.model(input_ids=torch.tensor([x])).last_hidden_state[0].float()
        lg = (hs @ b.lm_head.weight.float().T)
        want.append(lg)
        shift.append((lg - hs @ b.model.embed_tokens.weight.float().T).mean(1))  # h.v, the same for every token
    del b
    m = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16).eval()
    assert m.lm_head.weight is not m.model.embed_tokens.weight, "the merged checkpoint loaded with a tied head"
    vq = want[0].shape[1]
    right, wrong, agree, npos, hv = [], [], 0, 0, 0.0
    for x, w, s in zip(ids, want, shift):
        full = m(input_ids=torch.tensor([x])).logits[0].float()
        got = full[:, :vq]
        right.append((got - w).abs().mean(1))  # against the base's own head
        wrong.append((got - (w - s[:, None])).abs().mean(1))  # against a head built from the base's embedding
        agree += int((got.argmax(1) == w.argmax(1)).sum())
        npos += len(x)
        hv = max(hv, float(s.abs().max()))
        assert float(full[:, vq:].max()) < -1e6, "a chip token can follow text"
    right, wrong = torch.cat(right), torch.cat(wrong)
    say(f"loading: two matrices in the merged model. logits on {npos} text positions: mean |merged - untied base| per position "
        f"{float(right.mean()):.3f} (largest {float(right.max()):.3f}; bf16 arithmetic and the wider norm), top-1 equal at {agree}/{npos}. "
        f"Against a head built from the embedding the same figure is {float(wrong.mean()):.3f}: h.v, up to {hv:.1f} ({time.time() - t0:.0f}s)")
    assert float(right.mean()) < 0.2 * float(wrong.mean()) and agree >= 0.9 * npos, (float(right.mean()), float(wrong.mean()), agree, npos)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("build", "all"):
        build()
    if what in ("weights", "all"):
        weights()
    if what in ("logits", "all"):
        logits()
    say("OK" if what == "all" else f"{what} done")
