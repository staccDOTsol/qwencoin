"""The circuit as a stock ``Qwen3_5ForCausalLM`` (no custom modelling code, no trust_remote_code).

    python -m hashformer.hf --out dist/qwensha

Layer shapes are Qwen3.5-4B's (16 key heads, 32 value heads of 128 dims, MLP width 9216, conv
kernel 4), so the same layers graft in front of the 4B checkpoint. The vocabulary is the chip
tokens plus PAD. After the chip layers comes one all-zero full-attention layer, which HF's cache
needs to count positions. The output head is tied to the embedding: a token's row holds its
output code as +-CODE_AMP, the exit leaves the predicted code in the same dims, and the argmax
is the token whose code agrees on every bit.

The model starts from an empty cache. Nothing is injected: the IV, the round constants and the
whole control live in the weights, and all state is built from tokens.
"""

import json
from pathlib import Path

import numpy as np
import torch

from . import ir
from . import spec as S
from .circuit import NCODE, build, predictions
from .emit import check_bf16, embedding, emit_layers, nonzeros, pick_layout

CODE_GAIN = 2.0**20  # final-norm gain on the code dims

BASE = {  # Qwen3.5-4B's text config, minus the sizes this model picks
    "hidden_act": "silu",
    "intermediate_size": 9216,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True},
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
}


def build_model(c=None, dtype=torch.bfloat16, exit: bool = True):
    """(model, circuit, layout). ``exit=False`` leaves every circuit dim in the residual after
    the last chip layer, for the dim-by-dim comparison against the IR."""
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    c = c or build()
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(S.VOCAB + 1) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              eos_token_id=S.HALT, pad_token_id=S.PAD, bos_token_id=S.SHA)
    sd = emit_layers(c, lay, tc, keep=c.keep if exit else None)
    nnz = nonzeros(sd)
    sd["model.embed_tokens.weight"] = embedding(c, lay, n_vocab, S.VOCAB)
    norm = torch.full((lay.hidden,), -1.0)  # gain 1 + w = 0: the head sees only the code
    for d in c.keep:
        norm[lay.dim(d)] = CODE_GAIN
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    for p in model.parameters():
        p.data.zero_()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(k.startswith(f"model.layers.{n_chip}.") or k == "lm_head.weight" for k in missing), missing[:5]
    model.tie_weights()
    model.generation_config = GenerationConfig(eos_token_id=S.HALT, pad_token_id=S.PAD,
                                               do_sample=False)
    model.nonzero_weights = nnz
    return model.to(dtype).eval(), c, lay


def check_residual(model, c, lay, tokens, mode: str = "prefill") -> int:
    """Run tokens through the HF modules (a no-exit model) and assert that the residual after
    the last chip layer equals the IR evaluator's on every circuit dim at every position.
    "prefill": one forward pass (the chunked delta-rule kernel); "decode": token by token with
    the cache (the recurrent kernel); "mixed": a chunked prefix, then token by token."""
    want = ir.evaluate(c, tokens, check=False)
    dims = list(range(1, len(c.dims)))
    cols = [lay.dim(d) for d in dims]
    x = torch.tensor([list(tokens)])
    n_chip = c.n_layers()
    hs = []
    with torch.no_grad():
        if mode == "prefill":
            o = model(input_ids=x, use_cache=True, output_hidden_states=True)
            hs.append(o.hidden_states[n_chip][0])
        else:
            cache, start = None, 0
            if mode == "mixed":
                start = len(tokens) // 2
                o = model(input_ids=x[:, :start], use_cache=True, output_hidden_states=True)
                cache = o.past_key_values
                hs.append(o.hidden_states[n_chip][0])
            for t in range(start, len(tokens)):
                o = model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                cache = o.past_key_values
                hs.append(o.hidden_states[n_chip][0])
    h = torch.cat(hs).double().numpy()[:, cols]
    w = want[:, dims]
    bad = np.nonzero((h != w).any(0))[0]
    if len(bad):
        rows = []
        for i in bad[:6]:
            t = int(np.argmax(h[:, i] != w[:, i]))
            rows.append((c.dims[dims[i]].name, t, float(h[t, i]), float(w[t, i])))
        raise AssertionError(f"{len(bad)} dims differ from the IR: {rows}")
    return h.shape[0]


class HfSession:
    """Drives the checkpoint as the node does: the host answers IN, the model does the rest.
    Every token the model owns is its own greedy argmax through the stock lm_head."""

    def __init__(self, model) -> None:
        self.model = model
        self.cache = None
        self.tokens: list[int] = []
        self.owned: list[bool] = []

    @torch.no_grad()
    def _step(self, toks: list[int]) -> int:
        o = self.model(input_ids=torch.tensor([toks]), past_key_values=self.cache, use_cache=True)
        self.cache = o.past_key_values
        return int(o.logits[0, -1].float().argmax())

    def run(self, host: list[int], until: int) -> list[int]:
        """Feed the host tokens whenever the model asks for input; stop when it emits ``until``
        with the host's tokens used up."""
        i = 1
        self.tokens.append(host[0])
        self.owned.append(False)
        pred = self._step([host[0]])
        while True:
            if pred == S.IN:
                tok, mine = host[i], False
                i += 1
            else:
                tok, mine = pred, True
            self.tokens.append(tok)
            self.owned.append(mine)
            if mine and tok == until and i >= len(host):
                return self.tokens
            pred = self._step([tok])


def hf_sha256(model, msg: bytes) -> tuple:
    s = HfSession(model)
    toks = s.run([S.SHA] + [S.B(x) for x in msg] + [S.END], S.HALT)
    pay = [S.family(t)[1] for t in toks[-17:-1]]
    return b"".join((pay[2 * i] << 16 | pay[2 * i + 1]).to_bytes(4, "big") for i in range(8)), s


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Save the SHA-256 circuit as a stock Qwen3.5 checkpoint")
    ap.add_argument("--out", default="dist/qwensha")
    args = ap.parse_args()
    model, c, lay = build_model()
    out = Path(args.out)
    model.save_pretrained(out)
    meta = {"machine": "hashformer sha256", "vocab": S.VOCAB, "pad": S.PAD,
            "controls": S.CONTROLS, "families": {k: list(v) for k, v in S.FAMILIES.items()},
            "hidden": lay.hidden, "ballast_dims": lay.m, "chip_layers": c.n_layers(),
            "circuit": c.stats(), "nonzero_layer_weights": model.nonzero_weights,
            "code_dims": [lay.dim(d) for d in c.keep], "code_bits": NCODE}
    (out / "hashformer.json").write_text(json.dumps(meta, indent=1))
    n = sum(p.numel() for p in model.parameters())
    print(f"{out}: {c.stats()}, hidden {lay.hidden}, {n / 1e6:.0f}M parameters, "
          f"{model.nonzero_weights} nonzero layer weights")


if __name__ == "__main__":
    main()
