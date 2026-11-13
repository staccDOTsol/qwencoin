"""A merged checkpoint (hashformer.merge) against the reference machine and against its base.

    python3 tests/check_merge.py base   BASE_DIR    WORK      base logits on the text set (and the bf16 / fp32 floor)
    python3 tests/check_merge.py merged MERGED_DIR  WORK      everything else
    python3 tests/check_merge.py plain  MERGED_DIR  WORK      only the plain-chat probe of `merged` (after `base`)

    DEVICE=cpu|mps  DTYPE=bfloat16|float32  QUICK=1 (the short list)  THREADS=n

Chip    every token the model owns equals the reference trace, teacher-forced over the whole
        vocabulary (text tokens included) and free-running, alone and after a chat prefix.
Bridge  one stream from the chat prompt to the answer: the call as Qwen writes it, SHA at
        </tool_call>, the payload replayed from the circuit's memory, the digest printed as
        text. Teacher-forced on the reference stream, then a plain greedy generate() from the
        user's question, in which the language model writes the call itself.
Owner   after a text token the circuit does not own, no chip token can win.
Text    the merged model's next-token distribution against the base's on the same text, and
        the same text continued after a trace against the same text without one.
"""
import hashlib, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hashformer import ref, spec as S  # noqa: E402
from hashformer.bridge import Bridge, from_merged, to_merged  # noqa: E402
from hashformer.textspec import PRINT_HEAD, PRINT_TAIL, TOOL, TextSpec  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", str(torch.get_num_threads()))))
DEV = os.environ.get("DEVICE", "cpu")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
QUICK = bool(os.environ.get("QUICK"))
mode, model_dir, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
work.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "The proof of work in Bitcoin is a partial hash inversion: a miner varies a nonce until the double SHA-256 "
    "of the block header falls below a target. Nobody can find such a nonce faster than by trying, and anybody "
    "can check one with two hash evaluations. Every 2016 blocks the target is adjusted so that blocks keep "
    "arriving about ten minutes apart, whatever the total hash power of the network happens to be.",
    "def merkle_root(txids):\n    level = list(txids)\n    while len(level) > 1:\n        if len(level) % 2:\n"
    "            level.append(level[-1])\n        level = [sha256d(a + b) for a, b in zip(level[::2], level[1::2])]\n"
    "    return level[0]\n\n# The root commits to every transaction in the block.\n",
    "In the spring the river rose and took the lower field. My grandfather walked the fence line every morning "
    "anyway, as if the posts might have moved in the night, and came back with his boots wet and nothing to report.",
]
CHAT = [{"role": "user", "content": "What is the SHA-256 hash of the three letters abc? Answer briefly."}]
ASK = "What is the SHA-256 of the bytes 616263?"
TAIL = (" The digest above is what every node computes for those three bytes, and it is the same on every machine; "
        "a single changed bit in the input would give an unrelated value.")


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def chat_ids(tok, msgs, tools=None):
    x = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


def text_ids(tok):
    return [tok(t)["input_ids"] for t in TEXTS] + [chat_ids(tok, CHAT)]


def call_ids(tok, payload):
    return tok(f"<tool_call>\n<function=sha256>\n<parameter=hex>\n{payload}\n</parameter>\n</function>\n</tool_call>",
               add_special_tokens=False)["input_ids"]


def until_stop(ids, stops):
    """A generation up to and including its first stop token."""
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i + 1]
    return ids


@torch.no_grad()
def logits_of(model, ids, start=0):
    """Logits of positions start.. (the head is applied in slices: a few hundred positions times
    380k tokens is gigabytes)."""
    h = model.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0, start:]
    for a in range(0, h.shape[0], 96):
        yield model.lm_head(h[a:a + 96]).float().cpu()


def all_logits(model, ids):
    return torch.cat(list(logits_of(model, ids)))


def kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


if mode == "base":
    from hashformer.merge import load_base
    tok = tokenizer(model_dir)
    sets = text_ids(tok)
    model = load_base(model_dir, DTYPE, DEV)
    lp = [torch.log_softmax(all_logits(model, ids), -1) for ids in sets]
    torch.save({"ids": sets, "logp": [x.half() for x in lp], "top": [x.argmax(-1) for x in lp]}, work / "base.pt")
    with torch.no_grad():
        greedy = model.generate(torch.tensor([sets[-1]], device=DEV), max_new_tokens=24, do_sample=False)[0].tolist()
    torch.save(greedy, work / "base_greedy.pt")
    ts = TextSpec.from_tokenizer(model_dir, model.model.embed_tokens.weight.shape[0])
    digest = hashlib.sha256(bytes.fromhex("616263")).hexdigest()
    answer_to = (chat_ids(tok, [{"role": "user", "content": ASK}], [TOOL]) + call_ids(tok, "616263")
                 + [ts.printed[n] for n in PRINT_HEAD] + [ts.digit[int(ch, 16)] for ch in digest] + [ts.printed[n] for n in PRINT_TAIL])
    with torch.no_grad():
        ans = model.generate(torch.tensor([answer_to], device=DEV), max_new_tokens=120, do_sample=False)[0].tolist()
    torch.save({"context": answer_to, "answer": ans[len(answer_to):]}, work / "base_answer.pt")
    print(f"base, after the same call and response as text: {tok.decode(ans[len(answer_to):])!r}")
    print(f"base ({DEV}, {DTYPE}): {sum(len(s) for s in sets)} text positions; greedy answer: {tok.decode(greedy[len(sets[-1]):])!r}")
    if os.environ.get("FLOOR", "1") != "0":
        del model
        __import__("gc").collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        other = torch.bfloat16 if DEV == "cpu" and DTYPE == torch.float32 else torch.float32
        name = str(other).split(".")[-1]
        from hashformer.merge import Base, ram_available
        need, free = (2.2 if other == torch.float32 else 1.1) * Base(model_dir).bytes, ram_available()
        if need > 0.45 * free:  # a second copy that takes most of the free memory stops the machine, not just this check
            (work / "floor.json").unlink(missing_ok=True)
            print(f"floor: not measured on this machine (the base in {name} takes {need / 2**30:.0f} GB, {free / 2**30:.0f} GB of memory are free)")
            sys.exit(0)
        m32 = load_base(model_dir, other, "cpu")
        l32 = []
        with torch.no_grad():
            for ids in sets:
                h = m32.model(input_ids=torch.tensor([ids])).last_hidden_state[0]
                l32.append(torch.cat([m32.lm_head(h[a:a + 96]).float() for a in range(0, h.shape[0], 96)]))
        ks = torch.cat([kl(a, torch.log_softmax(b, -1)) for a, b in zip(lp, l32)])
        agree = torch.cat([(a.argmax(-1) == b.argmax(-1)) for a, b in zip(lp, l32)]).float().mean()
        json.dump({"kl_mean": ks.mean().item(), "kl_max": ks.max().item(), "top1": agree.item(), "other": name},
                  open(work / "floor.json", "w"))
        print(f"floor (the base as above against itself in {name} on the cpu): KL mean {ks.mean():.2e}, max {ks.max():.2e}, "
              f"top-1 agreement {agree:.4f}")
    sys.exit(0)

# ------------------------------------------------------------------ merged
from transformers import AutoModelForCausalLM  # noqa: E402

info = json.loads((Path(model_dir) / "hashformer.json").read_text())
VQ = info["chip_token_offset"]
IN, HALT, SHA, END = (VQ + x for x in (S.IN, S.HALT, S.SHA, S.END))
tok = tokenizer(model_dir)
ts = TextSpec.from_tokenizer(model_dir, VQ)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=DTYPE).to(DEV).eval()
print(f"merged ({DEV}, {DTYPE}): {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters, hidden {info['hidden']}, "
      f"{info['chip_layers']} chip layers + Qwen's, chip tokens from id {VQ}", flush=True)


def chip(tokens):
    return [VQ + t for t in tokens]


def plain_probe():
    """The chat prompt with no functions declared. A merged checkpoint's own template declares the
    node's functions when the caller passes none, so the caller passes an empty list: that has to
    render what the base's stock template renders, token for token, and from there the merged
    model's greedy answer is compared with the base's. What the model does with the same
    question when its own template speaks is printed for the record."""
    plain = list(torch.load(work / "base.pt")["ids"][-1])  # as the base's tokenizer rendered it
    assert chat_ids(tok, CHAT, tools=[]) == plain, "an empty tool list does not render the stock prompt"
    stops = set(model.generation_config.eos_token_id)
    with torch.no_grad():
        greedy = model.generate(torch.tensor([plain], device=DEV), max_new_tokens=24, do_sample=False)[0].tolist()
    mine, theirs = until_stop(greedy[len(plain):], stops), until_stop(torch.load(work / "base_greedy.pt")[len(plain):], stops)
    same = sum(1 for a, b_ in zip(mine, theirs) if a == b_)
    print(f"  greedy answer to a chat prompt with no functions declared (tools=[]: the base's own {len(plain)} prompt tokens): "
          f"{tok.decode(mine)!r}; {same} of {max(len(mine), len(theirs))} tokens equal the base's", flush=True)
    own = chat_ids(tok, CHAT)
    if own != plain:
        with torch.no_grad():
            g = model.generate(torch.tensor([own], device=DEV), max_new_tokens=24, do_sample=False)[0].tolist()
        print(f"  the same question through this checkpoint's own template ({len(own)} prompt tokens: the node's functions "
              f"declared), first 24 tokens: {show(until_stop(g[len(own):], stops))!r}", flush=True)


def show(ids):
    """Text as text, runs of chip tokens as a count."""
    out, n = [], 0
    for i in list(ids) + [None]:
        if i is not None and i >= VQ:
            n += 1
            continue
        if n:
            out.append(f"[{n} chip tokens]")
            n = 0
        if i is not None:
            out.append(tok.decode([i]))
    return "".join(out)


def check_stream(ids, want, start, what):
    """Teacher-forced: want[i] is the merged id the circuit must emit after position i (None:
    the language model speaks there, and then no chip token may win)."""
    bad, at, win, other, chip_free, n_own = 0, start, [], [], [], 0
    for lg in logits_of(model, ids, start):
        for r in range(lg.shape[0]):
            w, row = want[at], lg[r]
            if w is None:
                chip_free.append(row[VQ:].max().item())
                bad += int(row.argmax().item() >= VQ)
            else:
                n_own += 1
                bad += int(row.argmax().item() != w)
                win.append(row[w].item())
                row[w] = float("-inf")
                other.append(row.max().item())
            at += 1
    msg = f"  {what}: {at - start} positions, the circuit owns {n_own}, {bad} wrong"
    if win:
        msg += f"; winner {min(win):.3g}..{max(win):.3g}, runner-up at most {max(b / a for a, b in zip(win, other)):.3f} of it"
    if chip_free:
        msg += f"; where Qwen speaks the best chip logit is {max(chip_free):.3g}"
    print(msg, flush=True)
    assert bad == 0
    if chip_free:
        assert max(chip_free) < -1000


def check_trace(prefix, msg, what):
    tr = ref.trace(msg)
    ids = prefix + chip(tr.tokens[:-1])
    want = [None] * len(prefix) + [VQ + (tr.tokens[i + 1] if tr.owned[i + 1] else S.IN) for i in range(len(tr.tokens) - 1)]
    check_stream(ids, want, len(prefix), what)
    return tr


@torch.no_grad()
def step_all(ids):
    """Greedy token by token through the cache: (cache, argmax after the last token)."""
    o = model(input_ids=torch.tensor([ids], device=DEV), use_cache=True)
    return o.past_key_values, int(o.logits[0, -1].float().argmax())


@torch.no_grad()
def run(prefix, msg):
    """Free-running, host-driven: the host feeds the prefix and SHA, and answers IN with the bytes."""
    host = [SHA] + [VQ + S.B(x) for x in msg] + [END]
    cache, pred = step_all(prefix + host[:1])
    out, i = [host[0]], 1
    while True:
        nxt = pred
        if pred == IN:
            nxt, i = host[i], i + 1
        out.append(nxt)
        if nxt == HALT and i >= len(host):
            break
        o = model(input_ids=torch.tensor([[nxt]], device=DEV), past_key_values=cache, use_cache=True)
        cache, pred = o.past_key_values, int(o.logits[0, -1].float().argmax())
    pay = [(t - VQ - S.V_BASE) & 0xFFFF for t in out[-17:-1]]
    return b"".join(p.to_bytes(2, "big") for p in pay), out


if mode == "plain":
    print("the plain-chat probe alone:")
    plain_probe()
    sys.exit(0)

sets = text_ids(tok)
chat = sets[-1]
print("chip, teacher-forced over the whole vocabulary:")
check_trace([], b"abc", "pure trace, 'abc'")
if not QUICK:
    check_trace([], bytes(range(70)), "pure trace, 70 bytes (2 blocks)")
check_trace(chat, b"abc", f"after a chat prefix of {len(chat)} text tokens")
if not QUICK:
    check_trace(sets[0], bytes(range(121)), f"after {len(sets[0])} tokens of prose, 121 bytes (3 blocks)")

print("chip, free-running, the host answering IN (token by token with the cache):")
t0 = time.time()
d, out = run(chat, b"abc")
ok = d == hashlib.sha256(b"abc").digest() and out == chip(ref.trace(b"abc").tokens)
print(f"  inside the chat: {d.hex()} {'== hashlib, every token == reference' if ok else 'MISMATCH'} "
      f"({len(out)} tokens, {(time.time() - t0) / len(out) * 1e3:.0f} ms per token)", flush=True)
assert ok

# ------------------------------------------------------------------ the bridge
ask = chat_ids(tok, [{"role": "user", "content": ASK}], [TOOL])


def call(payload):
    return call_ids(tok, payload)


def reference(ids):
    """What the circuit owes after each position of a merged-id stream, and the bridge at its end."""
    b, want = Bridge(ts), []
    for i in ids:
        b.feed(from_merged(i, VQ))
        p = b.predict()
        want.append(None if p is None else to_merged(p, VQ))
    return want, b


def complete(ids):
    """Extend a stream with the circuit's own tokens for as long as it owns the next one."""
    ids = list(ids)
    want, b = reference(ids)
    while want[-1] is not None:
        ids.append(want[-1])
        b.feed(from_merged(ids[-1], VQ))
        p = b.predict()
        want.append(None if p is None else to_merged(p, VQ))
    return ids, want


print("bridge, teacher-forced over the whole vocabulary (prompt, call, hash, answer, text after it):")
cases = [("616263", ask), ("6a09e667f3bcc908b2fb1366ea957d3e3adec17512775099da2f590b0667322a", [])]
if QUICK:
    cases = cases[:1]
for payload, pre in cases:
    stream, _ = complete(pre + call(payload))
    stream += tok(TAIL)["input_ids"]
    want, _ = reference(stream)
    digest = hashlib.sha256(bytes.fromhex(payload)).hexdigest()
    assert f"<tool_response>\n{digest}\n</tool_response>" in show(stream)
    check_stream(stream, want, 0, f"sha256({payload[:16]}{'..' if len(payload) > 16 else ''})")
for payload, why in [("61626", "an odd number of digits"), ("xyz", "not hex")]:
    stream = call(payload) + tok(TAIL)["input_ids"]
    want, _ = reference(stream)
    assert all(w is None for w in want)
    check_stream(stream, want, 0, f"no call for {why}")

print("bridge, free-running: generate() from the user's question (greedy, stock stopping rules):")
t0 = time.time()
with torch.no_grad():
    gen = model.generate(torch.tensor([ask], device=DEV), max_new_tokens=800, do_sample=False)[0].tolist()
dt = time.time() - t0
new = gen[len(ask):]
want, b = reference(gen)
owed = [(i, w) for i, w in enumerate(want[:-1]) if w is not None]
bad = [i for i, w in owed if gen[i + 1] != w]
n_chip = sum(1 for i in new if i >= VQ)
print(f"  {len(new)} new tokens in {dt:.0f}s ({dt / len(new) * 1e3:.0f} ms per token): {show(new)!r}")
fired = SHA in new
print(f"  the circuit owned {len(owed)} of them, {len(bad)} differ from the reference; {n_chip} chip tokens")
assert fired, "the language model did not write a call the circuit answers"
assert not bad
payload = show(new).split("<parameter=hex>")[1].split("</parameter>")[0].strip()
digest = hashlib.sha256(bytes.fromhex(payload)).hexdigest()
text = show(new)
assert f"<tool_response>\n{digest}\n</tool_response>" in text, text
answer = text.split("</think>")[-1].strip()
print(f"  payload {payload}: printed digest == hashlib ({digest[:16]}..)")
print(f"  the language model's answer repeats the digest {'exactly' if digest in answer.replace(' ', '') else 'with errors (its own copying)'}: {answer!r}")
if (work / "base_answer.pt").exists():  # the base's answer to the same text with no trace in its context
    ba = torch.load(work / "base_answer.pt")
    text_only = [i for i in gen if i < VQ]
    stops = set(model.generation_config.eos_token_id)
    if text_only[:len(ba["context"])] == ba["context"]:
        mine, theirs = until_stop(text_only[len(ba["context"]):], stops), until_stop(ba["answer"], stops)
        same = sum(1 for a, b_ in zip(mine, theirs) if a == b_)
        print(f"  that answer against the base's answer to the same call and response as plain text (no trace, no chip): "
              f"{same} of {max(len(mine), len(theirs))} tokens equal")
    else:
        print("  (the call differs from the canonical one the base was given: answers not compared)")

# ------------------------------------------------------------------ text
print("owner and text fidelity:")
base = torch.load(work / "base.pt")
ks, agree, chip_best, text_low, n = [], [], [], 1e9, 0
for ids, lp_base, top in zip(base["ids"], base["logp"], base["top"]):
    lg = all_logits(model, ids)
    chip_best.append(lg[:, VQ:].max().item())
    text_low = min(text_low, lg[:, :VQ].max(-1).values.min().item())
    lp = torch.log_softmax(lg, -1)[:, :VQ]  # the chip tokens are in the softmax: their mass shows up as KL
    ks.append(kl(lp_base.float(), lp))
    agree.append(lg.argmax(-1) == top)
    n += len(ids)
ks, agree = torch.cat(ks), torch.cat(agree).float()
print(f"  {n} text positions: best chip logit anywhere {max(chip_best):.3g}, lowest winning text logit {text_low:.3g}")
assert max(chip_best) < -1000
print(f"  KL(base || merged) mean {ks.mean():.2e}, max {ks.max():.2e}; top-1 agreement {agree.mean():.4f}")
if (work / "floor.json").exists():
    floor = json.load(open(work / "floor.json"))
    print(f"  floor, the base against itself in {floor.get('other', 'float32')}: KL mean {floor['kl_mean']:.2e}, "
          f"max {floor['kl_max']:.2e}; top-1 agreement {floor['top1']:.4f}")
plain_probe()

print("a trace in the context (458 chip tokens between the chat and more text):")
tail = tok(TAIL)["input_ids"]
_, trace = run(chat, b"abc")
with torch.no_grad():  # Qwen's own recurrent state, before and after the trace, continuing from the cache
    o = model(input_ids=torch.tensor([chat], device=DEV), use_cache=True)
    lin = [i for i, l in enumerate(o.past_key_values.layers) if type(l).__name__ == "LinearAttentionLayer"]
    before = {i: [t.clone() for t in o.past_key_values.layers[i].recurrent_states.values()] for i in lin}
    o2 = model(input_ids=torch.tensor([trace], device=DEV), past_key_values=o.past_key_values, use_cache=True)
    after = {i: list(o2.past_key_values.layers[i].recurrent_states.values()) for i in lin}
eq = {i: all(torch.equal(a, b_) for a, b_ in zip(before[i], after[i])) for i in lin}
qwen = [i for i in lin if i >= info["chip_layers"]]
print(f"  Qwen's recurrent state after the trace: bit-identical in {sum(eq[i] for i in qwen)} of its {len(qwen)} DeltaNet layers "
      f"(the chip's own {info['chip_layers']} layers: changed in {sum(not eq[i] for i in lin if i < info['chip_layers'])})")
assert all(eq[i] for i in qwen)


def tail_logp(ids, start, pos=None):
    with torch.no_grad():
        kw = {} if pos is None else {"position_ids": torch.tensor([pos], device=DEV)}
        h = model.model(input_ids=torch.tensor([ids], device=DEV), **kw).last_hidden_state[0, start:]
        return torch.log_softmax(model.lm_head(h).float().cpu(), -1)[:, :VQ]


plain = tail_logp(chat + tail, len(chat))
shifted = tail_logp(chat + tail, len(chat), list(range(len(chat))) + [len(chat) + len(trace) + i for i in range(len(tail))])
traced = tail_logp(chat + trace + tail, len(chat) + len(trace))
k_all, k_pos, k_chip = kl(plain, traced), kl(plain, shifted), kl(shifted, traced)
print(f"  the next {len(tail)} text tokens, KL(without the trace || with it): mean {k_all.mean():.2e}, max {k_all.max():.2e}; "
      f"top-1 equal at {int((plain.argmax(-1) == traced.argmax(-1)).sum())}")
print(f"    of which the positions alone (the same text moved {len(trace)} places, no trace): mean {k_pos.mean():.2e}")
print(f"    and the chip tokens at equal positions: first 3 tokens {k_chip[:3].mean():.2e} (Qwen's 4-tap conv window "
      f"holds chip tokens), the rest {k_chip[3:].mean():.2e}")
print("OK")
