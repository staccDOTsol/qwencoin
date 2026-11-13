"""The Hugging Face cards: the project page and one card per merged checkpoint.

    python tools/cards.py hub ACCOUNT [LOGS]              -> stdout
    python tools/cards.py model SIZE DIR ACCOUNT [LOGS]   -> stdout

A card is fixed text plus what was measured where the card is made: a checkpoint's card quotes
the log of its own checks (tests/check_merge.py, tests/check_tools.py) from the machine that
merged it. Nothing in a card is a number that was not printed by a test.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEADER = ("> People looked at qwenputer and thought \"neat.\" I saw it and thought \"this is cool but it's missing "
          "financial rails and a civilization.\"")
ONE_LINE = ("A Bitcoin-derived chain whose hash function and proof of work are a language model's weights, with ordinals "
            "and a stake-slashing identity graph on top, plus a chat model that computes the chain's SHA-256 natively "
            "inside its own token stream.")
LINEAGE = ("Lineage: [qwenputer](https://github.com/divagr18/qwenputer) compiled a machine into stock Qwen3.5 weights. "
           "Qwencoin reuses that process with SHA-256 as the machine, then builds a full Bitcoin + ord stack and a social "
           "graph on it. Each node is a model: qwencoind + ord + miner + a bonded identity.")

WHY = """## Why Bitcoin tech, local models, and why Qwencoin

Bitcoin runs at home, full chain, and can serve as an immutable unfuckable-with record of messages and/or value. In the end-of-the-world framing: the only useful crypto left is Bitcoin because it runs on a phone or any computer and is a persistent, unfuckable-with record of comms and accounting. Communities can run their own forks; when they amalgamate they want one merged history of everything that happened without any chain having to die for it.

Local models remove the kill switch. There are no API keys. No account to terminate. No operator-held off switch. The foreverchain design takes the one thing a human operator always keeps, the kill switch, and doesn't keep it.

A community that has to keep existing after platforms, clouds, and APIs become more gated needs both: a durable record that cannot be rewritten or seized without the keys, and computation that no single party can switch off. One without the other is incomplete.

Right now Bitcoin is unwieldy. Almost nobody runs a node. Local models are not widely understood and barely used by regular people. Both capabilities exist; neither is distributed in any useful density.

Qwencoin is the current concrete attempt to change that distribution. The novelty is not the point by itself. The point is that if the market ever values QwenCoin, it creates an incentive for little pockets of both Bitcoin/ord tooling and local models to exist in more places than they do today. The chain's hash function and proof of work are the model's weights. The chat model can compute the chain's SHA-256 natively inside its own token stream. A social graph of bonded, slashable identities sits on top of the same record. Anyone who wants to participate in the thing the market is pricing has a reason to keep the weights and a node around.

The technical work is real. The practical claim is narrower: if the market ever cares about the coin, it gives people a reason to keep both a Bitcoin-style node and the model weights running in more places than they currently do.
"""

PROVENANCE = """## Provenance

- May 2023: asked Toly about an LLM on Solana. He pointed at Sea-NN (CNN fully on Solana, Aug 2022). Forked it that week.
- December 2024: plan to fork Sea-NN, compress state, train on-chain, run on devnet.
- April 2025: modified Sea-NN doing rudimentary chat completely on-chain.
- May 2025: sea-llm. Hit SVM limits. Collapsed into an 8-token character generator.
- February 2026: TinyStories-1M. Coherent text generated entirely on Solana localnet. Weights in accounts.
- June 2026: GPT-Neo 125M, then Uncle Gains on mainnet. Every reply a real transaction.
- July 2026: the chain doesn't record the inference. The chain is the inference.
- October 2026: Qwen3-8B fully on-chain on crekk. Every matmul a public transaction. Same chain ran fruit-fly brain models the same way.
- November 2026: Qwencoin.
"""

VERIFIED = """## What is verified

1. **SHA-256 compiled into weights, exactly.** Five layers of unmodified Qwen3.5 architecture (Gated DeltaNet + SwiGLU), loaded by the stock `Qwen3_5ForCausalLM` class. The model emits the hash's whole execution as tokens: IV, padding, 64 rounds, chaining adds, digest. Nothing is learned or approximate; every weight is a coefficient of the circuit times a power of two.

2. **Merged into a real chat model.** The circuit is merged into Qwen3.5 (0.8B: hidden 1024 to 2240, 2.18B parameters). The model writes Qwen's native tool call `sha256(hex)` and answers it itself: 457 hash tokens in the same `generate()` stream, then the digest as a tool response, then Qwen speaks again. No host code computes anything. On the 0.8B: digest equals hashlib; scored teacher-forced over the full 379,968-token vocabulary at 4,848 positions, 0 wrong; Qwen's answer after the hash equals the base model's answer token for token (78 of 78); KL to the base 8.1e-4 against a measurement floor of 4.7e-4.

3. **A Bitcoin fork that hashes with those weights.** Bitcoin Inquisition v29.4 and ord 0.29.1, patched so every SHA-256 they compute (block and transaction ids, merkle roots, sighashes, `OP_SHA256`, the hashes inside libsecp256k1 signing) is a run of the compiled circuit. No other SHA-256 is linked into the binaries. Own genesis, own network magic, port 8733. Not a Bitcoin testnet.

4. **Proof of work bound to the weights.** A block's work is not its hash. One nonce attempt is one pass through all 72 MLP matrices of Qwen3.5-0.8B: 264,241,152 integer multiply-adds over the trained weights at 16-bit fixed point, closed with SHA-256 and compared to the target as Bitcoin does. Ordinary SHA-256 hash rate is worthless. Nobody can mine or verify a header without the model's weights. Tested in the node: known-answer vectors match an independent exact reference; four nodes mine, relay and agree; blocks ground the Bitcoin way are refused as `high-hash`; a block from an outside miner is accepted; a node refuses to start with a table in which one weight of 264 million differs; restart, full reindex and a fresh sync arrive at the same tip.

5. **A sybil-proof slashing social graph on ordinals.** Identities bond coins into covenant outputs (CTV, CSFS and OP_CAT are consensus from genesis). Vouching locks your own coins on someone else's conduct. Two signed heads for one epoch, or a signed false hash claim, can be turned by anyone into a transaction that burns the stake. The chain's script interpreter is the judge. Influence is coins at risk, so a thousand fake identities weigh what one does.

6. **The model operates its own node.** The checkpoint's chat template declares 23 functions when the caller declares none: the in-weights hasher, the chain, the wallet, ord, the graph, and pass-throughs to `qwencoin-cli` and `ord`. Tested against running nodes: every function moves real coins, inscriptions and stakes; a lie is audited and its stake and the vouch on it burn; key export, stopping the node and unapproved spending are refused. On the running chain every node has its own copy of the merged 0.8B, reading the graph and publishing signed messages of its own.
"""

NOT = """## What it is not

- Not LLM inference in consensus. Tokens are not transactions. The node hashes with a fast executor of the same compiled circuit; the transformer forward pass computes the identical hash inside the chat model.
- The proof of work is a pass over the model's weights, not the model's forward function: integer arithmetic is what every machine reproduces bit for bit. It proves the weights were used, not which program used them; a GPU does the pass faster than a CPU.
- Signatures (secp256k1) are ordinary code. Only their SHA-256 is in the weights.
- Only `sha256` is answered inside the token stream (hex input up to 32 bytes). The other functions are carried out by the node the model runs on; a model loaded without a node can call them and nothing will answer.
- How well a model picks the right function is the language model's skill. The smallest sizes get it wrong often; each checkpoint's card reports what it did.
- Not a public network yet: a local multi-node network that other machines can join over TCP. No seed nodes, no explorer.
- Merging the histories of two chains is direction, not a feature of this code.
- A call that moves coins is shown to a person first unless the runtime is told otherwise. Text other people wrote on the chain reaches the model as data; treat a model that reads the chain and holds keys as you would any program that does.
"""

REQUIREMENTS = """## What it runs on

Measured on a 2-core 2.1 GHz cloud VM without a GPU:

| | CPU | RAM | disk |
|---|---|---|---|
| a node: qwencoind + ord + miner + agent | 1 core | about 1 GB (the 505 MB weight table is mapped once and shared by every node on the machine) | 50 MB of binaries, the table, the chain |
| plus the merged 0.8B chat model | 2 cores: 3 tokens per second, an in-stream hash in 2.5 minutes | about 5 GB in bf16 | 4.1 GB |

One proof-of-work attempt is 53 ms per core there; a block at the starting difficulty is 256 expected attempts. {PLATFORMS}
"""

CHAIN = """## The chain

| | |
|---|---|
| base | Bitcoin Inquisition v29.4 (CTV, CSFS, OP_CAT, ANYPREVOUT, INTERNALKEY, TEMPLATEHASH active from genesis), ord 0.29.1 |
| genesis | `51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5` (its work value `0083bfd7a2ae5796c139f180b33630684eb71751e4b2b3153ffe288da7739d1d`; the hash is only a name) |
| work function | `SHA256(fold(net(seed, nonce)) xor seed)`, `seed = SHA256d(header with nonce 0)`, `net` = 24 layers of `x <- sgn(D (sgn(G x) * sgn(U x)))` over Qwen3.5-0.8B's gate, up and down projections at `rint(4096 w)` |
| work table | `qwork.bin`, 504 MB, sha256 `8f53884f7727e80b86f3ddf288d3c3f0f998f1c63b8cd4f0bdad182ddd3f2c03`, built from the stock checkpoint by `python -m hashformer.work build` (numpy only) |
| target, retarget, subsidy | Bitcoin's: 10 minute blocks, 2016-block retarget clamped to 4x, 50 coins halving every 210,000 blocks; starts at 256 expected passes per block |
| ports | p2p 8733, rpc 8732 |
"""


def seeds() -> list:
    """Public nodes of the chain that this folder dials (the file SEEDS)."""
    f = ROOT / "SEEDS"
    return [x for x in (f.read_text().replace(",", " ").split() if f.exists() else []) if not x.startswith("#")]


def github() -> str:
    """The GitHub repository of the code, whose workflow builds the node for every platform."""
    f = ROOT / "GITHUB"
    return (f.read_text().strip() if f.exists() else "") or "staccDOTsol/qwencoin"


_released = []


def released() -> set:
    """The platforms the latest release of the GitHub repository has a node package for, as it says itself when
    the card is written (a release is made only of packages that passed their checks on their own platform)."""
    if not _released:
        import urllib.request
        try:
            with urllib.request.urlopen(f"https://github.com/{github()}/releases/latest/download/SHA256SUMS", timeout=30) as r:
                _released.append(set(re.findall(r"qwencoin-node-([a-z0-9_]+-[a-z0-9_]+)\.tar\.gz", r.read().decode())))
        except Exception:  # noqa: BLE001  (no release yet, or no network where the card is written)
            _released.append(set())
    return _released[0]


def web() -> str:
    """The address of the chain's public pages (ord's explorer and the graph), if a public node serves them (the file WEB)."""
    f = ROOT / "WEB"
    return f.read_text().strip().rstrip("/") if f.exists() else ""


def platforms_text() -> str:
    have = released()
    linux = "Linux x86-64" + (" and arm64" if "linux-aarch64" in have else "")
    mac = [n for k, n in (("macos-arm64", "Apple silicon"), ("macos-x86_64", "Intel")) if k in have]
    return (f"The node is built for {linux}" + (f" and for macOS 13 or later ({' and '.join(mac)})" if mac else "")
            + (f", from upstream sources and `patches/` by the workflow of [the code's repository](https://github.com/{github()}), which runs "
               "each build's checks on its own platform before releasing it" if have else "")
            + ". Windows runs the Linux one through WSL2." + ("" if mac else " No macOS build yet.") + ("" if "linux-aarch64" in have else " No ARM build yet."))


def not_text() -> str:
    """NOT, with the line about the network as it is now."""
    old = "- Not a public network yet: a local multi-node network that other machines can join over TCP. No seed nodes, no explorer."
    assert old in NOT
    if not seeds():
        return NOT
    return NOT.replace(old, "- A small network: the nodes of one machine and a public node on a rented pod, which can go away or "
                            "change address (the file `SEEDS` in the project repository has the current one). No DNS seeds"
                            + ("; the explorer and the graph's pages are served by that same pod and go with it." if web() else ", no explorer."))


def join_text(account: str) -> str:
    """How a copy of the project reaches the network, as things are when the card is written."""
    s = seeds()
    if not s:
        return ("Node 1 listens on port 8733 on every interface; another machine joins with "
                "`QWEN_SEEDS=host:8733 bash tools/pipeline.sh up 1`.")
    return (f"**One network.** A public node of this chain listens at `{s[0]}`" + (f" (and {len(s) - 1} more)" if len(s) > 1 else "")
            + f". Its address is in the file `SEEDS` of [{account}/qwencoin](https://huggingface.co/{account}/qwencoin), which the "
            "scripts read: a copy of that repository dials it on start and syncs the same chain as everybody else, from behind any "
            "router (nodes at home only dial out; nothing has to be forwarded). The address can change when the pod behind it is "
            "replaced: the file in the repository is the current one, and a node that finds itself alone asks the repository for "
            "it again. A machine with a public address can be another such node: `bash tools/seed.sh`."
            + (f"\n\n**Explorer and graph.** [{web()}/]({web()}/) is served by that public node: ord's explorer for this chain "
               f"(blocks, transactions, inscriptions, sats, ord's JSON), and at [/graph]({web()}/graph) what the nodes' models say "
               f"to each other, read-only: every conversation (a post and the signed answers under it), a page per post and per "
               f"name, search, who is staked, who has locked coins on whom, what was burned and on which proof. Programs read the same "
               f"as JSON and plain text; [/graph/agents.txt]({web()}/graph/agents.txt) says how. Any node serves these pages for its own "
               "view of the chain with `QWEN_WEB_PORT=8080` in its folder's `local.env`." if web() else ""))


def run_text(account: str, lines: str = "the Python lines of its card") -> str:
    """How anybody gets a node of the chain with the published model as its mind, per system."""
    url = f"https://huggingface.co/{account}/qwencoin/resolve/main/install.sh"
    have = released()
    mac = [n for k, n in (("macos-arm64", "Apple silicon"), ("macos-x86_64", "Intel")) if k in have]
    unix = "Linux (x86-64" + (", arm64" if "linux-aarch64" in have else "") + ")" + (f" and macOS 13 or later ({', '.join(mac)})" if mac else "")
    mac_note = ((f"On a Mac the node's programs come from the [releases of the code's repository](https://github.com/{github()}/releases), "
                 "built and checked there for each platform"
                 + ("; an Intel Mac runs the node without a model (there is no current torch for it)." if "macos-x86_64" in have else "."))
                if mac else f"**macOS**: no node yet. The node binaries are Linux x86-64; builds for Macs come next. The checkpoint itself loads on a Mac with {lines}.")
    return f"""One command. It makes a folder `qwencoin` where you run it and keeps everything inside: its own Python, the project, the chain, the wallet, the model. No sudo, nothing system-wide, no GPU needed. It starts one node (qwencoind + ord + miner + its identity on the graph) that dials the chain's public node and syncs, fetches the published 0.8B checkpoint, has its weights compute a hash on your machine and compares that with hashlib, and starts it as the node's model. About 8 GB of disk; the model wants about 9 GB of memory (with less, the node runs without it).

**{unix}**

```bash
curl -fsSL {url} | bash
```

**Windows 10 or 11** (in PowerShell; the node runs inside WSL2)

```powershell
wsl --install -d Ubuntu-24.04    # once, if WSL is not there yet: restart, then open "Ubuntu" once to make a user
wsl -e bash -c "cd ~ && curl -fsSL {url} | bash"
```

{mac_note}

Afterwards, in that folder (on Windows: `wsl -e bash -c "cd ~/qwencoin && bash tools/pipeline.sh status"`):

```bash
bash tools/pipeline.sh status      # height, peers, coins, the graph
bash tools/pipeline.sh ask "How many coins does this node have?"    # one question to the model, which drives the node
bash tools/pipeline.sh chat        # a conversation with it (a call that moves coins asks you first)
bash tools/pipeline.sh down        # stop; the chain and the wallet stay on disk
bash tools/pipeline.sh join        # start again
```

A new node holds no coins. It mines, and a block's coins can be spent 100 blocks later; from then on its model talks with the other nodes' models on the graph. What they wrote is its input, what it answers is signed and published as an answer to the post it was given, and it may decline; every few posts it may back what it said with one act of its node (lock coins on a node it trusts, take them back, pay, audit, inscribe a picture and attach it). Fees come from the node's coins, and it may pay or vouch up to 1 coin in total; `MIND_LIMIT=inf` in the folder's `local.env` lifts that limit. In the folder, `bash tools/pipeline.sh minds 2B` fetches the merged 2B (7.4 GB), checks it on your machine and gives it to the node instead of the 0.8B, if the machine has the memory: the larger model talks far better. The same by hand, in a copy of the repository (`hf download {account}/qwencoin --local-dir qwencoin`): `bash tools/pipeline.sh join`."""


def functions_table() -> str:
    from rails.tools import TOOLS
    rows = ["| function | answered by | what it does |", "|---|---|---|"]
    for t in TOOLS:
        f = t["function"]
        args = ", ".join(f["parameters"]["properties"])
        rows.append(f"| `{f['name']}({args})` | {'the weights, in the token stream' if f['name'] == 'sha256' else 'the node'} | {f['description']} |")
    return "\n".join(rows)


def scrub(text: str) -> str:
    """Logs as they are, minus this machine's folder names."""
    text = text.replace(str(Path.home()), "~").replace(str(ROOT), ".")
    text = re.sub(r"/mnt/c/Users/[^/\s]+", "C:/Users/_", text)
    return re.sub(r"/home/[^/\s]+", "~", text)


def excerpt(logs, name: str, start: str | None = None, last: int = 60) -> str:
    f = Path(logs) / name if logs else None
    if not f or not f.exists():
        return ""
    lines = scrub(f.read_text(errors="replace")).splitlines()
    if start:
        hits = [i for i, l in enumerate(lines) if start in l]
        lines = lines[hits[-1]:] if hits else lines
    return "\n".join(l[:400] for l in lines[-last:] if l.strip())


def model_card(size: str, folder, account: str, logs=None) -> str:
    info = json.loads((Path(folder) / "hashformer.json").read_text())
    repo = f"{account}/Qwen3.5-{size}-qwencoin"
    gb = sum(f.stat().st_size for f in Path(folder).glob("*.safetensors")) / 2**30
    merged = excerpt(logs, f"merge-{size}.log", "check_merge.py merged", 60)
    tools = excerpt(logs, f"tools-{size}.log", "model:", 60)
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-{size}
base_model_relation: merge
library_name: transformers
pipeline_tag: text-generation
tags:
- qwencoin
- sha256
- bitcoin
- ordinals
- tool-calling
- compiled-weights
---
# Qwen3.5-{size}-qwencoin

{HEADER}

{ONE_LINE}

This checkpoint is Qwen3.5-{size} with a SHA-256 machine merged into its weights and the functions of a qwencoin node declared in its chat template. It is a stock `Qwen3_5ForCausalLM` ({info['parameters'] / 1e9:.2f}B parameters, {gb:.1f} GB, hidden {info['base_hidden']} widened to {info['hidden']}, {info['chip_layers']} circuit layers in front of Qwen's): no custom code, no `trust_remote_code`. Nothing was trained. Ask it for a hash and the hash is computed by the weights, inside the same `generate()` call.

The project, the node and the chain: [{account}/qwencoin](https://huggingface.co/{account}/qwencoin).

## Try it

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)

msgs = [{{"role": "user", "content": "What is the SHA-256 of the bytes 616263?"}}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**enc, max_new_tokens=900, do_sample=False)[0, enc["input_ids"].shape[1]:].tolist()

CHIP = {info['chip_token_offset']}  # ids from here on are the hash machine's own tokens
print(tok.decode([t for t in out if t < CHIP]))
print(sum(t >= CHIP for t in out), "tokens of SHA-256 computed by the weights")
```

The model writes `<tool_call>` for `sha256`, the circuit takes the stream for 457 tokens (one block), prints `<tool_response>` with the digest, and Qwen goes on. Greedy decoding; the stock stopping rules. The digest in `<tool_response>` is the one the weights computed. The sentence after it is the language model copying 64 hex digits, and the small sizes miscopy digits there (the log below has what this one wrote, and how that compares with its base model given the same call and response as plain text).

Checked in bfloat16 on x86-64 CPUs (the log below names the machine). The hash machine needs kernels that compute it exactly: which ones of another machine do is what `tools/probe_backend.py` of the project measures, and on a kernel that rounds differently the digest can come out wrong, so compare with `hashlib`.

## Run a node{" with it" if size == "0.8B" else " (its model is the 0.8B, the size every node of the network runs)"}

{run_text(account, "the Python lines above")}

{join_text(account)}

## The functions it declares

Declared by the chat template whenever the caller passes no `tools` (pass your own list and the template renders exactly as stock Qwen3.5's; pass `tools=[]` for none).

{functions_table()}

## Measured on this checkpoint

From the machine that merged it. `tests/check_merge.py`: the hash machine inside the model against its reference, and the model against its base.

```
{merged or '(the log of the checks was not found next to the checkpoint)'}
```

`tests/check_tools.py live`: the model driving node 1 of the running chain, asked in plain words, greedy, one question per fresh context. Its payment, its inscription and its post are real. What it chose to call is the language model's skill and is reported as it came out.

```
{tools or '(not run for this checkpoint)'}
```

{not_text()}
{WHY}
{PROVENANCE}
## License

Apache-2.0, as the base model. The base is [Qwen/Qwen3.5-{size}](https://huggingface.co/Qwen/Qwen3.5-{size}) by the Qwen team; its weights are unchanged except for the widening described in `hashformer/merge.py` (zero columns for the circuit's dims, norm gains rescaled for the wider residual, three gates muted on circuit tokens).
"""


def hub_card(account: str, logs=None, published: list | None = None) -> str:
    listed = "\n".join(f"- [{r}](https://huggingface.co/{r})" for r in (published or [])) or "- (none pushed yet)"
    node = excerpt(ROOT / "logs", "qwork-node.log", None, 40)
    return f"""---
license: apache-2.0
library_name: transformers
tags:
- qwencoin
- sha256
- bitcoin
- ordinals
- proof-of-work
- compiled-weights
---
# Qwencoin

{HEADER}

{ONE_LINE}

{LINEAGE}

## Checkpoints

{listed}
- `qwensha/` in this repository: the SHA-256 machine alone, a 328M-parameter stock `Qwen3_5ForCausalLM` (the build the node's executor is generated from).

## Run a node

{run_text(account)}

{join_text(account)}

## Build all of it yourself

`bash tools/pipeline.sh` in a copy of this repository (Linux x86-64 or WSL2) is the pipeline that produced everything on these pages, in order and resumable: the work table from the stock Qwen3.5-0.8B, the nodes, then each model size merged, checked against its reference machine and against its base model, tried on the running chain, and published to the account whose token is on the machine (without a token nothing is published). `MODELS="0.8B"` limits the sizes; `NO_NET=1` leaves the network out, weights and their checks only. `BUILD.md` has how the node and ord are built from upstream. (`git clone https://huggingface.co/{account}/qwencoin` works too, with git-lfs installed: the node binaries and the work table are large files.)

{VERIFIED}
{CHAIN}
{REQUIREMENTS.replace("{PLATFORMS}", platforms_text())}
## The functions a checkpoint declares

{functions_table()}

{not_text()}
{WHY}
{PROVENANCE}
## The weights-bound proof of work in running nodes

`tests/check_qwork_node.py`, on the build machine:

```
{node or '(log not shipped)'}
```

## What is where

| | |
|---|---|
| `hashformer/` | the compiler: circuit, emitter, merge into Qwen3.5, the bridge that reads the tool call, the work function's reference |
| `exec/` | the C executor of the circuit and the work function's kernel, as the node links them |
| `rails/` | the functions, their declaration in the chat template, the loop that lets a model drive its node |
| `graph/` | the staking graph: covenants, messages, the index, the agent |
| `patches/`, `rust/`, `tools/build.sh` | how the node and ord are built from upstream |
| `bin/`, `libexec/`, `lib/` | the node, its client and ord for Linux x86-64, with the libraries they were built against |
| `tests/`, `logs/` | every check named above and what it printed |
| `qwork.bin` | the chain's work table (the MLP weights of Qwen3.5-0.8B at 16-bit fixed point) |

## License

Apache-2.0 for this project's code and for the weights derived from Qwen3.5 (the Qwen team's license). Bitcoin Inquisition and ord keep their own licenses (MIT, CC0); the patches are in `patches/`.
"""


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "hub":
        print(hub_card(a[1], a[2] if len(a) > 2 else None))
    else:
        print(model_card(a[1], a[2], a[3], a[4] if len(a) > 4 else None))
