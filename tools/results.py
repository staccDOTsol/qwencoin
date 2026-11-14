"""Assemble RESULTS.md from logs/ (the raw output of the checks) and the artifacts' hashes.

    python3 tools/results.py > RESULTS.md
"""
import hashlib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "logs"


def read(name):
    p = L / name
    return p.read_text().rstrip() if p.exists() else ""


def sha(path):
    p = ROOT / path
    if not p.exists():
        return "(not built)"
    h = hashlib.sha256()  # file checksums only; nothing here is consensus
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block(text):
    return "```\n" + text + "\n```"


def table(log, kinds):
    rows = []
    for line in read(log).splitlines():
        m = re.match(r"(\S+): (?:(\d+) cases, )?(.*?) \[(\d+)s\]$", line) or re.match(r"(\S+): (?:(\d+) cases, )?(.*)()$", line)
        if not m:
            continue
        name, cases, res, secs = m.groups()
        short = res[:60].rstrip() + " ..." if res.startswith("FAILED") and len(res) > 100 else res
        if name == "bip324_tests" and res.startswith("FAILED"):
            short = "FAILED: the vectors are bound to Bitcoin's network magic (section 9)"
        rows.append(f"| `{name}` | {cases or ''} | {short} | {secs + ' s' if secs else ''} |")
    return f"| {kinds} | cases | result | time |\n|---|---|---|---|\n" + "\n".join(rows)


def tally(log):
    lines = read(log).splitlines()
    ok = [l for l in lines if re.search(r": (\d+ cases, )?ok \[", l)]
    cases = sum(int(m.group(1)) for l in ok for m in [re.search(r": (\d+) cases", l)] if m)
    return len(ok), len(lines), cases


out = []
w = out.append
w("# Results\n")
w("One machine: 2 cores, 8 GB, no GPU. Ubuntu 24.04, gcc 13.3, rustc 1.97, torch 2.14.1 (CPU), "
  "transformers 5.22.2. `logs/` holds the raw output; this file is `tools/results.py` over it.\n")
if read("pending.txt"):
    w("**Still running when this file was generated:** " + read("pending.txt") + "\n")
w("| artifact | sha256 |\n|---|---|")
for path, note in [("dist/qwensha/model.safetensors", "the checkpoint, 656 MB; `python -m hashformer.hf` rebuilds it bit for bit"),
                   ("exec/qsha_circuit.h", "the same nonzero weights, as the executor reads them"),
                   ("patches/bitcoin-v29.4-inq-qwencoin.patch", "applies to tag `v29.4-inq`"),
                   ("patches/ord-0.29.1-qwencoin.patch", "applies to tag `0.29.1`"),
                   ("patches/bitcoin_hashes-0.14.101-qsha.patch", "rust-bitcoin's SHA-256, routed to the model"),
                   ("patches/secp256k1-sys-0.10.1-qsha.patch", "ord's libsecp256k1, routed to the model"),
                   ("dist/qwork.bin", "the chain's work table: the MLP weights of Qwen3.5-0.8B at 16-bit fixed point"),
                   ("exec/qwork.c", "the work function's kernel, as the node links it"),
                   ("dist/qwen08-sha/model.safetensors", "Qwen3.5-0.8B with the circuit merged in, as sections 9 and 11 ran it"),
                   ("bin/qwencoind", "the node of sections 10 and 11 (sections 2 to 7 ran the node before the work function changed)"),
                   ("bin/ord", "the ord of section 11")]:
    w(f"| `{path}`{' (' + note + ')' if note else ''} | `{sha(path)}` |")

w("\n## 1. The model\n")
w("`logs/model-checks.log`: the checks in `tests/` on the compiler, the stock checkpoint and the executor.\n")
w(block(read("model-checks.log")) if read("model-checks.log") else "(not run in this pass; see README)")

w("\n## 2. Block hashes of the first chain, recomputed by the checkpoint\n")
w("(Sections 2 to 5 ran on this project's first chain, whose proof of work was the SHA-256d of the header; "
  "the current chain's work function is section 10.) `tests/check_hf_chain.py` loads `dist/qwensha` with stock "
  "transformers and double-hashes block headers taken from node 1 with plain `generate()`. The node computed the "
  "same hashes in its executor.\n")
w(block("\n".join(l for l in read("hf-chain.log").splitlines() if not l.startswith("[transformers]"))))

w("\n## 3. Three nodes, from genesis\n")
w("`tools/launch.py`: three `qwencoind` + `ord` pairs on localhost mine 118 blocks, then seven identities "
  "run the graph. Every block, transaction and script check on every node hashed in the weights, and so did "
  "ord (its index's txids, the inscriptions' commit and reveal transactions, their signatures) and the agents.\n")
w(block(read("launch.log")))

w("\n## 4. The slashing covenants, decided by consensus\n")
w("`tests/check_covenant.py` on the same chain (`testmempoolaccept`, then the confirmed burns):\n")
w(block("\n".join(l for l in read("covenant.log").splitlines() if l.strip() and not l.startswith(("real", "user", "sys")))))

w("\n## 5. Attacks on the accounting\n")
w("`tests/check_sybil.py` on the same chain; the three nodes' indexes must agree:\n")
w(block("\n".join(l for l in read("sybil.log").splitlines() if l.strip() and not l.startswith(("real", "user", "sys")))))

n_ok, n_all, n_cases = tally("unit-tests.log")
w("\n## 6. Bitcoin's unit tests on in-weights SHA-256\n")
w(f"`tools/unit.sh`, `test_bitcoin` built from the patched tree. {n_ok} of {n_all} filters pass ({n_cases} of the "
  "binary's 672 cases). Selected: the suites that test hashing, keys, scripts, sighashes, merkle trees and "
  "transactions directly; section 9 says what was left out and why.\n")
w(table("unit-tests.log", "filter"))

f_ok, f_all, _ = tally("functional-tests.log")
w("\n## 7. Bitcoin's functional tests against the node\n")
w(f"`tools/functional.sh` on regtest: the node hashes in the weights, the Python framework with hashlib and its "
  f"own secp256k1, so every txid, sighash and template hash is cross-checked. {f_ok} of {f_all} pass.\n")
w(table("functional-tests.log", "test").replace("| cases |", "| |").replace(" |  | ", " | | "))

w("\n## 8. What is in the binaries\n")
w("`tests/check_binary.py bin/qwencoind` and `tests/check_binary.py --ord bin/ord`. The node holds no SHA-256 "
  "but the circuit. ord's chain hashing and signing hold none either; its TLS stack and the `sha2` crate keep theirs.\n")
w(block(read("binary.log")))

w("\n`rust/qsha-check` runs the two patched crates directly: digests against the independent `sha2` crate, a "
  "tagged hash, HMAC, a block header, the BIP340 signing vector (its three tagged hashes run inside libsecp256k1).\n")
w(block(read("rust-crates.log")))

w("\n## 9. The chip inside Qwen3.5: the merge and the bridge\n")
w("`tests/check_bridge.py`: the merged circuit (the hasher plus what reads Qwen's tool call, replays the payload "
  "and prints the answer) in exact arithmetic against its reference machine, `hashformer/bridge.py`. The text "
  "tokens are Qwen3.5's real ids; a stream runs from the chat prompt to the text after the answer.\n")
w(block(read("bridge-ir.log")))
w("\n`hashformer.merge` on Qwen3.5-0.8B, then `tests/check_merge.py` on the result: stock transformers, bf16, "
  "this machine's CPU. The free-running part is one greedy `generate()` from the user's question.\n")
w(block(read("merge-0.8B.log")))
if read("merge-4B.log"):
    w("\nQwen3.5-4B, merged and run on a second machine (`mac/setup.sh`; its probe of which kernels run the "
      "chip exactly is `logs/backend-mac.json`):\n")
    if read("backend-mac.txt"):
        w(block(read("backend-mac.txt")))
        w("")
    w(block(read("merge-4B.log")))

def clean(name):
    return "\n".join(l for l in read(name).replace("\r", "\n").splitlines()
                     if l.strip() and not l.startswith(("[transformers]", "Loading weights", "real", "user", "sys")))


w("\n## 10. Proof of work bound to the weights\n")
w("`tests/check_work.py`: the C kernel the node links against the exact numpy reference, and the statistics of the "
  "function over the real table (`dist/qwork.bin`, built from Qwen/Qwen3.5-0.8B).\n")
w(block(clean("work.log")))
w("\n`pow_tests/qwork_*` in the node's own unit-test binary (`QWORK_TABLE=dist/qwork.bin`); the node's SHA-256 "
  "there is the compiled circuit, the expected values come from the reference with hashlib:\n")
w(block("\n".join(l for l in clean("unit-qwork.log").splitlines() if "test case" in l or "mined a header" in l or "errors" in l.lower())))
w("\n`tests/check_qwork_node.py`: a private network of the chain, four `qwencoind`.\n")
w(block(clean("qwork-node.log")))
w("\nBitcoin's functional tests on the node with the new rule (regtest keeps Bitcoin's own):\n")
w(table("functional-qwork.log", "test").replace("| cases |", "| |").replace(" |  | ", " | | "))

w("\n## 11. The model's functions on a live chain\n")
w("`tests/check_tools.py` on a private two-node network of the chain (real blocks under the weights-bound rule, "
  "real ord, real coins of that network). First the node side with no model, then the chat template, then "
  "`dist/qwen08-sha` asked in plain words (greedy, one question per fresh context, all 23 functions declared), "
  "then one round of a node's model loop.\n")
w(block(clean("tools-build.log") or clean("tools.log")))

w("\n## 12. The merge's newer paths\n")
w("`tests/check_remerge.py`: the merge as it is now against the checkpoint section 9 ran, tensor by tensor, and a "
  "sharded checkpoint against a single file.\n")
w(block(clean("remerge.log")) if read("remerge.log") else "(not run)")
w("\n`tests/check_untied.py`: the path the 9B and 27B bases need (an output matrix of their own), on a 0.8B given one.\n")
w(block(clean("untied.log")) if read("untied.log") else "(not run)")
w("")
w((ROOT / "tools" / "results-notes.md").read_text().rstrip())
sys.stdout.write("\n".join(out) + "\n")
