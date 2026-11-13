"""Publish to Hugging Face what has passed its checks.

    python tools/publish.py whoami                 the account the token on this machine belongs to
    python tools/publish.py hub LOGS               the project        -> ACCOUNT/qwencoin
    python tools/publish.py model SIZE DIR LOGS    a merged checkpoint -> ACCOUNT/Qwen3.5-SIZE-qwencoin
    python tools/publish.py pending LOGS           every merged size that passed and is not published yet
    python tools/publish.py cards LOGS             the code, the file SEEDS and every card (the project's and each published
                                                   checkpoint's) as things are now; no weights are sent

The token is read on this machine and goes to huggingface.co only: $HF_TOKEN, or the first of
~/.hugginface.key, ~/.huggingface.key, the same two names in a Windows user's home
(/mnt/c/Users/*/) or next to go.cmd, the Hugging Face CLI's token file. It is never printed,
logged or copied. Without a
token nothing is published and the command says so (exit 0: the rest of the pipeline goes on).

A checkpoint is published only if its checks passed here (.done/merged-SIZE, and .done/tools-SIZE
unless NO_TOOLS is set); its card quotes the logs of those checks. Repositories are public and
are added, as they land, to one collection on the account ("Qwencoin").
"""
import os, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import cards  # noqa: E402

DIRS = {"0.8B": "dist/qwen08-sha", "2B": "dist/qwen2b-sha", "4B": "dist/qwen4b-sha", "9B": "dist/qwen9b-sha", "27B": "dist/qwen27b-sha"}
# bin/ libexec/ lib/: the node programs for Linux x86-64; node/PLATFORM/: the same for every other platform they are built for
SOURCE = ["hashformer", "rails", "graph", "tools", "tests", "exec", "patches", "rust", "vendor", "bin", "libexec", "lib", "node"]


def token() -> str | None:
    t = os.environ.get("HF_TOKEN", "").strip()
    if t:
        return t
    home = Path.home()
    names = (".hugginface.key", ".huggingface.key", ".hugginface.key.txt", ".huggingface.key.txt", "hugginface.key", "huggingface.key")
    places = [home / n for n in names] + [ROOT / n for n in names]
    win = Path("/mnt/c/Users")
    if os.environ.get("HF_TOKEN_FILE"):
        places.insert(0, Path(os.environ["HF_TOKEN_FILE"]))
    if (ROOT / "HF_TOKEN_FILE").exists():  # a file holding the PATH of the token file (Windows or Linux spelling)
        w = (ROOT / "HF_TOKEN_FILE").read_text().strip().strip('"')
        if len(w) > 2 and w[1] == ":":  # C:\Users\.. as WSL sees it
            w = "/mnt/" + w[0].lower() + w[2:].replace("\\", "/")
        places.insert(0, Path(w.replace("~", str(home), 1)))
    if win.is_dir():  # a Windows user's home, seen from WSL (and the folder this kit was started from)
        for n in names:
            places += sorted(win.glob(f"*/{n}"))
        for stem in ("hf", "huggingface", "hugginface", "hugging_face"):  # next to other keys, e.g. Documents\runpod_key.txt
            for kind in ("key", "token"):
                places += sorted(win.glob(f"*/Documents/{stem}_{kind}.txt")) + sorted(win.glob(f"*/{stem}_{kind}.txt"))
        places += sorted(win.glob("*/.cache/huggingface/token"))
    if os.environ.get("KIT_SRC"):
        places += [Path(os.environ["KIT_SRC"]).parent / n for n in names]
    places += [home / ".cache" / "huggingface" / "token", home / ".huggingface" / "token"]
    for p in places:
        t = first_line(p)
        if t:
            return t
    return None


def first_line(p) -> str | None:
    """The first line of a small text file, whatever an editor made of it (a byte-order mark, UTF-16, quotes)."""
    try:
        raw = Path(p).read_bytes()[:4096]
    except OSError:
        return None
    text = raw.decode("utf-16", "ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:64] else raw.decode("utf-8-sig", "ignore")
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if line:
            return line
    return None


def api():
    from huggingface_hub import HfApi

    t = token()
    if not t:
        print("publish: no Hugging Face token found on this machine; nothing is published. Looked for HF_TOKEN, then a file named "
              ".hugginface.key or .huggingface.key in the Linux home, in a Windows user's home (C:\\Users\\NAME) and next to go.cmd, "
              "hf_key.txt / huggingface_key.txt in Documents, then the Hugging Face CLI's own token file. Put the token on the first "
              "line of one of those (or the PATH of your token file in a file named HF_TOKEN_FILE next to go.cmd) and run: go.cmd publish")
        sys.exit(0)
    a = HfApi(token=t)
    try:
        who = a.whoami()["name"]
    except Exception as e:  # noqa: BLE001
        print(f"publish: the token here was not accepted by huggingface.co ({type(e).__name__}); nothing is published")
        sys.exit(0)
    return a, who


def published(logs) -> list:
    f = Path(logs) / "published.txt"
    return [l.split()[0] for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def note(logs, repo: str) -> None:
    f = Path(logs) / "published.txt"
    have = published(logs)
    if repo not in have:
        with open(f, "a") as o:
            o.write(f"{repo}  https://huggingface.co/{repo}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def collect(a, who: str, repo: str, what: str, logs) -> None:
    """Every repository of the project sits in one collection on the account's page."""
    try:
        c = a.create_collection(title="Qwencoin", namespace=who, exists_ok=True, description=(
            "Bitcoin whose hash function and proof of work are a language model's weights, and the chat models that carry its SHA-256."))
        a.add_collection_item(c.slug, item_id=repo, item_type="model", note=what[:480], exists_ok=True)
        f = Path(logs) / "collection.txt"
        f.write_text(f"https://huggingface.co/collections/{c.slug}\n")
        print(f"publish: {repo} is in the collection https://huggingface.co/collections/{c.slug}")
    except Exception as e:  # noqa: BLE001  (the repository is up either way)
        print(f"publish: the collection could not be updated ({type(e).__name__}: {str(e)[:200]})")


def logs_folder(logs, names) -> Path:
    """The named logs, scrubbed of this machine's folder names, in a folder of their own."""
    d = Path(tempfile.mkdtemp(prefix="qwlogs"))
    for n in names:
        for f in sorted(Path(logs).glob(n)) + sorted((ROOT / "logs").glob(n)):
            if f.is_file() and not (d / f.name).exists():
                (d / f.name).write_text(cards.scrub(f.read_text(errors="replace")))
    return d


def model(size: str, folder, logs) -> bool:
    folder = Path(folder)
    if not (ROOT / ".done" / f"merged-{size}").exists():
        print(f"publish: Qwen3.5-{size} has not passed its checks here; it is not published")
        return False
    if not os.environ.get("NO_TOOLS") and not (ROOT / ".done" / f"tools-{size}").exists():
        print(f"publish: Qwen3.5-{size} has not passed the check of its functions here; it is not published")
        return False
    a, who = api()
    repo = f"{who}/Qwen3.5-{size}-qwencoin"
    try:
        plain_probe(size, folder, logs)
    except Exception as e:  # noqa: BLE001  (the card is written from the log as it is)
        print(f"publish: the plain-chat probe was not appended ({type(e).__name__}: {str(e)[:200]})")
    (folder / "README.md").write_text(cards.model_card(size, folder, who, logs))
    gb = sum(f.stat().st_size for f in folder.glob("*.safetensors")) / 2**30
    print(f"publish: {folder} ({gb:.1f} GB) -> https://huggingface.co/{repo}")
    a.create_repo(repo, repo_type="model", exist_ok=True)
    t0 = time.time()
    if gb > 20:  # resumable, in several commits
        a.upload_large_folder(repo_id=repo, repo_type="model", folder_path=str(folder))
    else:
        a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), commit_message=f"Qwen3.5-{size} with the SHA-256 machine merged in")
    ld = logs_folder(logs, [f"merge-{size}.log", f"tools-{size}.log", "machine.txt", "backend.json"])
    a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ld), path_in_repo="logs", commit_message="the logs of this checkpoint's checks")
    shutil.rmtree(ld, ignore_errors=True)
    (ROOT / ".done" / f"published-{size}").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    note(logs, repo)
    collect(a, who, repo, f"Qwen3.5-{size} with the SHA-256 machine merged into its weights and a qwencoin node's functions declared in its chat template.", logs)
    print(f"publish: https://huggingface.co/{repo} is up ({time.time() - t0:.0f}s)")
    try:  # the project page again: it lists this checkpoint now, and carries the code as it is now (unchanged files are not sent twice)
        hub(logs)
    except Exception as e:  # noqa: BLE001
        print(f"publish: the project page was not refreshed ({type(e).__name__}: {str(e)[:300]})")
    return True


def plain_probe(size: str, folder, logs) -> None:
    """A merge log written before tests/check_merge.py's plain-chat probe was corrected (it rendered
    the prompt with the checkpoint's own template, which declares the node's functions, and compared
    at that offset) gets the corrected probe, run now and appended under its own command line."""
    import subprocess
    f = Path(logs) / f"merge-{size}.log"
    work = ROOT / "work" / size
    if not f.exists() or "with no functions declared" in f.read_text(errors="replace") or not (work / "base_greedy.pt").exists():
        return
    print(f"publish: Qwen3.5-{size}: the plain-chat probe of its checks, corrected, is run before its card is written")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tests" / "check_merge.py"), "plain", str(folder), str(work)],
                           capture_output=True, text=True, timeout=3600, cwd=ROOT)
        out = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and not l.startswith(("[transformers]", "Loading weights", "Loading checkpoint"))]
        if r.returncode:
            out.append(f"(exit {r.returncode})")
    except Exception as e:  # noqa: BLE001
        out = [f"(the probe did not run: {type(e).__name__}: {str(e)[:200]})"]
    with open(f, "a") as o:
        o.write(f"\n$ python tests/check_merge.py plain {Path(folder).relative_to(ROOT) if Path(folder).is_relative_to(ROOT) else folder} work/{size}\n"
                "(the line \"greedy answer to a chat prompt without the tool\" above came from a probe that rendered the prompt with this "
                "checkpoint's own template, which declares the node's functions, and compared at that offset: it could not match. "
                "The corrected probe, run again:)\n" + "\n".join(out) + "\n")
    print("\n".join("   " + l for l in out[-4:]))


def refresh_cards(a, who: str, logs) -> None:
    """The card of every checkpoint already published, written again from what is here now (the
    public node's address, the latest logs of its checks). Only README.md is sent."""
    for size, d in DIRS.items():
        folder = ROOT / d
        if not (ROOT / ".done" / f"published-{size}").exists() or not (folder / "hashformer.json").exists():
            continue
        repo = f"{who}/Qwen3.5-{size}-qwencoin"
        try:
            text = cards.model_card(size, folder, who, logs)
            (folder / "README.md").write_text(text)
            a.upload_file(repo_id=repo, repo_type="model", path_in_repo="README.md", path_or_fileobj=text.encode(),
                          commit_message="the card as things are now")
            print(f"publish: the card of https://huggingface.co/{repo} is current")
        except Exception as e:  # noqa: BLE001
            print(f"publish: the card of {repo} was not refreshed ({type(e).__name__}: {str(e)[:200]})")


def seeds(logs) -> bool:
    """The code, the card and the public nodes' addresses (the file SEEDS, which a copy of the
    repository dials) -> ACCOUNT/qwencoin, without the large files and the logs."""
    return hub(logs, full=False)


def hub(logs, full: bool = True) -> bool:
    a, who = api()
    repo = f"{who}/qwencoin"
    stage = Path(tempfile.mkdtemp(prefix="qwhub"))
    (stage / "README.md").write_text(cards.hub_card(who, logs, [r for r in published(logs) if r != repo]))
    for f in ("RESULTS.md", "MODELS", "SEEDS"):
        if (ROOT / f).exists() and (ROOT / f).read_text().strip():
            shutil.copy(ROOT / f, stage / f)
    (stage / "ORIGIN").write_text(who + "\n")  # the account the checkpoints are under: `pipeline.sh join` fetches from it
    if (ROOT / "tools" / "install.sh").exists():  # the one command of the cards: this repository, then `pipeline.sh join`
        # the GitHub repository whose workflow builds the node for the other platforms (its releases hold the packages)
        github = os.environ.get("QWEN_GITHUB") or ((ROOT / "GITHUB").read_text().strip() if (ROOT / "GITHUB").exists() else "staccDOTsol/qwencoin")
        (stage / "GITHUB").write_text(github + "\n")
        (stage / "install.sh").write_text((ROOT / "tools" / "install.sh").read_text().replace("@ACCOUNT@", who).replace("@GITHUB@", github))
    if (ROOT / "README.md").exists():
        shutil.copy(ROOT / "README.md", stage / "BUILD.md")  # the engineering notes; README.md is the card
    print(f"publish: the project -> https://huggingface.co/{repo}")
    a.create_repo(repo, repo_type="model", exist_ok=True)
    a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(stage), commit_message="qwencoin: the project page")
    shutil.rmtree(stage, ignore_errors=True)
    skip = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]
    for d in SOURCE:
        if (ROOT / d).is_dir():
            a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ROOT / d), path_in_repo=d, ignore_patterns=skip,
                            commit_message=f"qwencoin: {d}/")
    try:  # the launcher of the Windows build kit works from the kit's folder, not from a copy of this repository
        a.delete_folder("win", repo_id=repo, repo_type="model", commit_message="the kit's launcher is not part of the repository")
    except Exception:  # noqa: BLE001  (not there)
        pass
    if not ((ROOT / "SEEDS").exists() and (ROOT / "SEEDS").read_text().strip()):  # no public node now: none named there either
        try:
            a.delete_file("SEEDS", repo_id=repo, repo_type="model", commit_message="no public node at the moment")
        except Exception:  # noqa: BLE001  (there was none)
            pass
    refresh_cards(a, who, logs)
    if not full:
        print(f"publish: https://huggingface.co/{repo} has the code, the card and the public nodes as they are now")
        return True
    if (ROOT / "dist" / "qwork.bin").exists():
        a.upload_file(repo_id=repo, repo_type="model", path_or_fileobj=str(ROOT / "dist" / "qwork.bin"), path_in_repo="qwork.bin",
                      commit_message="the chain's work table (the MLP weights of Qwen3.5-0.8B at 16-bit fixed point)")
    if (ROOT / "dist" / "qwensha" / "model.safetensors").exists():
        a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ROOT / "dist" / "qwensha"), path_in_repo="qwensha",
                        commit_message="the SHA-256 machine alone, as a stock Qwen3_5ForCausalLM")
    ld = logs_folder(logs, ["qwork-node.log", "unit-qwork.log", "work.log", "tools.log", "untied.log", "bridge-ir.log", "merge-*.log",
                            "tools-*.log", "functional-qwork.log", "ord-busy.log", "ord-orphan.log", "untied-merge.log", "published-*.log", "machine.txt"])
    a.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ld), path_in_repo="logs", commit_message="what the checks printed")
    shutil.rmtree(ld, ignore_errors=True)
    note(logs, repo)
    collect(a, who, repo, "The project: the node and ord for the chain, the compiler, the work table, the stand-alone SHA-256 machine, every check and its log.", logs)
    print(f"publish: https://huggingface.co/{repo} is up")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "whoami":
        print("the token on this machine belongs to:", api()[1])
    elif cmd == "hub":
        hub(sys.argv[2])
    elif cmd in ("seeds", "cards"):
        seeds(sys.argv[2])
    elif cmd == "model":
        model(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "pending":
        logs = sys.argv[2]
        if not (ROOT / ".done" / "minds").exists() and not os.environ.get("NO_NET"):
            sys.exit("publish: not yet: the nodes' models have not all spoken on the graph (python tools/chain.py minds)")
        for size, d in DIRS.items():
            if (ROOT / ".done" / f"merged-{size}").exists() and not (ROOT / ".done" / f"published-{size}").exists():
                model(size, ROOT / d, logs)
        hub(logs)
    else:
        sys.exit(__doc__)
