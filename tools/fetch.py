"""Download a model's files into a local folder (nothing leaves this folder).

    python tools/fetch.py Qwen/Qwen3.5-4B base/Qwen3.5-4B

If a Hugging Face token is on this machine (the one tools/publish.py uses: HF_TOKEN, or
~/.hugginface.key / ~/.huggingface.key) the download is authenticated with it: higher rate
limits, faster. The token goes to huggingface.co only and is never printed. Without one the
download is anonymous, which is enough for these public models.
"""
import sys, time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from publish import token
    tok = token()
except Exception:  # noqa: BLE001  (anonymous then)
    tok = None
repo, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 8):  # a connection that drops is not the end of it: what has arrived is kept, the rest is asked for again
    try:
        p = snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], token=tok or None)
        break
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if attempt == 7 or name in ("RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError"):
            raise
        wait = min(15 * attempt, 60)
        print(f"   the download was interrupted ({name}: {" ".join(str(e).split())[:120]}); again in {wait}s, what has arrived is kept", flush=True)
        time.sleep(wait)
print("downloaded", repo, "->", p, "(authenticated)" if tok else "(anonymous)")
