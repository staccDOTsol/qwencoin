"""A public node of the chain on a rented RunPod pod: the meeting point that nodes behind
routers dial out to.

    python tools/runpod.py seed [LOGS]     rent one pod (or go on with the one this account has), start
                                           a node on it, point this machine's nodes at it
    python tools/runpod.py status [LOGS]   the pod, its price, its address, who is connected, what it printed last
    python tools/runpod.py logs [LOGS]     the last 200 lines its container printed
    python tools/runpod.py down [LOGS]     terminate the pod (billing stops; its copy of the chain goes with it)

The pod downloads the node from the project's Hugging Face repository (so that has to be
published first) and runs tools/seed.sh: one qwencoind with its ord and a miner, listening on
8733. What it mines is paid to node 1 of this machine: nothing of value lives on the pod, whose
disk is wiped whenever it restarts. RunPod maps the port to a public address, which is written
to the file SEEDS (here, next to go.cmd and in the project's repository, so that a copy of the
repository dials it) and given to the running nodes at once. The address changes when the pod
is reset: `status` notices and repoints everything.

The RunPod key is read on this machine and goes to api.runpod.io only: $RUNPOD_API_KEY, or
runpod_key.txt in Documents or the home folder (Linux home, or a Windows user's home as WSL
sees it), or next to go.cmd. It is never printed or logged. Renting costs money on that
account: the hourly price is printed before and after the pod is created, one pod is rented at
most (a second `seed` continues with the first), and a pod whose node does not come up is
terminated again unless SEED_KEEP=1.
"""
import base64, ipaddress, json, os, re, socket, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
API = os.environ.get("RUNPOD_API", "https://api.runpod.io/v2").rstrip("/")
SEEN = {}  # the last answer of each interface, written to LOGS/seed-debug.json (no key in it: answers only)
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
NAME = "qwencoin-seed"
IMAGE = "python:3.12-slim"
PORT = 8733
STATUS = 8080  # the pod serves a small status page here, reached through RunPod's HTTP proxy (no API permission needed)
PROXY = os.environ.get("RUNPOD_PROXY", "https://{id}-{port}.proxy.runpod.net")
VERSION = 3  # of the pod's start script: a pod started by an earlier one is replaced
BOOK = ROOT / "seed-pod.json"  # the pod this folder rented (not in run/, which is set aside when the chain changes)
# What is asked for, in this order, until one comes up. A small CPU pod would do for a node, but the ones rented
# on 2026-11-13 sat in "Initializing" for minutes without a line of log, so GPU pods come first: they start in
# under a minute and always have a public address. SEED_KIND=cpu|gpu|any chooses; SEED_MAX_USD_HOUR caps the price.
GPUS = ["NVIDIA RTX A2000", "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A4500",
        "NVIDIA RTX A5000", "Tesla T4", "NVIDIA L4", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4080", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090"]
CPUS = [("cpu5c", 4), ("cpu3c", 4), ("cpu5g", 2), ("cpu3g", 2)]
MAX_USD_HOUR = float(os.environ.get("SEED_MAX_USD_HOUR", "0.60"))
CLIENTS = ["qwencoin-seed/1", "curl/8.5.0"]  # how this script names itself; the second if something in front of the API refuses the first
_client = [0]

# What the pod runs. Everything it prints is its log (`logs` reads it). It never exits by itself: a
# failure stays readable until this script, on the owner's machine, has read it and terminated the pod.
POD = r'''mkdir -p /status
exec > >(tee -a /status/log) 2>&1
stay() { echo "SEED FAILED: $*"; sleep infinity; }
echo "seed: $(date -u +%FT%TZ), $(nproc) cpus, $(awk '/MemTotal/ {printf "%.1f", $2 / 1048576}' /proc/meminfo) GB, $(python3 -V)"
# /status/address: the public address RunPod gave this pod for the node's port, from the pod's environment or,
# failing that, from RunPod's API with the pod's own key (which stays in here). Served with the log on port @STATUS@.
python3 - <<'PY' &
import json, os, time, urllib.request
port = "@PORT@"
def from_api():
    k, pid = os.environ.get("RUNPOD_API_KEY"), os.environ.get("RUNPOD_POD_ID")
    if not (k and pid):
        return None
    q = {"query": 'query { pod(input: {podId: "%s"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }' % pid}
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps(q).encode(), headers={
        "Authorization": "Bearer " + k, "Content-Type": "application/json", "User-Agent": "qwencoin-seed/1"})
    r = json.load(urllib.request.urlopen(req, timeout=20))
    for p in (((r.get("data") or {}).get("pod") or {}).get("runtime") or {}).get("ports") or []:
        if str(p.get("privatePort")) == port and p.get("isIpPublic") and p.get("publicPort"):
            return "%s:%s" % (p["ip"], p["publicPort"])
    return None
while True:
    ip, p = os.environ.get("RUNPOD_PUBLIC_IP", ""), os.environ.get("RUNPOD_TCP_PORT_" + port, "")
    a, how = ("%s:%s" % (ip, p), "environment") if ip and p else (None, "environment")
    if not a:
        try:
            a, how = from_api(), "api"
        except Exception as e:
            how = "api: " + type(e).__name__
    names = sorted(k for k in os.environ if k.startswith("RUNPOD_"))  # names only
    open("/status/address.tmp", "w").write(json.dumps({"address": a, "from": how, "at": int(time.time()), "public_ip": ip or None, "runpod_variables": names}))
    os.replace("/status/address.tmp", "/status/address")
    time.sleep(15)
PY
(cd /status && exec python3 -m http.server @STATUS@ >/dev/null 2>&1) &
for k in 1 2 3 4 5 6 7 8 9 10 11 12; do [ -s /status/address ] && break; sleep 1; done
ADDR=$(python3 -c "import json; print(json.load(open('/status/address')).get('address') or '')" 2>/dev/null || true)
echo "seed: public address ${ADDR:-not given to this pod}"
(apt-get -qq update && apt-get -qq install -y --no-install-recommends procps) >/dev/null 2>&1 || echo "seed: procps not installed (only stopping by hand needs it)"
pip install -q --no-input --root-user-action=ignore numpy huggingface_hub || stay "pip could not install numpy and huggingface_hub"
python3 - <<'PY' || stay "the project could not be downloaded from @HUB@/@REPO@"
import time
from huggingface_hub import snapshot_download
for k in range(6):
    try:
        snapshot_download("@REPO@", local_dir="/qwencoin", ignore_patterns=["qwensha/*", "logs/*"])
        break
    except Exception as e:
        print("seed: download:", type(e).__name__, str(e)[:300], flush=True)
        if k == 5:
            raise
        time.sleep(20)
PY
cd /qwencoin || stay "no /qwencoin"
echo "seed: the project is here ($(du -sh . 2>/dev/null | cut -f1))"
rm -f SEEDS
mkdir -p run
[ -n "@MINE_TO@" ] && [ ! -f run/addrs.json ] && echo '{"1": "@MINE_TO@"}' > run/addrs.json
[ -n "$ADDR" ] && export QWEN_EXTERNAL="$ADDR"
echo "seed: starting the node"
QWEN_MINE_WAIT=1800 AGENTS=0 bash tools/seed.sh || stay "the node did not start"
echo "seed: UP"
while true; do python3 -u tools/chain.py watch; echo "seed: watch ended; again in 10s"; sleep 10; done
'''


def key() -> str | None:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if k:
        return k
    home, names = Path.home(), ("runpod_key.txt", ".runpod_key", "runpod.key")
    places = [home / "Documents" / n for n in names] + [home / n for n in names] + [ROOT / n for n in names]
    win = Path("/mnt/c/Users")
    if win.is_dir():
        for n in names:
            places += sorted(win.glob(f"*/Documents/{n}")) + sorted(win.glob(f"*/{n}"))
    if os.environ.get("KIT_SRC"):
        places += [Path(os.environ["KIT_SRC"]).parent / n for n in names]
    from publish import first_line
    for p in places:
        k = first_line(p)
        if k:
            return k
    return None


def need_key() -> str:
    k = key()
    if not k:
        sys.exit("runpod: no key found on this machine (RUNPOD_API_KEY, or runpod_key.txt in Documents, in the home folder or next to go.cmd)")
    return k


def call(method: str, path: str, body=None, k: str | None = None, timeout: int = 60):
    """(status, body). Status 0: no answer, and for a POST the request may or may not have arrived."""
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(5):
        headers = {"Authorization": f"Bearer {k}", "Accept": "application/json", "User-Agent": CLIENTS[_client[0]]}
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            url = path if path.startswith("http") else API + path
            with urllib.request.urlopen(urllib.request.Request(url, method=method, data=data, headers=headers), timeout=timeout) as r:
                raw = r.read()
                try:
                    return r.status, (json.loads(raw) if raw.strip() else {})
                except ValueError:
                    return r.status, {"detail": raw[:300].decode("utf-8", "replace")}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                r = json.loads(raw)
            except ValueError:
                r = None
            if not isinstance(r, dict):
                r = {"detail": raw[:300]}
            if e.code == 403 and "title" not in r and "status" not in r and _client[0] + 1 < len(CLIENTS):
                _client[0] += 1  # not the API's own answer: something in front of it turned this client away
                continue
            if e.code == 429 and attempt < 4:  # nothing was done: wait as told, again
                try:
                    wait = int(e.headers.get("Retry-After") or 5)
                except ValueError:
                    wait = 5
                time.sleep(min(max(wait, 1), 60))
                continue
            return e.code, r
        except (urllib.error.URLError, OSError) as e:
            return 0, {"detail": f"{type(e).__name__}: {str(getattr(e, 'reason', e))[:200]}"}
    return 0, {"detail": "no answer"}


def problem(r) -> str:
    return "; ".join(str(r.get(x)) for x in ("title", "detail", "message", "errors") if r.get(x))[:500] or json.dumps(r)[:500]


def repo_of(logs) -> str | None:
    """ACCOUNT/qwencoin, if it is published."""
    f = Path(logs) / "published.txt" if logs else None
    if f and f.exists():
        for line in f.read_text().splitlines():
            if line.split() and line.split()[0].endswith("/qwencoin"):
                return line.split()[0]
    try:
        import publish
        t = publish.token()
        if t:
            from huggingface_hub import HfApi
            return HfApi(token=t).whoami()["name"] + "/qwencoin"
    except Exception:  # noqa: BLE001
        pass
    return None


def in_hub(repo: str, path: str) -> bool:
    try:
        req = urllib.request.Request(f"{HUB}/{repo}/resolve/main/{path}", method="HEAD", headers={"User-Agent": CLIENTS[0]})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def public(ip) -> bool:
    try:
        return ipaddress.ip_address(str(ip)).is_global
    except ValueError:
        return False


def addr_of(pod) -> str | None:
    """ip:port under which the pod's PORT is reachable, from a pod object of any of RunPod's interfaces."""
    if not isinstance(pod, dict):
        return None
    for p in ((pod.get("runtime") or {}).get("ports") or []):
        try:
            inner, outer = p.get("private", p.get("privatePort")), p.get("public", p.get("publicPort"))
            if int(inner or 0) == PORT and outer and public(p.get("ip")):
                return f"{p['ip']}:{int(outer)}"
        except (TypeError, ValueError, AttributeError):
            continue
    try:
        maps = pod.get("portMappings") or {}
        outer = maps.get(str(PORT)) or maps.get(PORT)
        if outer and public(pod.get("publicIp")):
            return f"{pod['publicIp']}:{int(outer)}"
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def page(pid: str, path: str, timeout: float = 10):
    """A file of the pod's status page, through RunPod's HTTP proxy. None: the page does not answer."""
    try:
        req = urllib.request.Request(PROXY.format(id=pid, port=STATUS) + "/" + path, headers={"User-Agent": CLIENTS[0]})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(400000).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def look(k, pid: str) -> dict:
    """status: RunPod's word for the pod. addr: its public address for PORT, or None. doing: the last thing the
    pod's start script said. page: what the pod's own status page says about its address (None: no page yet).
    The address is taken from RunPod's API if it reports one, else from the pod itself."""
    st, r = call("GET", f"/pods/{pid}", k=k)
    SEEN["v2 pod"] = [st, r]
    x = {"status": r.get("status") if st == 200 else ("gone" if st == 404 else None), "addr": addr_of(r) if st == 200 else None,
         "doing": None, "page": None}
    t = page(pid, "address")
    SEEN["status page"] = t
    if t:
        try:
            x["page"] = json.loads(t)
        except ValueError:
            x["page"] = {"address": None, "from": "unreadable"}
        if not isinstance(x["page"], dict):
            x["page"] = {"address": None, "from": "unreadable"}
        a = x["page"].get("address")
        if not x["addr"] and a and public(str(a).rsplit(":", 1)[0]):
            x["addr"] = a
        said = [l for l in (page(pid, "log") or "").splitlines() if l.startswith(("seed:", "SEED FAILED"))]
        x["doing"] = said[-1][:200] if said else None
    return x


def debug(logs) -> None:
    if logs:
        try:
            (Path(logs) / "seed-debug.json").write_text(json.dumps(SEEN, indent=1, default=str)[:400000])
        except OSError:
            pass


def save(b: dict) -> dict:
    tmp = BOOK.with_suffix(".tmp")
    tmp.write_text(json.dumps(b))
    os.replace(tmp, BOOK)
    return b


def named(k):
    """This account's pods with the seed's name, or None if the list cannot be read."""
    st, r = call("GET", "/pods?limit=1000", k=k)
    if st != 200 or not isinstance(r, dict):
        return None
    return [p for p in (r.get("pods") or []) if p.get("name") == NAME and p.get("status") != "TERMINATED"]


def book(k):
    """The seed pod this account has: the one recorded here, else one with the seed's name. None: there is none."""
    b = json.loads(BOOK.read_text()) if BOOK.exists() else None
    if b:
        st, r = call("GET", f"/pods/{b['id']}", k=k)
        if st == 404 or (st == 200 and r.get("status") == "TERMINATED"):
            BOOK.rename(BOOK.with_name(f"seed-pod-gone-{time.strftime('%Y%m%d-%H%M%S')}.json"))
            print(f"runpod: pod {b['id']}, recorded here, no longer exists on the account")
            unpoint(b.get("address"))
            b = None
        elif st != 200:
            sys.exit(f"runpod: could not ask about pod {b['id']}: HTTP {st}: {problem(r)}")
    if not b:
        ps = named(k)
        if ps is None:
            sys.exit("runpod: the account's pods could not be listed (is the key valid?); nothing was rented")
        if ps:
            p = ps[0]
            b = save({"id": p["id"], "cost": p.get("cost"), "cpu": p.get("cpu"), "created": p.get("createdAt"), "found": True})
    return b


def script(repo: str, mine_to: str) -> str:
    text = (POD.replace("@REPO@", repo).replace("@HUB@", HUB).replace("@MINE_TO@", mine_to or "").replace("@PORT@", str(PORT))
            .replace("@STATUS@", str(STATUS)))
    # one line without quotes or line breaks, whatever happens to the command on its way into the container
    return f"echo {base64.b64encode(text.encode()).decode()} | base64 -d > /seed-start.sh; exec bash /seed-start.sh"


def asks(repo: str, mine_to: str) -> list:
    """(what, request body) for every kind of pod that may be asked for, in order."""
    base = {"name": NAME, "image": IMAGE, "cloud": "SECURE", "ports": [f"{PORT}/tcp", f"{STATUS}/http"],
            "entrypoint": ["bash", "-c"], "cmd": [script(repo, mine_to)]}
    kind = os.environ.get("SEED_KIND", "gpu")
    out = []
    if kind in ("gpu", "any"):
        out += [(g, {**base, "gpu": {"id": g, "count": 1}, "disk": 30}) for g in GPUS]
    if kind in ("cpu", "any"):
        out += [(f"{f} with {n} vCPU", {**base, "cpu": {"id": f, "vcpuCount": n}, "disk": 10}) for f, n in CPUS]
    return out


def create(k, what: str, body: dict, repo: str):
    """One request for a pod. (its record, None), or (None, why not); the account is checked for a pod that
    exists all the same when RunPod's answer was not clear."""
    st, r = call("POST", "/pods", body, k)
    if st in (200, 201) and r.get("id"):
        return save({"id": r["id"], "cost": r.get("cost"), "what": what, "repo": repo, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "v": VERSION}), None
    why = f"HTTP {st}: {problem(r)}"
    if st in (200, 201) or st == 0 or st >= 500:  # no clear answer: a pod may exist all the same
        time.sleep(5)
        ps = named(k)
        if ps:
            p = ps[0]
            return save({"id": p["id"], "cost": p.get("cost"), "what": what, "repo": repo, "created": p.get("createdAt"), "v": VERSION}), None
        if ps is None:
            sys.exit(f"runpod: no clear answer from RunPod ({why}) and its list of pods cannot be read: look at the account's "
                     f"pods for one named {NAME} before asking again")
    if st in (401, 402):  # the key, the balance: asking for something else would not change it
        sys.exit(f"runpod: {why}. Nothing is rented.")
    return None, why


def wait_address(k, b: dict, logs, patience: int):
    """The pod's public address once its container is up, or (None, why) after `patience` seconds."""
    shown, t0, up_since = None, time.time(), None
    while time.time() - t0 < patience:
        x = look(k, b["id"])
        line = f"runpod: the pod is {x['status']}" + ("; its container has not answered yet" if x["page"] is None else f"; {x['doing'] or 'its container is up'}")
        if line != shown:
            shown = line
            print(line, flush=True)
        debug(logs)
        if x["addr"]:
            return x["addr"], None
        if x["status"] in ("EXITED", "ERROR", "TERMINATED", "gone"):
            return None, f"it went to {x['status']}"
        if x["page"] is not None:  # the pod answers and has no address to give
            up_since = up_since or time.time()
            if time.time() - up_since > 75:
                return None, ("it is up, and RunPod gave it no public address for a TCP port (it looked in its environment and asked "
                              f"RunPod's API: {json.dumps(x['page'])[:400]})")
        time.sleep(6)
    return None, f"its container did not start within {patience // 60} minutes (RunPod still shows it initializing)"


def nodes():
    """The numbers of this machine's running nodes."""
    import net
    out = []
    for d in sorted(net.RUN.glob("n[0-9]*")):
        i = int(d.name[1:])
        if net.cli(i, "getblockcount", check=False, timeout=20) != "":
            out.append(i)
    return out


def seed_files() -> list:
    return [ROOT / "SEEDS"] + ([Path(os.environ["KIT_SRC"]).parent / "SEEDS"] if os.environ.get("KIT_SRC") else [])


def point(addr: str, old: str | None = None) -> int:
    """SEEDS (here and next to go.cmd) names the address, and the running nodes dial it."""
    import net
    for f in seed_files():
        have = [x for x in (f.read_text().split() if f.exists() else []) if x != old]
        if addr not in have:
            have.append(addr)
        try:
            f.write_text("\n".join(have) + "\n")
        except OSError as e:
            print(f"runpod: could not write {f} ({e})")
    ns = nodes()
    for i in ns:
        if old and old != addr:
            net.cli(i, "addnode", old, "remove", check=False)
        net.cli(i, "addnode", addr, "add", check=False)
    return len(ns)


def unpoint(addr: str | None) -> None:
    import net
    if not addr:
        return
    for f in seed_files():
        if f.exists():
            try:
                f.write_text("".join(x + "\n" for x in f.read_text().split() if x != addr))
            except OSError:
                pass
    for i in nodes():
        net.cli(i, "addnode", addr, "remove", check=False)
        net.cli(i, "disconnectnode", addr, check=False)


def peers(addr: str) -> list:
    """(node, the seed's height as that node knows it) for this machine's nodes connected to the seed."""
    import net
    out = []
    for i in nodes():
        info = net.cli(i, "getpeerinfo", check=False, timeout=30)
        for p in info if isinstance(info, list) else []:
            if p.get("addr") == addr:
                out.append((i, max(p.get("synced_blocks", -1), p.get("startingheight", -1))))
    return out


def listening(addr: str, timeout: float = 6) -> bool:
    host, port = addr.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def pod_log(k, pid: str, tail: int = 100, seconds: float = 6) -> list:
    """The last lines the pod's container printed. The API answers with a stream; it is read for a few seconds."""
    t = page(pid, "log")
    if t is not None:
        return [l.rstrip() for l in t.splitlines() if l.strip()][-tail:]
    req = urllib.request.Request(f"{API}/pods/{pid}/logs?source=container&tail={tail}",
                                 headers={"Authorization": f"Bearer {k}", "Accept": "text/event-stream", "User-Agent": CLIENTS[_client[0]]})
    out, raw_seen, t0 = [], [], time.time()
    try:
        with urllib.request.urlopen(req, timeout=seconds) as r:
            raw_seen.append(f"HTTP {r.status} {r.headers.get('Content-Type')}")
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if len(raw_seen) < 40 and line:
                    raw_seen.append(line[:300])
                if line.startswith("data:"):
                    try:
                        d = json.loads(line[5:])
                        out.append(str(d.get("line", d.get("message", "")) if isinstance(d, dict) else d).rstrip())
                    except ValueError:
                        out.append(line[5:].strip())
                if time.time() - t0 > seconds:
                    break
    except urllib.error.HTTPError as e:
        raw_seen.append(f"HTTP {e.code} {e.read()[:300].decode('utf-8', 'replace')}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raw_seen.append(f"({type(e).__name__})")
    SEEN["log stream"] = raw_seen
    return out


def terminate(k, b: dict) -> bool:
    st, r = call("DELETE", f"/pods/{b['id']}", k=k)
    if st not in (200, 202, 204, 404):
        st, r = call("POST", f"/pods/{b['id']}/action", {"action": "terminate"}, k)
    if st not in (200, 202, 204, 404):
        print(f"runpod: could not terminate pod {b['id']}: HTTP {st}: {problem(r)}. It is still rented: terminate it on runpod.io")
        return False
    if BOOK.exists():
        BOOK.rename(BOOK.with_name(f"seed-pod-terminated-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    left = named(k)
    print(f"runpod: pod {b['id']} is terminated" + ("; the account has no seed pod now, nothing is billed for one" if left == []
                                                    else "" if left is None else f"; the account still lists {[p['id'] for p in left]}"))
    return True


def give_up(k, b: dict, logs, why: str, addr: str | None = None):
    lines = pod_log(k, b["id"], tail=200, seconds=8)
    debug(logs)
    if logs and lines:
        (Path(logs) / "seed-pod.log").write_text("\n".join(lines) + "\n")
    for l in lines[-25:]:
        print("   pod |", l[:300])
    print(f"runpod: {why}")
    if os.environ.get("SEED_KEEP"):
        print(f"runpod: SEED_KEEP is set: pod {b['id']} stays, and it is billed until `down`")
    else:
        terminate(k, b)
        unpoint(addr)
    sys.exit(1)


def share(logs) -> None:
    """The code, the file SEEDS and the card that names the address, into the project's repository."""
    try:
        import publish
        publish.seeds(logs)
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"runpod: the project's page was not updated ({type(e).__name__}: {str(e)[:200]}); `publish` does it later")


def seed(logs=None):
    import net
    k = need_key()
    b = book(k)
    if b and b.get("v") != VERSION:
        print(f"runpod: pod {b['id']} was started by an earlier version of this script and cannot say where it is: it is replaced")
        if not terminate(k, b):
            sys.exit(1)
        unpoint(b.get("address"))
        b = None
    addr = None
    if b:
        print(f"runpod: this account already has the seed pod {b['id']}: going on with it (no second one is rented)")
    else:
        repo = os.environ.get("QWEN_REPO") or repo_of(logs)
        if not repo or not in_hub(repo, "tools/seed.sh") or not in_hub(repo, "qwork.bin"):
            sys.exit(f"runpod: the pod downloads the node from the project's Hugging Face repository, and "
                     f"{repo or 'ACCOUNT/qwencoin'} is not there (or not complete) yet: publish first. Nothing was rented.")
        mine_to = ""
        try:
            mine_to = json.loads((net.RUN / "addrs.json").read_text()).get("1", "")
        except (OSError, ValueError):
            pass
        print("runpod: first the project's repository gets the code as it is in this folder (the pod downloads it from there)", flush=True)
        share(logs)
        for what, body in asks(repo, mine_to):
            print(f"runpod: asking for a pod with {what}", flush=True)
            b, why = create(k, what, body, repo)
            if not b:
                print(f"runpod:   {why}", flush=True)
                continue
            cost = b.get("cost")
            if isinstance(cost, (int, float)) and cost > MAX_USD_HOUR:
                print(f"runpod:   ${cost:.2f} an hour is above the limit of ${MAX_USD_HOUR:.2f} (SEED_MAX_USD_HOUR): not kept", flush=True)
                terminate(k, b)
                b = None
                continue
            print(f"runpod: pod {b['id']} is rented ({what}" + (f", ${cost:.3f} an hour, about ${cost * 730:.0f} a month" if isinstance(cost, (int, float)) else "")
                  + f"); `seed down` terminates it. What it mines is paid to {mine_to or 'its own wallet'}", flush=True)
            print(f"runpod: its own status page: {PROXY.format(id=b['id'], port=STATUS)}/log", flush=True)
            addr, why = wait_address(k, b, logs, 240)
            if addr:
                break
            print(f"runpod: that pod is given up: {why}. It is terminated and the next kind is asked for.", flush=True)
            terminate(k, b)
            b = None
        if not b:
            sys.exit("runpod: no pod came up. Nothing is rented.")

    if not addr:  # a pod from an earlier run of this script: where is it now?
        print(f"runpod: its own status page: {PROXY.format(id=b['id'], port=STATUS)}/log", flush=True)
        addr, why = wait_address(k, b, logs, 900)
        if not addr:
            give_up(k, b, logs, f"the pod is given up: {why}")
    old = b.get("address")
    b["address"] = addr
    save(b)
    n = point(addr, old)
    print(f"runpod: its public address is {addr}: written to SEEDS, and {n} running node(s) here told to dial it", flush=True)

    # ---- its node: installed, started, reachable, on this chain
    shown, up, t0 = 0, [], time.time()
    while time.time() - t0 < 1800:
        lines = pod_log(k, b["id"], tail=60, seconds=4)
        fresh = [l for l in lines if l.startswith(("seed:", "SEED FAILED")) or "nodes up" in l or "wallet is ready" in l]
        for l in fresh[shown:]:
            print("   pod |", l[:300], flush=True)
        shown = max(shown, len(fresh))
        if any(l.startswith("SEED FAILED") for l in lines):
            give_up(k, b, logs, "the pod could not start its node (its log is above" + (" and in seed-pod.log" if logs else "") + ")", addr)
        if listening(addr):
            for i in nodes():
                net.cli(i, "addnode", addr, "onetry", check=False)
            time.sleep(3)
            up = peers(addr)
            if up or not nodes():
                break
        st, r = call("GET", f"/pods/{b['id']}", k=k)
        if st == 404 or (st == 200 and r.get("status") in ("EXITED", "ERROR", "TERMINATED")):
            give_up(k, b, logs, f"the pod stopped ({r.get('status', 'gone')})", addr)
        time.sleep(12)
    else:
        give_up(k, b, logs, "the pod's node did not become reachable within 30 minutes", addr)
    mine = net.cli(nodes()[0], "getblockcount", check=False) if nodes() else None
    if up:
        print(f"runpod: the seed's node is up and node(s) {[i for i, h in up]} here are connected to it; it is at block {max(h for i, h in up)}"
              f" of {mine} and syncs the rest from them", flush=True)
    else:
        print("runpod: the seed's node accepts connections; none of this machine's nodes is running, so the chain was not compared", flush=True)
    if logs:
        (Path(logs) / "seed.txt").write_text(f"{addr}  pod {b['id']}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    debug(logs)
    share(logs)
    print(f"runpod: done. {addr} is this chain's public node: a file SEEDS with that line (it is in the project's repository) is all "
          f"another machine needs. `seed status` shows it, `seed down` ends the rental.", flush=True)


def status(logs=None):
    import net
    k = need_key()
    b = book(k)
    if not b:
        print("runpod: this account has no seed pod; nothing is rented")
        return
    st, r = call("GET", f"/pods/{b['id']}", k=k)
    if st != 200:
        sys.exit(f"runpod: pod {b['id']}: HTTP {st}: {problem(r)}")
    cost, up_s = r.get("cost"), (r.get("runtime") or {}).get("uptime")
    kind = (r.get("gpu") or {}).get("id") or " ".join(str(x) for x in ((r.get("cpu") or {}).get("id"), (r.get("cpu") or {}).get("vcpuCount"), "vCPU") if x)
    addr = look(k, b["id"])["addr"]
    debug(logs)
    print(f"runpod: pod {b['id']}: {r.get('status')}, {kind}"
          + (f", ${cost:.3f} an hour (about ${cost * 730:.0f} a month)" if isinstance(cost, (int, float)) else "")
          + (f", up {up_s // 3600}h{up_s % 3600 // 60:02d}m" if isinstance(up_s, int) else "") + f", address {addr or 'none at the moment'}")
    if addr and addr != b.get("address"):
        old = b.get("address")
        b["address"] = addr
        save(b)
        n = point(addr, old)
        print(f"runpod: its address changed ({old} -> {addr}): SEEDS rewritten, {n} running node(s) told")
        share(logs)
    if addr:
        up = peers(addr)
        mine = net.cli(nodes()[0], "getblockcount", check=False) if nodes() else None
        print(f"runpod: " + (f"node(s) {[i for i, h in up]} here are connected to it; it is at block {max(h for i, h in up)} of {mine}" if up
                             else f"it {'accepts connections' if listening(addr) else 'does NOT accept connections'}; no node here is connected to it"))
    for l in pod_log(k, b["id"], tail=12, seconds=4)[-12:]:
        print("   pod |", l[:300])


def logs_(logs=None):
    k = need_key()
    b = book(k)
    if not b:
        print("runpod: this account has no seed pod")
        return
    for l in pod_log(k, b["id"], tail=200, seconds=8):
        print(l[:400])


def down(logs=None):
    k = need_key()
    b = book(k)
    if not b:
        print("runpod: this account has no seed pod; nothing is rented")
        return
    if terminate(k, b):
        unpoint(b.get("address"))
        if b.get("address"):
            print(f"runpod: {b['address']} is out of SEEDS and this machine's nodes no longer dial it")
            share(logs)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    run = {"seed": seed, "status": status, "logs": logs_, "down": down}.get(cmd)
    if not run:
        sys.exit(__doc__)
    run(arg)
