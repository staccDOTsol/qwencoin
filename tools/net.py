"""Run a local qwencoin network: N nodes, each a qwencoind (SHA-256 in the weights) with its own ord.

    python3 tools/net.py up 3      start nodes n1..n3 (n1 listens, the others connect to it)
    python3 tools/net.py down
    python3 tools/net.py cli 2 getblockcount

Over the internet the same commands apply with -addnode=<host>:8733 instead of localhost.
"""
import fcntl, json, os, signal, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(os.environ.get("QWEN_RUN") or ROOT / "run")  # the chain, the wallets, the logs
BIN = Path(os.environ.get("QWEN_BIN") or ROOT / "bin")  # the node programs (launchers that run libexec/ on lib/)
# the chain's work table: the MLP weights of Qwen3.5-0.8B, which every proof of work is a pass through
QWORK = Path(os.environ.get("QWEN_QWORK") or ROOT / "dist" / "qwork.bin")


BASE = int(os.environ.get("QWEN_PORT", "8733"))  # node 1 listens on the chain's own port
ORD_BASE = int(os.environ.get("QWEN_ORD_PORT", "18080"))


def port(i): return BASE + 10 * (i - 1)
def rpcport(i): return BASE - 1 + 10 * (i - 1)
def ordport(i): return ORD_BASE + i
def datadir(i): return RUN / f"n{i}"


def conf(i, n):
    # QWEN_BIND=0.0.0.0 opens n1's p2p port to other machines; QWEN_SEEDS=host:port,.. are peers elsewhere
    bind = os.environ.get("QWEN_BIND", "127.0.0.1") if i == 1 else "127.0.0.1"
    lines = ["server=1", "txindex=1", "fallbackfee=0.0001", "keypool=5", "dbcache=64", "listen=1",
             f"port={port(i)}", f"rpcport={rpcport(i)}", f"bind={bind}", "rpcbind=127.0.0.1",
             "rpcallowip=127.0.0.1", "dnsseed=0", "natpmp=0", "listenonion=0", "par=1",
             "maxtxfee=1", "datacarriersize=100000", "minrelaytxfee=0.00001", f"qwork={QWORK}"]
    # one link per pair of this machine's nodes: a node dials the ones before it (each dialing every other gave a
    # pair two connections, and a node among four others showed 7 peers)
    lines += [f"addnode=127.0.0.1:{port(j)}" for j in range(1, i)]
    lines += [f"addnode={s}" for s in seeds()]
    if i == 1 and os.environ.get("QWEN_EXTERNAL"):  # host:port under which others reach node 1 (a mapped port, say)
        lines.append(f"externalip={os.environ['QWEN_EXTERNAL']}")
    return "\n".join(lines) + "\n"


def seeds() -> list:
    """Nodes elsewhere to dial: QWEN_SEEDS=host:port,.. or the file SEEDS in this folder (one per line or comma
    separated). QWEN_SEEDS=none: nobody, whatever the file says: a private network (the tests') stays private."""
    text = os.environ.get("QWEN_SEEDS") or ((ROOT / "SEEDS").read_text() if (ROOT / "SEEDS").exists() else "")
    if text.strip().lower() == "none":
        return []
    return [s for s in text.replace(",", " ").split() if s and not s.startswith("#")]


def cli(i, *args, wallet=None, check=True, timeout=900):
    cmd = [str(BIN / "qwencoin-cli"), f"-datadir={datadir(i)}", f"-rpcclienttimeout={timeout}"]
    if wallet:
        cmd.append(f"-rpcwallet={wallet}")
    try:
        r = subprocess.run(cmd + [str(a) for a in args], capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"n{i} {args}: no answer in {timeout}s")
        return ""
    if r.returncode and check:
        raise RuntimeError(f"n{i} {args}: {r.stderr.strip()}")
    out = r.stdout.strip()
    try:
        return json.loads(out)
    except ValueError:
        return out


def refresh_seeds(timeout: int = 15) -> list:
    """An installed copy of the project asks the project for its current list of public nodes and keeps it in the file
    SEEDS: a rented public node gets a new address when it is replaced, and a node that only knew the old one would be
    alone. Nothing is asked where the list is this folder's own business: QWEN_SEEDS is set, or there is no file ORIGIN
    (the folder the project is published from has none: its list is the one that gets published). Returns the seeds."""
    if os.environ.get("QWEN_SEEDS") or not (ROOT / "ORIGIN").exists():
        return seeds()
    try:
        import re, urllib.request
        origin = (ROOT / "ORIGIN").read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", origin):
            return seeds()
        req = urllib.request.Request(f"https://huggingface.co/{origin}/qwencoin/resolve/main/SEEDS", headers={"User-Agent": "qwencoin-node/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read(4000).decode("utf-8", "replace")
        # host:port and nothing else: these lines go into the node's configuration
        new = [x for x in text.replace(",", " ").split() if re.fullmatch(r"(?:[A-Za-z0-9-]+\.)+[A-Za-z0-9-]+:[0-9]{2,5}", x)][:8]
        if new and new != seeds():
            (ROOT / "SEEDS").write_text("\n".join(new) + "\n")
    except Exception:  # noqa: BLE001  (no way out, or the project has no public node now: the list stays as it is)
        pass
    return seeds()


def up(n):
    if not QWORK.exists():
        raise RuntimeError(f"the chain's work table is not at {QWORK}: build it with "
                           f"`python -m hashformer.work build base/Qwen3.5-0.8B {QWORK}` (the pipeline does)")
    for i in range(1, n + 1):
        d = datadir(i)
        d.mkdir(parents=True, exist_ok=True)
        (d / "qwencoin.conf").write_text(conf(i, n))
        if cli(i, "getblockcount", check=False, timeout=20) == "":
            # -daemonwait would hold this call until the node is initialised; plain -daemon returns at once,
            # and its output goes to a file (a pipe here would stay open in the daemon on some systems)
            with open(d / "start.log", "a") as out:
                subprocess.run([str(BIN / "qwencoind"), f"-datadir={d}", "-daemon"], check=True,
                               stdout=out, stderr=out, stdin=subprocess.DEVNULL, timeout=120)
    for i in range(1, n + 1):
        for _ in range(180):
            if cli(i, "getblockcount", check=False, timeout=20) != "":
                break
            time.sleep(1)
        else:
            log = datadir(i) / "debug.log"
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-12:]) if log.exists() else "(no debug.log)"
            raise RuntimeError(f"n{i} did not start; the end of its debug.log:\n{tail}")


def settle(n=3, timeout=120):
    """Wait until block relay has given all n nodes the same tip."""
    for _ in range(timeout * 5):
        if len({cli(i, "getbestblockhash") for i in range(1, n + 1)}) == 1:
            return
        time.sleep(0.2)
    raise RuntimeError("tips did not converge")


def down():
    for d in sorted(RUN.glob("n[0-9]*")):
        i = int(d.name[1:])
        cli(i, "stop", check=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "up":
        up(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "down":
        down()
    elif a[0] == "cli":
        print(json.dumps(cli(int(a[1]), *a[2:]), indent=1))


# ---- ord: one indexer + wallet per node --------------------------------------------------
def ord_base(i):
    return [str(BIN / "ord"), "--bitcoin-rpc-url", f"http://127.0.0.1:{rpcport(i)}", "--cookie-file",
            str(datadir(i) / ".cookie"), "--data-dir", str(RUN / f"ord{i}"), "--index-sats"]


STARTED = "== ord server started"  # written to the server's log by ord_up, before the server writes anything


def ord_up(i):
    (RUN / f"ord{i}").mkdir(parents=True, exist_ok=True)
    log = open(RUN / f"ord{i}" / "server.log", "a")
    log.write(f"{STARTED} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    p = subprocess.Popen(ord_base(i) + ["server", "--http-port", str(ordport(i)), "--address", "127.0.0.1", "--polling-interval", "500ms"],
                         stdout=log, stderr=log, start_new_session=True)
    (RUN / f"ord{i}" / "server.pid").write_text(str(p.pid))


def ord_pids(i) -> list:
    """The ord server processes of node i (answering or not)."""
    pids = set()
    try:
        pid = int((RUN / f"ord{i}" / "server.pid").read_text())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if b"server" in cmd and str(RUN / f"ord{i}").encode() in cmd and Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            pids.add(pid)
    except (OSError, ValueError, IndexError):
        pass
    try:  # one started before there was a pid file
        r = subprocess.run(["pgrep", "-f", "--", f"--data-dir {RUN}/ord{i} --index-sats server"], capture_output=True, text=True)
        pids |= {int(x) for x in r.stdout.split()}
    except (FileNotFoundError, ValueError):
        pass
    return sorted(pids)


def ord_running(i) -> bool:
    return bool(ord_pids(i))


def ord_stop(i, patience=40) -> None:
    """Stop node i's ord server and wait until it is gone (it closes its index first)."""
    for k in range(patience * 2):
        pids = ord_pids(i)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM if k == 0 else signal.SIGKILL if k == patience * 2 - 20 else 0)
            except OSError:
                pass
        time.sleep(0.5)


def ord_forked(i):
    """True if ord's index holds, at its own height, another block than the node has there: it
    followed a chain the node has since left. None if that cannot be told right now."""
    try:
        h = min(ord_get(i, "/status")["height"], cli(i, "getblockcount", timeout=30))
        return ord_get(i, f"/r/blockhash/{h}") != cli(i, "getblockhash", h, timeout=30)
    except Exception:  # noqa: BLE001
        return None


SURE = ("its indexer stopped", "it holds an orphan")  # what ord does not get over by itself


def ord_height(i) -> int:
    """The height node i's index has committed (-1: nothing yet)."""
    h = ord_get(i, "/status")["height"]
    return -1 if h is None else h


def ord_since_start(i) -> str:
    """What node i's ord server has written to its log since it was last started."""
    try:
        with open(RUN / f"ord{i}" / "server.log", "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - 300_000, 0))
            text = f.read().decode(errors="replace")
    except OSError:
        return ""
    cut = text.rfind(STARTED)
    return text[max(cut if cut >= 0 else text.rfind("Listening on"), 0):]


def ord_panic(i):
    """The words with which the index thread of node i's ord server died, if it has since the
    server started. That thread has no name (the threads that answer requests have one, and a
    request that fails does not stop the indexer); once it is gone the server goes on answering
    from an index that never moves again."""
    lines = ord_since_start(i).splitlines()
    for k, l in enumerate(lines):
        if "panicked at" in l and "<unnamed>" in l:
            return " ".join(x.strip() for x in lines[k:k + 2])[:240]
    return None


def ord_sick(i, deep: int = 0):
    """Why node i's index cannot be relied on, or None (also when that cannot be told right now).

    "its indexer stopped at height H": the server's index thread died and the index is behind the
    node. The same answer twice, some seconds apart, is an index that will not move again.
    "it holds an orphan at height h": below its own height, within `deep` blocks, it has another
    block than the node. ord checks that a block links to the one before it only against what it
    has committed, so two blocks read in one batch across a reorganisation go in unlinked; the
    index then lacks the outputs of the block that replaced the orphan and stops at the first
    transaction that spends one (tests/check_ord_orphan.py does it to a real ord).
    "its tip is not the node's": at its own height it has another block than the node. That is
    also what a reorganisation looks like for the moment before ord takes it in."""
    try:
        at, node = ord_height(i), cli(i, "getblockcount", timeout=30)
        died = ord_panic(i)
        if died and at < node:
            return f"{SURE[0]} at height {at} ({died})"
        top = min(at, node)
        if top < 0:
            return None
        if ord_get(i, f"/r/blockhash/{top}") != cli(i, "getblockhash", top, timeout=30):
            return f"its tip is not the node's (height {top})"
        for h in range(top - 1, max(top - deep, 0) - 1, -1):
            if ord_get(i, f"/r/blockhash/{h}") != cli(i, "getblockhash", h, timeout=30):
                return f"{SURE[1]} at height {h}, under blocks of the node's chain"
    except Exception:  # noqa: BLE001
        return None
    return None


def ord_autopsy(i, limit: float = 90.0) -> list:
    """What exactly is wrong with node i's index, read from the server while it still answers:
    the heights at which it has another block than the node (the last 400), and, if its indexer
    stopped, the first input after its height whose output it does not have: where that output
    was made, and whether the index has the node's block there. For the record that is kept with
    an index that is set aside."""
    out, t0 = [], time.time()
    try:
        at, node = ord_height(i), cli(i, "getblockcount", timeout=30)
        out.append(f"index at height {at}, node at {node}")
        died = ord_panic(i)
        if died:
            out.append(f"index thread: {died}")
        off = []
        for h in range(min(at, node), max(min(at, node) - 400, -1), -1):
            if time.time() - t0 > limit / 2:
                out.append(f"(heights below {h} not compared)")
                break
            if ord_get(i, f"/r/blockhash/{h}") != cli(i, "getblockhash", h, timeout=30):
                off.append(h)
        out.append(f"heights at which the index has another block than the node: {sorted(off) if off else 'none'}")
        if not died:
            return out
        for h in range(at + 1, node + 1):
            if time.time() - t0 > limit:
                out.append(f"(inputs from height {h} on not looked at)")
                return out
            for tx in cli(i, "getblock", cli(i, "getblockhash", h, timeout=30), 2, timeout=60)["tx"]:
                for vin in tx["vin"]:
                    if "txid" not in vin:
                        continue
                    point = f"{vin['txid']}:{vin['vout']}"
                    try:
                        if ord_get(i, f"/output/{point}").get("indexed"):
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                    made = cli(i, "getblockheader", cli(i, "getrawtransaction", vin["txid"], 1, timeout=60)["blockhash"], timeout=30)
                    if made["height"] > at:
                        continue  # made after the index's height: the indexer would have had it at hand
                    theirs = ord_get(i, f"/r/blockhash/{made['height']}")
                    out.append(f"the first input the index cannot serve: block {h}, transaction {tx['txid']} spends {point}, made in block "
                               f"{made['height']} ({made['hash']}); the index has at that height "
                               + ("the same block, and not this output" if theirs == made["hash"] else f"another block ({theirs}): an orphan"))
                    return out
        out.append(f"no input in blocks {at + 1}..{node} that the index cannot serve")
    except Exception as e:  # noqa: BLE001
        out.append(f"(the look ended: {type(e).__name__}: {str(e)[:200]})")
    return out


def ord_rebuild(i, why: str = "", log=None) -> bool:
    """The index of node i is set aside (renamed, not deleted) and built again from the node's
    chain; the wallet is not in it. For an index that will not come right by itself: ord undoes a
    reorganisation only a few blocks deep (a node that mined by itself and then met the network
    leaves its own chain whole), and the cases of ord_sick. What was wrong with it is written
    next to it (ord_autopsy). An index whose thread died but which holds nothing but the node's
    chain is not rebuilt: its server is started again."""
    d = RUN / f"ord{i}"
    with open(d / "rebuild.lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False  # another process is at it
        try:
            again = ord_sick(i, 200)
            if again and again.startswith(SURE):  # the same some seconds later: a live indexer moves, and one that is
                time.sleep(3)                     # undoing a reorganisation by itself is through with it
                again = again if ord_sick(i, 200) == again else None
            if not again:
                return False
            say = log or (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True))
            stamp = time.strftime("%Y%m%d-%H%M%S")
            notes = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} ord {i}: {again}"] + ord_autopsy(i)
            if again.startswith(SURE[0]) and not any("another block than the node: [" in n for n in notes):
                # The index thread died, and what it had committed is the node's chain (it died inside a
                # batch, which was not committed). Started again, the server goes on from there: no need to
                # index the whole chain anew. If it dies again or does not move, the index is rebuilt after all.
                try:
                    at = ord_height(i)
                except Exception:  # noqa: BLE001
                    at = -1
                ord_stop(i)
                ord_up(i)
                t0, moved = time.time(), False
                while time.time() - t0 < 90 and not moved:
                    time.sleep(2)
                    try:
                        if ord_panic(i):
                            break
                        moved = ord_height(i) > at
                    except Exception:  # noqa: BLE001  (it is starting)
                        pass
                if moved and not ord_panic(i):
                    say(f"ord {i}: {again}; what it had committed is the node's chain: its server is started again and goes on from height {at}")
                    return True
                notes.append(f"started again at {time.strftime('%H:%M:%S')}, it " + ("died again" if ord_panic(i) else "did not move in 90 s"))
            (d / f"index-set-aside-{stamp}.txt").write_text("\n".join(notes) + "\n")
            ord_stop(i)
            if (d / "index.redb").exists():
                (d / "index.redb").rename(d / f"index-set-aside-{stamp}.redb")
            ord_up(i)
            say(f"ord {i}: {again}; its index is set aside and built again from the node's chain")
            for line in notes[1:]:
                say(f"   {line}")
            return True
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def ord_ensure(i, tries=5) -> None:
    """Node i's ord server answers: started if there is none, and started again if it gave up
    because the one before it was still closing the index."""
    for _ in range(tries):
        try:
            ord_get(i, "/status")
            return
        except Exception:  # noqa: BLE001
            pass
        if not ord_running(i):
            ord_up(i)
        for _ in range(60):
            try:
                ord_get(i, "/status")
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
            if not ord_running(i):
                break
        time.sleep(2)


def ord_error(stderr: str) -> str:
    """What ord said went wrong: its `error:` line and the reasons under it, without the backtrace."""
    lines = stderr.strip().splitlines()
    start = next((k for k, l in enumerate(lines) if l.startswith("error")), None)
    if start is None:
        return stderr.strip()[-600:]
    keep = []
    for l in lines[start:]:
        if l.strip().lower().startswith(("backtrace", "stack backtrace")) or (l[:1] == " " and l.strip()[:1].isdigit() and ": " in l):
            break
        if l.strip():
            keep.append(l.strip())
    return " ".join(keep)[:600]


# ord refuses to start a wallet command while another ord process has the wallet's database open, or
# while its index is a block behind the node (it gives the index one second). Both are decided
# before the command does anything, so the command is simply started again.
ORD_OPEN, ORD_BEHIND = "failed to open wallet database", "blocks behind `bitcoind`"
ORD_PATIENCE = int(os.environ.get("QWEN_ORD_PATIENCE", "240"))  # seconds


@contextmanager
def wallet_lock(i):
    """One `ord wallet` command at a time per node, across processes (the miner's node, its agent,
    its model, a person at the keyboard): they share one wallet database."""
    d = RUN / f"ord{i}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "wallet.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def ord_wallet(i, *args, check=True):
    cmd = ord_base(i) + ["wallet", "--server-url", f"http://127.0.0.1:{ordport(i)}"] + [str(a) for a in args]
    t0 = time.time()
    while True:
        with wallet_lock(i):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"ord{i} wallet {args}: no answer in 900s")
        if not r.returncode or time.time() - t0 > ORD_PATIENCE:
            break
        if ORD_BEHIND in r.stderr:  # a block arrived: wait for the index to take it in, then at once
            try:
                ord_wait(i, timeout=30)
            except Exception:  # noqa: BLE001
                pass
        elif ORD_OPEN in r.stderr:  # a process that does not take the lock above (an older one) has the wallet
            time.sleep(0.5)
        else:
            break
    if r.returncode and check:
        raise RuntimeError(f"ord{i} wallet {args}: {ord_error(r.stderr)}")
    try:
        return json.loads(r.stdout)
    except ValueError:
        return r.stdout.strip()


def ord_wallet_ready(i, log=print, tries=3) -> None:
    """Node i has ord's wallet, whole: created if there is none, loaded, not scanning, both of its
    descriptors active. ord's RPC library sends a call again when the first answer takes long, and
    the node answers the repeat with "Wallet is currently rescanning": `ord wallet create` then
    reports an error for an import that went through. So the node is asked what it has. A wallet
    that really is incomplete is renamed (never deleted) and made again."""
    for _ in range(tries):
        if "ord" not in [w["name"] for w in cli(i, "listwalletdir")["wallets"]]:
            try:
                r = ord_wallet(i, "create")
                if isinstance(r, dict) and r.get("mnemonic"):  # the wallet's recovery phrase: kept on the node
                    f = datadir(i) / "ord-wallet-recovery.json"
                    f.write_text(json.dumps(r))
                    os.chmod(f, 0o600)
            except RuntimeError as e:
                log(f"   n{i}: `ord wallet create` reported: {str(e)[-140:]} (asking the node what it has)")
        if "ord" not in cli(i, "listwallets"):
            cli(i, "loadwallet", "ord", check=False)
        whole = False
        for _ in range(1800):
            try:
                info = cli(i, "getwalletinfo", wallet="ord", timeout=60)
            except RuntimeError:
                time.sleep(1)
                continue
            if not info.get("scanning"):
                ds = cli(i, "listdescriptors", wallet="ord")["descriptors"]
                whole = {d.get("internal", False) for d in ds if d.get("active")} == {False, True}
                break
            time.sleep(1)
        if whole:
            return
        cli(i, "unloadwallet", "ord", check=False)
        d = datadir(i) / "ord"
        if d.exists():
            d.rename(d.with_name(f"ord-incomplete-{time.strftime('%Y%m%d-%H%M%S')}"))
            log(f"   n{i}: the wallet was incomplete; kept as {d.name}-incomplete-.., making a new one")
    raise RuntimeError(f"n{i}: ord's wallet could not be made")


def ord_get(i, path):
    import urllib.request
    req = urllib.request.Request(f"http://127.0.0.1:{ordport(i)}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


_fork = {}  # node -> [when this process last looked at its index, looks in a row that found it off, rebuilds, since when, last deep look, what the last look found]


def ord_heal(i) -> bool:
    """One look (at most every five seconds) at whether node i's index can be relied on
    (ord_sick); every half minute, and after a look that found something, the look goes 24 blocks
    deep. A reorganisation ord can undo is over within its polling interval, so a tip that differs
    is given a minute of looking (four looks or more, none more than a minute after the one
    before); a stopped indexer or a buried orphan is certain when the next look finds the very
    same. Then the index is rebuilt. True: it is not right at the moment."""
    st = _fork.setdefault(i, [0.0, 0, 0, 0.0, 0.0, None])
    now = time.time()
    if now - st[0] >= 5:
        gap, st[0] = now - st[0], now
        deep = 24 if st[1] or now - st[4] >= 30 else 0
        if deep:
            st[4] = now
        why, before = ord_sick(i, deep), st[5]
        st[5] = why
        if why:
            if st[1] == 0 or gap > 60:
                st[1], st[3] = 1, now
            else:
                st[1] += 1
            sure = why.startswith(SURE) and why == before and gap <= 60
            if (sure or (st[1] >= 4 and now - st[3] >= 60)) and ord_rebuild(i, why):
                st[1], st[2], st[5] = 0, st[2] + 1, None
        else:
            st[1] = 0
    return st[1] > 0


def ord_wait(i, height=None, timeout=300):
    """Wait until node i's ord has indexed its bitcoind's tip (or the given height). An index left
    on a chain the node has abandoned (ord_rebuild) never gets there by itself: it is built again."""
    t0 = time.time()
    while True:
        want = height if height is not None else cli(i, "getblockcount")
        there = False
        try:
            there = ord_height(i) >= want
        except Exception:  # noqa: BLE001
            pass
        built = _fork.get(i, [0, 0, 0])[2]
        off = ord_heal(i)
        if _fork[i][2] != built:
            t0 = time.time()  # counted from here: the whole chain is indexed again
        if there and not off:
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"ord{i} did not reach {want}")
        time.sleep(0.5)
