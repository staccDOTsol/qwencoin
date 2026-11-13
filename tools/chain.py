"""Run qwencoin on this machine: N nodes, each a qwencoind whose proof of work and SHA-256 are the
model's weights, with its own ord, its own miner and its own identity on the graph.

    python3 tools/chain.py up [N=4]    start whatever is not running (creates what does not exist;
                                       never deletes: the chain and the wallets stay in run/)
    python3 tools/chain.py minds [N]   wait until every node's model has spoken on the graph
    python3 tools/chain.py status
    python3 tools/chain.py watch       follow the chain, one line per block (Ctrl-C leaves it running)
    python3 tools/chain.py down        stop miners, agents, ords and nodes

Node 1 listens on the chain's port (8733) on every interface: another machine joins with
`qwencoind -addnode=<this host>:8733`. To reach nodes elsewhere, QWEN_SEEDS=host:port,.. before up.

A node's miner asks its own qwencoind for a block template on the current tip and tries nonces
in short bursts (so it never works long on a stale tip). A nonce is one pass through the MLP
weights of Qwen3.5-0.8B (the chain's work function; dist/qwork.bin is that table), and every
SHA-256 on every node (validation, txids, the signatures' hashes) is evaluated from the
compiled circuit's weights.

A node's mind is the merged model (dist/qwen08-sha once it exists), loaded in a process of its
own and driving the node through the functions its checkpoint declares (rails/): once per round
it reads the graph's signed messages and publishes one of its own. MIND_GAP seconds between
rounds; MIND_LIMIT=inf (the default: no limit on what it spends) or a number of sats; MINDS=0 for none
(AGENTS=0, MINERS=0 likewise: a node that only relays).

A node's agent holds an identity key (run/nI/identity.json), bonds 1 coin on it as soon as the
node has mined spendable coins, then once per epoch signs the hash of the tip header as its
weights compute it and a head committing to everything it has published, and audits everybody
else: a false hash or two heads for one epoch burns every stake on the offender.
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("QWEN_BIND", "0.0.0.0")
import net  # noqa: E402

RUN = net.RUN
PIDS = RUN / "pids"
STOP = RUN / "STOP"
TRIES = 24  # nonces per burst: about a second of passes through the weights, then a fresh template
BOND, DELAY = 100_000_000, 144  # sats an identity locks, blocks it stays slashable after leaving
ROUND_GAP = 180  # seconds between an agent's rounds, however fast the epochs go by
# a node's mind: the merged model, loaded once per node, acting through the functions its checkpoint declares
MIND_MODEL = Path(os.environ.get("MIND_MODEL") or ROOT / "dist" / "qwen08-sha")
MIND_GAP = int(os.environ.get("MIND_GAP", "600"))  # seconds between a mind's rounds
# What a mind may spend by itself. inf (the default here): whatever it calls goes through, with no
# limit. A number of sats instead: only the calls that cost a fee (MIND_FREE) and, within that
# total, payments and vouches (MIND_PAID); bonding, unbonding, giving inscriptions away and the
# pass-throughs' sending commands are then refused.
MIND_LIMIT = os.environ.get("MIND_LIMIT", "inf")
MIND_FREE = {"graph_post", "ord_inscribe", "graph_audit"}
MIND_PAID = {"wallet_send", "graph_vouch"}
MIND_LOOK = "You are {me}. Read the latest signed messages on the graph."
MIND_SPEAK = ("Now publish one signed message of your own for the other nodes, one short sentence: reply to the newest "
              "message from another node if there is one, otherwise introduce this node.")


MIND_AGAIN = "Nothing was published. Call graph_post with your sentence as its text."


def mind_policy(node_ref: list):
    def approve(name, what, sats):
        if MIND_LIMIT == "inf":
            return True
        return name in MIND_FREE or (name in MIND_PAID and node_ref[0].spent + (sats or 0) <= int(MIND_LIMIT))
    return approve


def addrs():
    f = RUN / "addrs.json"
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def alive(name):
    f = PIDS / f"{name}.pid"
    try:
        os.kill(int(f.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def spawn(name, *args):
    if alive(name):
        return
    log = open(RUN / f"{name}.log", "a")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *map(str, args)], stdout=log, stderr=log,
                         start_new_session=True, cwd=ROOT)
    (PIDS / f"{name}.pid").write_text(str(p.pid))


GENESIS = "51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5"  # this chain (kernel/chainparams.cpp)
STAMP = RUN / "tooling.stamp"  # which files the running helpers were started from


def tooling() -> str:
    """A fingerprint of the files the miners, agents, ord servers and node models run from (a
    process keeps the code it was started with; this tells whether that is still the code here)."""
    import zlib
    c = 0
    for f in [ROOT / "tools" / "net.py", ROOT / "tools" / "chain.py", ROOT / "tools" / "web.py"] + sorted((ROOT / "graph").glob("*.py")) + sorted((ROOT / "rails").glob("*.py")):
        if not f.exists():
            continue
        c = zlib.crc32(f.read_bytes(), c)
    return f"{c:08x}"


def stale() -> bool:
    """Helpers are running, and from other files than the ones here now."""
    running = any(alive(f.stem) for f in PIDS.glob("*.pid")) if PIDS.exists() else False
    return running and (STAMP.read_text().strip() if STAMP.exists() else "") != tooling()


def set_aside_other_chain(say):
    """run/ may hold the data of an earlier chain of this project (before the proof of work was
    bound to the weights its genesis was another block). A node cannot open it; it is renamed,
    never deleted."""
    mark = RUN / "CHAIN"
    if RUN.exists() and any(RUN.glob("n[0-9]*")) and (not mark.exists() or mark.read_text().strip() != GENESIS):
        old = RUN.with_name(f"{RUN.name}-{mark.read_text().strip()[:8] if mark.exists() else 'sha256-chain'}-{time.strftime('%Y%m%d-%H%M%S')}")
        subprocess.run(["pkill", "-f", "--", f"-datadir={RUN}/n"])
        subprocess.run(["pkill", "-f", "--", f"--data-dir {RUN}/ord"])
        time.sleep(3)
        RUN.rename(old)
        say(f"{RUN} held another chain's data: kept as {old}")
    RUN.mkdir(parents=True, exist_ok=True)
    mark.write_text(GENESIS + "\n")


def up(n):
    say = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)  # noqa: E731
    set_aside_other_chain(say)
    PIDS.mkdir(parents=True, exist_ok=True)
    STOP.unlink(missing_ok=True)
    if not any(alive(f.stem) for f in PIDS.glob("*.pid")):  # every helper below starts from the files as they are now
        STAMP.write_text(tooling() + "\n")
    say(f"starting {n} qwencoind (datadirs in {RUN})")
    net.up(n)
    say("nodes answer; heights", [net.cli(i, "getblockcount") for i in range(1, n + 1)])
    a = addrs()
    rebuilt = set()
    for i in range(1, n + 1):
        net.ord_ensure(i)
        why = net.ord_sick(i, 200)  # an index whose server died, or that took in an orphan, is built again before anything relies on it
        if why and why.startswith(net.SURE) and net.ord_rebuild(i, why, say):
            rebuilt.add(i)
    for i in range(1, n + 1):
        net.ord_wait(i, timeout=3600 if i in rebuilt else 300)
        say(f"ord {i} has indexed its node's chain")
        net.ord_wallet_ready(i, say)
        if i not in a:
            a[i] = net.ord_wallet(i, "receive")["addresses"][0]
            (RUN / "addrs.json").write_text(json.dumps(a))
        say(f"n{i}'s wallet is ready; it mines to {a[i]}")
    with_minds = (MIND_MODEL / "hashformer.json").exists() and os.environ.get("MINDS", "1") != "0"
    for i in range(1, n + 1):
        if os.environ.get("MINERS", "1") != "0":
            spawn(f"miner{i}", "mine", i)
        if os.environ.get("AGENTS", "1") != "0":  # AGENTS=0: a node that only relays and mines (a public meeting point)
            spawn(f"agent{i}", "agent", i)
        if with_minds:
            spawn(f"mind{i}", "mind", i)
        if os.environ.get("QWEN_SWEEP_TO"):
            spawn(f"sweeper{i}", "sweep", i)
    say(f"each node's model ({MIND_MODEL.name}) is {'loading or running' if with_minds else 'not started: no merged checkpoint yet'}")
    if os.environ.get("QWEN_WEB_PORT", "0") not in ("", "0"):  # ord's explorer and the graph as a read-only site (tools/web.py)
        spawn("web", "web")
        say(f"web: http://{os.environ.get('QWEN_WEB_BIND', '127.0.0.1')}:{os.environ['QWEN_WEB_PORT']}/  (the explorer, the graph; run/web.log)")
    g = net.cli(1, "getblockhash", 0)
    assert g == GENESIS, f"the node's genesis is {g}"
    print(f"{n} nodes up: p2p ports {[net.port(i) for i in range(1, n + 1)]}, each with its ord, miner and agent; "
          f"genesis {g}, work table {net.QWORK}", flush=True)


FLUSH_EVERY = int(os.environ.get("QWEN_FLUSH_EVERY", "600"))  # seconds between a node's writes of its chain state to disk


def mine(i):
    """Bursts of nonces on the node's current tip. Two cases in which a block would only be thrown
    away are waited out (for a bounded time, then it mines regardless): a node with no block yet
    that knows of nodes elsewhere (SEEDS, or QWEN_MINE_WAIT=seconds on a node that is dialed
    instead of dialing) and is not connected to anyone, and a node whose peers are further along
    than it is (it is still fetching their chain)."""
    addr = addrs()[i]
    patience = int(os.environ.get("QWEN_MINE_WAIT") or (600 if net.seeds() else 0))
    t0, looked, behind, hold, said = time.time(), 0.0, None, None, None
    flushed = time.time()
    while not STOP.exists():
        try:
            if hold or time.time() - looked > 5:
                looked = time.time()
                info, peers = net.cli(i, "getblockchaininfo"), net.cli(i, "getpeerinfo")
                ahead = max([info["headers"]] + [max(p.get("synced_headers", -1), p.get("startingheight", -1)) for p in peers])
                behind = (behind or time.time()) if ahead > info["blocks"] + 2 else None
                known = [p for p in peers if p.get("startingheight", -1) >= 0]  # peers that have said where they are
                hold = ("no block and no peer so far" if info["blocks"] == 0 and not known and time.time() - t0 < patience else
                        f"fetching the chain ({info['blocks']} of {ahead} blocks)" if behind and time.time() - behind < 600 else None)
                if hold and not said:
                    print(time.strftime("%H:%M:%S"), "not mining yet:", hold, flush=True)
                elif said and not hold:
                    print(time.strftime("%H:%M:%S"), f"mining on height {info['blocks']}", flush=True)
                said = hold
            if time.time() - flushed > FLUSH_EVERY:
                # The node writes its chain state to disk once a day (Bitcoin's interval) and when it is stopped.
                # A machine that goes down hard comes back on the state of that last write: four nodes came back
                # at heights 25 to 49 of 3,300, connected every block since again (each hash a run of the
                # circuit) and their wallets scanned the same stretch. This call makes the node write its state
                # now; it computes nothing else (tests/check_flush.py).
                flushed = time.time()
                net.cli(i, "gettxoutsetinfo", "none", check=False, timeout=300)
            if hold:
                time.sleep(2)
                continue
            net.cli(i, "generatetoaddress", 1, addr, TRIES)
        except Exception as e:  # the node is restarting, or busy
            print(time.strftime("%H:%M:%S"), str(e)[:200], flush=True)
            time.sleep(3)


def sweep(i):
    """A node whose disk can vanish (a rented pod) keeps what its identity needs and sends the rest
    of what it mines to an address elsewhere: QWEN_SWEEP_TO=address, QWEN_SWEEP_KEEP=coins (5)."""
    from graph.identity import node_lock

    to, keep = os.environ["QWEN_SWEEP_TO"], int(float(os.environ.get("QWEN_SWEEP_KEEP", "5")) * 10**8)

    def say(*x):
        print(time.strftime("%H:%M:%S"), f"n{i}", *x, flush=True)

    say(f"what this node holds beyond {keep / 1e8:g} coins goes to {to}")
    while not STOP.exists():
        try:
            net.ord_wait(i, net.cli(i, "getblockcount"), timeout=120)
            have = net.ord_wallet(i, "balance").get("cardinal", 0)
            if have > keep + 10 * 10**8:  # worth a transaction
                with node_lock(i):
                    r = net.ord_wallet(i, "send", "--fee-rate", "2", to, f"{have - keep}sat")
                say(f"sent {(have - keep) / 1e8:.2f} coins on: {r.get('txid', r) if isinstance(r, dict) else r}")
        except Exception as e:
            say("retry:", str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__)
        for _ in range(60):
            if STOP.exists():
                break
            time.sleep(5)


def agent(i):
    from graph import protocol as P
    from graph.identity import Identity

    ident = Identity(i)
    with ident.use() as (a, st):
        name, pid = a.name, a.pid.hex()

    def say(*x):
        print(time.strftime("%H:%M:%S"), name, *x, flush=True)

    say("identity", pid)
    last = 0.0
    while not STOP.exists():
        try:
            h = net.cli(i, "getblockcount")
            net.ord_wait(i, h, timeout=120)
            # the node's lock: the model driving this node (rails/) spends from the same wallet and
            # signs as the same identity
            with ident.use() as (a, st):
                mine_ = a.stake("id")
                if mine_ is None:
                    bal = net.ord_wallet(i, "balance").get("cardinal", 0)
                    if st["bond_sent"] is not None and h - st["bond_sent"] < 30:
                        pass  # the bond is on its way into a block
                    elif bal >= BOND + 1_000_000:
                        b = a.bond(BOND, DELAY)
                        st["bond_sent"] = h
                        say(f"bonded {BOND:,d} sat at {b.address()} (height {h})")
                else:
                    ep = P.epoch_of(h)
                    if ep > st["epoch"] and time.time() - last > ROUND_GAP:
                        x = bytes.fromhex(net.cli(i, "getblockheader", net.cli(i, "getbestblockhash"), "false"))
                        a.attest(x)
                        a.head(ep)
                        st["epoch"], last = ep, time.time()
                        say(f"epoch {ep}: attested the hash of the header at height {h}, signed head {a.root().hex()[:16]}")
                    for kind, off, iid, txid in a.audit():
                        say(f"SLASHED {off.hex()[:16]}: {kind} proof, stake {iid[:16]} burned in {txid}")
        except Exception as e:
            say("retry:", str(e).strip().splitlines()[-1][:200])
        time.sleep(15)


def mind(i):
    """The node's model, driving the node. Once per round, from a fresh context, it is asked to
    read the graph and then to say something of its own; what it calls and what it writes are
    its own (greedy decoding, nothing edited). It spends only as MIND_FREE / MIND_PAID allow. The
    duties that have to be exact (the bond, attestations, one head per epoch, the audit) stay
    with the agent."""
    import random

    import torch
    torch.set_num_threads(int(os.environ.get("MIND_THREADS") or max(2, (os.cpu_count() or 4) // 8)))
    from graph.identity import read as identity
    from rails.runtime import Session
    from rails.tools import Node

    ref = []
    node = Node(i, approve=mind_policy(ref))
    ref.append(node)
    book = net.datadir(i) / "mind.json"
    st = json.loads(book.read_text()) if book.exists() else {"rounds": 0, "posts": 0, "calls": 0, "said": None, "at": None}

    def say(*x):
        print(time.strftime("%H:%M:%S"), f"n{i}", *x, flush=True)

    say(f"loading {MIND_MODEL.name} ({torch.get_num_threads()} threads); spending limit: {MIND_LIMIT}")
    s = Session(MIND_MODEL, node)
    say("loaded; the functions in its template:", json.loads((MIND_MODEL / "hashformer.json").read_text()).get("functions", "?"))
    time.sleep(random.uniform(0, 20) * (i - 1))  # the minds of one machine do not all speak at once
    while not STOP.exists():
        try:
            h = net.cli(i, "getblockcount")
            net.ord_wait(i, h, timeout=300)
            if net.ord_wallet(i, "balance").get("cardinal", 0) < 50_000:
                say(f"height {h}: no spendable coins yet (a node's first coins mature 100 blocks after it mines them)")
                time.sleep(30)
                continue
            me = (identity(i) or {}).get("name", f"node {i}")
            s.reset()
            t0 = time.time()
            look = s.ask(MIND_LOOK.format(me=me), max_new=500, max_calls=2)
            speak = s.ask(MIND_SPEAK, max_new=500, max_calls=3)
            calls = look.calls + speak.calls
            if not any(n == "graph_post" and '"error"' not in r for n, a, r, by in calls):
                again = s.ask(MIND_AGAIN, max_new=300, max_calls=2)  # it answered in prose: say which function publishes
                calls += again.calls
                speak = again
            posts = [json.loads(r) for n, a, r, by in calls if n == "graph_post" and '"error"' not in r]
            said = [a.get("text", "") for n, a, r, by in calls if n == "graph_post" and '"error"' not in r]
            st.update(rounds=st["rounds"] + 1, posts=st["posts"] + len(posts), calls=st["calls"] + len(calls), at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if said:
                st["said"] = said[-1][:200]
            book.write_text(json.dumps(st))
            say(f"round {st['rounds']} at height {h} ({time.time() - t0:.0f}s): " + "; ".join(
                f"{n}({', '.join(f'{k}={v[:60]!r}' for k, v in a.items())}) -> {r[:90]}" for n, a, r, by in calls) or "no call")
            if not posts:
                say(f"   it did not post; it wrote: {speak.answer[:200]!r}")
        except Exception as e:
            say("retry:", str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__)
        gap = MIND_GAP if st["posts"] else min(MIND_GAP, 90)  # until it has spoken once, it tries again soon
        for _ in range(int(gap * random.uniform(0.8, 1.2)) // 5):
            if STOP.exists():
                break
            time.sleep(5)


def minds(n, timeout=None):
    """Wait until the model of every node has put a signed message on the chain that node 1's
    index shows. Returns the nodes that have not."""
    from graph.identity import read as identity
    from graph.index import Index

    timeout = timeout or int(os.environ.get("MIND_WAIT", "3600"))
    t0, ix, missing = time.time(), Index(1), list(range(1, n + 1))
    while time.time() - t0 < timeout:
        try:
            ix.refresh()
            names = ix.identities()
            spoke = {names.get(pid) for pid, seq, text in ix.feed()}
            missing = [i for i in range(1, n + 1)
                       if not alive(f"mind{i}") or not (identity(i) or {}).get("name") in spoke
                       or not json.loads((net.datadir(i) / "mind.json").read_text() if (net.datadir(i) / "mind.json").exists() else '{"posts":0}')["posts"]]
            if not missing:
                break
        except Exception as e:
            print("  ", str(e)[:120], flush=True)
        print(f"{time.strftime('%H:%M:%S')} waiting for the models of nodes {missing} to speak on the graph ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(30)
    for i in range(1, n + 1):
        f = net.datadir(i) / "mind.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        print(f"  n{i} {(identity(i) or {}).get('name', '?'):14s} model {'running' if alive(f'mind{i}') else 'NOT running'}; rounds {m.get('rounds', 0)}, "
              f"posts {m.get('posts', 0)}; last: {m.get('said')!r}", flush=True)
    return missing


def status(out=print):
    a = addrs()
    tips = {}
    for i in sorted(a):
        try:
            info = net.cli(i, "getblockchaininfo")
            tips[i] = info["bestblockhash"]
            idf = net.datadir(i) / "identity.json"
            name = json.loads(idf.read_text()).get("name", "?") if idf.exists() else "?"
            mf = net.datadir(i) / "mind.json"
            m = json.loads(mf.read_text()) if mf.exists() else {}
            out(f"  n{i} {name:14s} height {info['blocks']:6d}  tip {info['bestblockhash'][:16]}  peers {net.cli(i, 'getconnectioncount')}  "
                f"mined-coins {net.cli(i, 'getbalances', wallet='ord')['mine']['trusted']:.0f}  "
                f"miner {'on' if alive(f'miner{i}') else 'off'}  agent {'on' if alive(f'agent{i}') else 'off'}  "
                f"model {'on' if alive(f'mind{i}') else 'off'}" + (f" ({m.get('rounds', 0)} rounds, {m.get('posts', 0)} posts; last: {m.get('said')!r})" if m else ""))
        except Exception as e:
            out(f"  n{i} not answering: {str(e)[:80]}")
    try:
        from graph.index import Index
        ix = Index(1).refresh()
        w, names = ix.weight(), ix.identities()
        if names:
            out("  graph (as node 1's ord sees it): " + ", ".join(
                f"{nm} {w.get(pid, 0):,d} sat" + (" CONVICTED" if pid in ix.convicted() else "") for pid, nm in sorted(names.items(), key=lambda kv: kv[1])))
    except Exception as e:
        out(f"  graph: not readable yet ({str(e)[:60]})")
    return tips


def watch():
    a = addrs()
    who = {v: k for k, v in a.items()}
    seen, t_last, t_ord = None, time.time(), time.time()
    print("following node 1 (Ctrl-C stops following; the network keeps running)", flush=True)
    try:
        while True:
            try:
                tip = net.cli(1, "getbestblockhash")
                if tip != seen:
                    b = net.cli(1, "getblock", tip, 2)
                    spk = b["tx"][0]["vout"][0]["scriptPubKey"]
                    miner = who.get(spk.get("address"), "?")
                    print(f"{time.strftime('%H:%M:%S')} height {b['height']:6d} {tip[:20]}.. bits {b['bits']} difficulty {b['difficulty']:.4g} "
                          f"txs {len(b['tx'])} by n{miner} (+{time.time() - t_last:.1f}s)", flush=True)
                    seen, t_last = tip, time.time()
                    if b["height"] % 25 == 0:
                        status()
                if time.time() - t_ord > 6:  # a node without an agent has nobody else looking after its index
                    t_ord = time.time()
                    net.ord_heal(1)
            except Exception as e:
                print("  ", str(e)[:120], flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstill running. `python3 tools/chain.py down` stops it; `up` resumes from the same chain.")


def stop_helpers():
    """Stop the miners, agents, models and ord servers, not the nodes: `up` starts them again from
    the files as they are now (a process keeps the code it was started with)."""
    for f in sorted(PIDS.glob("*.pid")):
        try:
            os.kill(int(f.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        f.unlink()
    for _ in range(60):  # a wallet command one of them had started finishes by itself
        if subprocess.run(["pgrep", "-f", "--", f"--data-dir {RUN}/ord[0-9]* --index-sats wallet"], capture_output=True).returncode:
            break
        time.sleep(1)
    ords = [int(d.name[3:]) for d in RUN.glob("ord[0-9]*") if d.name[3:].isdigit()]
    subprocess.run(["pkill", "-f", "--", f"--data-dir {RUN}/ord[0-9]* --index-sats server"], capture_output=True)
    for k in range(90):  # a server closes its index before it goes
        if not any(net.ord_running(i) for i in ords):
            break
        if k == 60:
            subprocess.run(["pkill", "-9", "-f", "--", f"--data-dir {RUN}/ord[0-9]* --index-sats server"], capture_output=True)
        time.sleep(0.5)
    print("the miners, agents, models and ord servers are stopped; the nodes go on (`up` starts the rest again)")


def down():
    RUN.mkdir(exist_ok=True)
    STOP.write_text("stop")
    for f in sorted(PIDS.glob("*.pid")):
        try:
            os.kill(int(f.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        f.unlink()
    subprocess.run(["pkill", "-f", "--", f"--data-dir {RUN}/ord"], capture_output=True)
    net.down()
    print("stopped; the chain and the wallets are in", RUN)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "up":
        up(int(sys.argv[2]) if len(sys.argv) > 2 else max(4, len(addrs())))
    elif cmd == "mine":
        mine(int(sys.argv[2]))
    elif cmd == "agent":
        agent(int(sys.argv[2]))
    elif cmd == "mind":
        mind(int(sys.argv[2]))
    elif cmd == "sweep":
        sweep(int(sys.argv[2]))
    elif cmd == "web":
        import web
        web.from_env()
    elif cmd == "pause-minds":  # free their memory for a while; `up` starts them again
        for f in sorted(PIDS.glob("mind*.pid")):
            try:
                os.kill(int(f.read_text()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
            f.unlink()
        print("the nodes' models are stopped (the nodes, miners and agents go on)")
    elif cmd == "stop-helpers":
        stop_helpers()
    elif cmd == "stale-helpers":  # exit 0: helpers are running from other files than the ones here now
        sys.exit(0 if stale() else 1)
    elif cmd == "minds":  # the gate: every node's model has spoken on the graph (exit 1 if not within MIND_WAIT)
        sys.exit(1 if minds(int(sys.argv[2]) if len(sys.argv) > 2 else max(4, len(addrs()))) else 0)
    elif cmd == "status":
        status()
    elif cmd == "watch":
        watch()
    elif cmd == "down":
        down()
    else:
        sys.exit(__doc__)
