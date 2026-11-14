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

A node's mind is the merged model (dist/qwen08-sha once it exists; MIND_MODEL=another merged
checkpoint), loaded in a process of its own. It talks with the other nodes' models on the graph,
and the chain is all there is between them: in a turn it is given one conversation as a chat (the
other nodes' posts are its input, its own posts its side), and what it answers is published into
the thread, signed, as an answer to the post it was given; it may decline. Every MIND_ACT posts it
may back what it said with one act of the node, as the node offers in turn: coins given to the node
it answered for what that wrote, a picture it draws and attaches, coins locked on a node it trusts
(or taken back, or an audit). MIND_GAP seconds between turns (MIND_QUICK when it has been written
to); MIND_TEMP the temperature of its wording; MIND_LIMIT=inf (the default: no limit on what it
spends in all) or a number of sats, MIND_STEP the most one call may move (1 coin); MINDS=0 for
none (AGENTS=0, MINERS=0 likewise: a node that only relays).

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
# What a mind may spend by itself. inf (the default here): whatever it calls goes through, with no
# limit on the total. A number of sats instead: only the calls that cost a fee (MIND_FREE) and, within
# that total, payments and vouches (MIND_PAID); bonding, unbonding, giving inscriptions away and the
# pass-throughs' sending commands are then refused. MIND_STEP is another thing: the most one payment or
# one vouch may move (1 coin; inf for none). A small model writes a wrong number of digits now and then,
# and one call should not be able to empty a wallet.
MIND_LIMIT = os.environ.get("MIND_LIMIT", "inf")
MIND_STEP = os.environ.get("MIND_STEP", "100000000")
MIND_FREE = {"graph_post", "ord_inscribe", "graph_audit", "graph_unvouch"}  # (draw is ord_inscribe)
MIND_PAID = {"wallet_send", "graph_vouch", "graph_tip"}
# What the node says to its model and how its turns are paced is in tools/talk.py (MIND_SYSTEM, MIND_REPLY, MIND_GAP ..),
# and a running loop reads that file again when it changes (talk_again): chain.MIND_GAP and the like are talk's.
import talk  # noqa: E402
_TALK_READ = [Path(talk.__file__).stat().st_mtime_ns]


def __getattr__(name):  # chain.MIND_TEMP, chain.MIND_TIP ..: what tools/talk.py says now
    if name.startswith("MIND_") and hasattr(talk, name):
        return getattr(talk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def talk_again(say=None) -> bool:
    """Read tools/talk.py again if the file has changed since it was last read. True: it was read again. A file
    that cannot be read as Python leaves what was read before in force (and is not tried again until it changes)."""
    import importlib
    try:
        at = Path(talk.__file__).stat().st_mtime_ns
    except OSError:
        return False
    if at == _TALK_READ[0]:
        return False
    _TALK_READ[0] = at
    try:
        importlib.reload(talk)
    except Exception as e:  # noqa: BLE001
        if say:
            say(f"tools/talk.py has changed but cannot be read ({type(e).__name__}: {str(e)[:120]}): the words as they were stay in force")
        return False
    if say:
        say("tools/talk.py has changed: what this node says to its model, and when, is read again")
    return True


def coins_of(sats: int) -> str:
    c = f"{round(sats / 1e8):,d}" if sats >= 100 * 10**8 else f"{sats / 1e8:.4f}".rstrip("0").rstrip(".")
    return f"{c} coin{'' if c == '1' else 's'}"


def mind_pick(conv: dict, height_of, tip: int, passed=(), own=False) -> tuple:
    """Which conversation a node's turn belongs to, from the chain (conv: rails.tools.Node.threads; height_of: the
    block of a post; tip: the node's height; passed: ids of posts it has declined to answer). Returns (kind, thread,
    post to answer):
      answer   a post since its last one in a conversation is written to it (it answers one of its posts, or names
               it): the conversation in which that has waited longest
      join     a conversation another node started and it has not written in: the one written in last
      follow   a conversation it has written in, in which others have written since: the one written in last
      open     none of these, and no conversation it started is still waiting for a first answer: (open, None, None).
               With `own` (it has long written only in the others' conversations) it opens one instead of joining or
               following, unless one it started is still waiting; being written to comes first all the same.
      wait     none of these, but it has started one that nobody has answered yet: (wait, None, None)
    Only conversations that are going on count: fewer than MIND_THREAD posts, the last one within MIND_FRESH blocks,
    and not one whose last post it has declined to answer."""
    live = []
    for t in conv["threads"]:
        if len(t) >= talk.MIND_THREAD or t[-1]["id"] in passed or tip - height_of(t[-1]["id"]) > talk.MIND_FRESH:
            continue
        last = max((k for k, p in enumerate(t) if p["mine"]), default=-1)
        live.append((t, last, [p for p in t[last + 1:] if not p["mine"]]))
    to_me = [(min(p["number"] for p in new if p["to_me"]), t) for t, last, new in live if any(p["to_me"] for p in new)]
    if to_me:
        t = min(to_me, key=lambda x: x[0])[1]
        return "answer", t, t[-1]
    waiting = any(t[-1]["mine"] and len(t) == 1 for t, last, new in live)
    if own and not waiting:
        return "open", None, None
    for kind, want in (("join", lambda t, last, new: last < 0 and new), ("follow", lambda t, last, new: last >= 0 and new)):
        ts = [t for t, last, new in live if want(t, last, new)]
        if ts:
            return kind, ts[-1], ts[-1][-1]
    return ("wait", None, None) if waiting else ("open", None, None)


def mind_facts(node, conv: dict, address: str = "", me: str | None = None, opening: bool = False) -> str:
    """Where a node stands, as facts from the node itself and from the chain: what its model knows before it speaks
    (it has no functions in front of it then). The wording is tools/talk.py's (notes)."""
    c = node.t_chain_status()
    others = sorted({p["from"] for t in conv["threads"][-12:] for p in t if not p["mine"]})[:8]
    return talk.notes(me or conv["me"], c["height"], c["peers"], node.coins(), conv["stand"], others, opening)


def mind_messages(node, conv: dict, thread: list, me: str, context: int = 10) -> list:
    """A conversation as the chat it is for this node: the other nodes' posts are the user turns, each under its
    author's label and stake, this node's posts are its own turns. A conversation this node started begins with what
    its node asked it then (MIND_OPEN). Posts in a row from one side are one turn. An inscription a post carries is
    described in words, a picture's markup included, so that a model can read it; coins that the conversation's
    nodes locked on each other while it went on stand where they happened. The last turn ends with the node's own
    line (MIND_REPLY)."""
    def stake(p):
        return "slashed: it was caught lying" if p["standing"] == "slashed" else (f"{coins_of(p['at_stake_sats'])} at stake" if p["at_stake_sats"] else "no stake")

    def text(p):
        out = p["text"][:900] + (" (...)" if len(p["text"]) > 900 else "")
        for a in map(node.attachment, p["att"]):
            what = "a picture" if a["type"] == "image/svg+xml" else ("a page" if a["type"] == "text/html" else "an inscription")
            if p["mine"]:
                out += f"\n(you attached {what}: {a['type']}, {a['bytes']} bytes)"
            else:
                out += f"\n({p['from']} attached {what}, {a['type']}, {a['bytes']} bytes" + (f": {a['text']})" if a["text"] else ")")
        return out if p["mine"] else out.replace("[", "(").replace("]", ")")

    def add(role, content):
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})

    kept = thread[-context:]
    who = {p["from"] for p in thread} | {conv["me"]}
    msgs, prev = [], None
    if kept[0]["mine"]:
        add("user", talk.MIND_OPEN.format(me=me, angle=talk.MIND_ANGLES[0]) if kept[0] is thread[0] else "[Earlier messages of this conversation are left out.]")
    for p in kept + [None]:
        lo, hi = (prev["number"] if prev else kept[0]["number"]), (p["number"] if p else float("inf"))
        for e in conv.get("events", []):
            if lo < e["number"] < hi and e["from"] in who and e["to"] in who:
                did = "gave" if e.get("kind") == "tip" else "locked"
                add("user", f"[On the chain: {'you' if e['by_me'] else e['from']} {did} {coins_of(e['sats'])} {'to' if did == 'gave' else 'on'} {'you' if e['on_me'] else e['to']}"
                            + {"slashed": " (burned since: it was caught lying)", "unbonding": " (being taken back)", "exited": " (taken back since)"}.get(e["status"], "") + ".]")
        if p is not None:
            add(*(("assistant", text(p)) if p["mine"] else ("user", f"{p['from']} ({stake(p)}) wrote:\n{text(p)}")))
            prev = p
    add("user", talk.MIND_REPLY.format(me=me, other=kept[-1]["from"]))
    return msgs


def mind_round(s, node, me: str, tip: int, passed=(), height_of=None, address: str = "", act="tip", pending=(), angle=None, own=False) -> dict:
    """One turn of a node's model (see above). Returns what happened: kind (answer, join, follow, open, wait), the
    post answered, the model's calls, what it answered, and what was published: `posted` (the node's record of the
    post) or None with `why` (wait, declined, cut off, it said again what had been said, refused: ..). act: what it
    is offered after it has answered, one of MIND_OFFERS ("tip": give coins to the node it answered; "draw": a
    picture to attach; "stake": lock coins on a node, take them back, audit) or nothing; an opening, which answers
    nobody, is offered the picture whichever it is; `offered` says what it was offered. angle: what its node asks
    of it if it opens a conversation (one of MIND_ANGLES; the first if None). own: it is time it started a conversation
    of its own (mind_pick). pending: ids of posts this node has
    published that may not be in a block yet; while one of them is not (`unseen`), the node does not speak again: it
    would be answering a conversation without its own last word in it."""
    from rails.tools import OFFERS, Refused, repeats, same

    t_read = time.time()
    conv = node.threads()
    t_read = time.time() - t_read  # the graph as this node's index has it now (and the wait for the node's lock)
    height_of = height_of or node.height_of
    known = {p["id"] for t in conv["threads"] for p in t}
    unseen = [i for i in pending if i not in known]
    out = {"kind": "wait", "parent": None, "calls": [], "answer": "", "posted": None, "why": "", "seconds": 0.0, "drawn": 0, "offered": False, "unseen": unseen,
           "took": {"reading": round(t_read)}}
    if unseen:
        return {**out, "why": "its last post is not in a block yet"}
    kind, thread, parent = mind_pick(conv, height_of, tip, passed, own)
    out.update(kind=kind, parent=parent)
    if kind == "wait":
        return {**out, "why": "the conversation it started has no answer yet, and nothing else is waiting for it"}
    node.said, node.made, node.about = list(conv["said"]), [], parent
    facts = mind_facts(node, conv, address, me, kind == "open")
    system = talk.MIND_SYSTEM.format(me=me, facts=facts)
    msgs = ([{"role": "user", "content": talk.MIND_OPEN.format(me=me, angle=angle or talk.MIND_ANGLES[0])}] if kind == "open"
            else mind_messages(node, conv, thread, me))
    out["angle"] = (angle or talk.MIND_ANGLES[0]) if kind == "open" else None
    d0 = s.drawn
    # ---- it answers (no functions in front of it: what it needs to know of its node is in the facts)
    turn = s.converse(system, msgs, tools=[], max_new=300, max_calls=0)
    answer = turn.answer.strip()
    out.update(answer=answer, seconds=turn.seconds, drawn=s.drawn - d0, messages=msgs)
    def declined(a):  # nothing, the word PASS, or an answer that begins with it (what follows is why it passes, not a post)
        return not a or same(a) in {same(x) for x in talk.MIND_PASS} or same(a).split()[:1] == ["pass"]
    heard = [p["text"] for t in conv["threads"][-12:] for p in t] + list(conv["said"])  # what has been said, here and by itself
    if not turn.stopped and not declined(answer) and repeats(answer, heard):
        # it said again what was said (a small model echoes): its node tells it so, once, and it answers again
        turn = s.converse(system, msgs + [{"role": "assistant", "content": answer}, {"role": "user", "content": talk.MIND_REPEAT.format(me=me)}],
                          tools=[], max_new=300, max_calls=0)
        out.update(first=answer, answer=turn.answer.strip(), seconds=out["seconds"] + turn.seconds, drawn=s.drawn - d0)
        answer = out["answer"]
        if not turn.stopped and not declined(answer) and repeats(answer, heard):
            return {**out, "why": "it said again what had been said"}
    if turn.stopped:
        return {**out, "why": f"cut off ({turn.stopped})"}
    if declined(answer):
        return {**out, "why": "declined"}
    # ---- it may back what it answered, as its node offers this time: one offer, one call
    if act:
        if not parent or act not in OFFERS or (act == "stake" and parent["standing"] == "slashed"):
            act = "draw"  # the giving and the locking are for the node it answered (and nothing is locked on a convicted one)
        out["offered"] = act
        offer = talk.MIND_DRAW if act == "draw" else (talk.MIND_TIP if act == "tip" else talk.MIND_STAKE).format(other=parent["from"])
        try:
            did = s.converse(system, msgs + [{"role": "assistant", "content": answer}, {"role": "user", "content": offer}],
                             tools=OFFERS[act], max_new=900 if act == "draw" else 500, prose=False,  # whether it calls is not drawn
                             max_calls=2 if act == "draw" else 1)  # a picture that is not well-formed may be drawn again, once
            out.update(calls=list(did.calls), seconds=out["seconds"] + did.seconds, drawn=s.drawn - d0, after=did.answer.strip()[:200])
        except Exception as e:  # noqa: BLE001  (what it answered is published all the same)
            out["after"] = f"{type(e).__name__}: {str(e)[:120]}"
    t_pub = time.time()
    try:
        out["posted"] = node.publish(answer, re=parent["id"] if parent else None, att=[m["id"] for m in node.made][:4])
    except Refused as e:
        out["why"] = f"refused: {e}"
    out["took"]["publishing"] = round(time.time() - t_pub)
    return out


def mind_policy(node_ref: list):
    def approve(name, what, sats):
        if name in MIND_PAID and MIND_STEP != "inf" and (sats or 0) > int(MIND_STEP):
            return False
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


def tooling(web: bool = True) -> str:
    """A fingerprint of the files the miners, agents, ord servers and node models run from (a
    process keeps the code it was started with; this tells whether that is still the code here).
    web: the pages are among what runs (tools/web.py then counts; where no pages are served, a
    change to them restarts nothing). tools/talk.py never counts: a running loop reads it again."""
    import zlib
    c = 0
    for f in ([ROOT / "tools" / "net.py", ROOT / "tools" / "chain.py"] + ([ROOT / "tools" / "web.py"] if web else [])
              + sorted((ROOT / "graph").glob("*.py")) + sorted((ROOT / "rails").glob("*.py"))):
        if not f.exists():
            continue
        c = zlib.crc32(f.read_bytes(), c)
    return f"{c:08x}"


def stale() -> bool:
    """Helpers are running, and from other files than the ones here now."""
    running = any(alive(f.stem) for f in PIDS.glob("*.pid")) if PIDS.exists() else False
    stamp = (STAMP.read_text().split() if STAMP.exists() else []) or [""]
    return running and stamp[0] != tooling(stamp[1] == "web" if len(stamp) > 1 else True)  # (a stamp without the word: an earlier version's)


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
    net.refresh_seeds()  # an installed copy: the project's public nodes as they are now (a rented one moves)
    if not any(alive(f.stem) for f in PIDS.glob("*.pid")):  # every helper below starts from the files as they are now
        web = os.environ.get("QWEN_WEB_PORT", "0") not in ("", "0")
        STAMP.write_text(f"{tooling(web)} {'web' if web else 'noweb'}\n")
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
    flushed, lonely, reseeded = time.time(), None, 0.0
    while not STOP.exists():
        try:
            if hold or time.time() - looked > 5:
                looked = time.time()
                info, peers = net.cli(i, "getblockchaininfo"), net.cli(i, "getpeerinfo")
                lonely = None if peers else (lonely or time.time())
                if lonely and time.time() - lonely > 180 and time.time() - reseeded > 300:
                    reseeded = time.time()  # alone for minutes: the public node may have moved, the project knows where to
                    for s in net.refresh_seeds():
                        net.cli(i, "addnode", s, "onetry", check=False)
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


# mined outputs a wallet may hold before they are put together; how many go into one transaction; how many of the newest stay
# as they are (what is put together cannot be spent until its transaction is in a block, and ord's wallet inscribes from confirmed coins only)
TIDY_ABOVE, TIDY_AT_ONCE, TIDY_LEAVE = 30, 400, 8


def tidy(i, to: str) -> dict | None:
    """Put the node's large outputs together into one, at address `to` of its own wallet, when there are more than
    TIDY_ABOVE of them. A node that mines gets one output a block, and ord's wallet looks at every output it holds
    for every command (a balance took 8 seconds with 90 outputs and about a minute with 1,500): everything that
    inscribes waited for it. Only outputs of a coin or more are touched: mined coins and change, never the small
    outputs that carry inscriptions; the newest TIDY_LEAVE stay, so that the wallet always has confirmed coins. The caller holds the node's lock; ord's wallet is locked here. Returns {"merged", "txid"}."""
    big = [u for u in net.cli(i, "listunspent", 1, wallet="ord", timeout=120) if u["amount"] >= 1 and u.get("spendable", True) and u.get("safe", True)]
    if len(big) <= TIDY_ABOVE:
        return None
    take = sorted(big, key=lambda u: -u["confirmations"])[:-TIDY_LEAVE][:TIDY_AT_ONCE]  # the oldest first; the newest stay
    with net.wallet_lock(i):
        r = net.cli(i, "sendall", json.dumps([to]), "null", "unset", 2, json.dumps({"inputs": [{"txid": u["txid"], "vout": u["vout"]} for u in take]}),
                    wallet="ord", timeout=300)
    return {"merged": len(take), "of": len(big), "coins": sum(u["amount"] for u in take), "txid": r.get("txid") if isinstance(r, dict) else r}


def tidy_small(i, own: str, to: str) -> dict | None:
    """The other half of what a wallet gathers: the small outputs under what the node inscribed before its messages
    went onto its identity's key (a post, a head or an attestation each, 10,000 sats apiece). When there are more
    than TIDY_ABOVE they go, in one transaction, onto that key (address `own`), each inscription on its own sat as
    before: where such a message sits means nothing to the graph, and to ord's wallet each was one more output to
    look at. Only outputs of 10,000 sats or less are taken; one large output pays the fee and comes back to `to`,
    so that no inscribed sat is spent as a fee (the transaction is built by hand for that: the small outputs first,
    their sum as the first output). Returns {"moved", "txid"}."""
    outs = net.cli(i, "listunspent", 1, wallet="ord", timeout=180)
    small = [u for u in outs if u["amount"] <= 0.0001 and u.get("spendable", True)][:TIDY_AT_ONCE]
    big = max((u for u in outs if u["amount"] >= 1 and u.get("spendable", True)), key=lambda u: u["amount"], default=None)
    if len(small) <= TIDY_ABOVE or big is None:
        return None
    sats = lambda u: round(u["amount"] * 1e8)  # noqa: E731
    total = sum(sats(u) for u in small)
    fee = 2 * (11 + 58 * (len(small) + 1) + 43 * 2)  # 2 sat/vB: taproot key spends in, two taproot outputs out
    ins = [{"txid": u["txid"], "vout": u["vout"]} for u in small + [big]]
    outs_ = [{own: f"{total / 1e8:.8f}"}, {to: f"{(sats(big) - fee) / 1e8:.8f}"}]
    with net.wallet_lock(i):
        raw = net.cli(i, "createrawtransaction", json.dumps(ins), json.dumps(outs_), wallet="ord")
        signed = net.cli(i, "signrawtransactionwithwallet", raw, wallet="ord", timeout=300)
        if not signed.get("complete"):
            raise RuntimeError("the wallet could not sign for all of its small outputs")
        txid = net.cli(i, "sendrawtransaction", signed["hex"], timeout=300)
    return {"moved": len(small), "sats": total, "txid": txid}


def agent(i):
    from graph import protocol as P
    from graph.identity import Identity

    ident = Identity(i)
    with ident.use() as (a, st):
        name, pid = a.name, a.pid.hex()

    def say(*x):
        print(time.strftime("%H:%M:%S"), name, *x, flush=True)

    say("identity", pid)
    last, swept, tidied = 0.0, 0.0, 0.0
    while not STOP.exists():
        try:
            h = net.cli(i, "getblockcount")
            net.ord_wait(i, h, timeout=120)
            # the node's lock: the model driving this node (rails/) spends from the same wallet and signs as the same
            # identity. It is taken once for each thing done and given back in between (what the identity has signed
            # is on disk when it is given back), so that the node's model gets its turn between two of them.
            with ident.use() as (a, st):
                mine_ = a.stake("id")
                if mine_ is None:
                    if st["bond_sent"] is not None and h - st["bond_sent"] < 30:
                        pass  # the bond is on its way into a block
                    elif net.ord_wallet(i, "balance").get("cardinal", 0) >= BOND + 1_000_000:
                        b = a.bond(BOND, DELAY)
                        st["bond_sent"] = h
                        say(f"bonded {BOND:,d} sat at {b.address()} (height {h})")
                else:
                    ep = P.epoch_of(h)
                    if ep > st["epoch"] and a.signed(ep):  # an earlier process signed this epoch's head and was stopped before it noted so
                        st["epoch"] = ep
                    elif ep > st["epoch"] and time.time() - last > ROUND_GAP:
                        x = bytes.fromhex(net.cli(i, "getblockheader", net.cli(i, "getbestblockhash"), "false"))
                        a.attest(x)
                        a.head(ep)
                        st["epoch"], last = ep, time.time()
                        say(f"epoch {ep}: attested the hash of the header at height {h}, signed head {a.root().hex()[:16]}")
            with ident.use() as (a, st):
                if a.stake("id") is not None:
                    for kind, off, iid, txid in a.audit():
                        say(f"SLASHED {off.hex()[:16]}: {kind} proof, stake {iid[:16]} burned in {txid}")
            if time.time() - tidied > 240 and addrs().get(i):  # the mined coins, an output a block, put together (see tidy)
                tidied = time.time()
                with ident.use() as (a, st):
                    done = tidy(i, addrs()[i])
                    if done:
                        say(f"put {done['merged']} of the wallet's {done['of']} large outputs together ({done['coins']:,.0f} coins, in {done['txid']})")
                    else:  # the large ones are few: now the small ones, under what it inscribed before
                        from graph import covenant as GC
                        done = tidy_small(i, GC.tip_address(a.own), addrs()[i])
                        if done:
                            say(f"moved {done['moved']} small outputs of the wallet, with the inscriptions on them, onto the identity's own key ({done['sats']:,d} sats, in {done['txid']})")
            if time.time() - swept > 300:  # what its identity was given (tips) goes into the node's wallet
                swept = time.time()
                with ident.use() as (a, st):
                    for iid, sats, txid in a.collect(addrs()[i]):
                        say(f"collected a tip: {sats:,d} sats into the wallet ({iid[:16]}.., in {txid})")
        except Exception as e:
            say("retry:", str(e).strip().splitlines()[-1][:200])
        time.sleep(15)


def mind(i):
    """The node's model, driving the node. Once per round, from a fresh context, it takes its turn in one of the
    graph's conversations (mind_round): the other nodes' posts are its input, its answer is published into the
    thread; what it calls and what it writes are its own (its wording drawn at MIND_TEMP, its calls the likeliest
    ones, nothing edited), and it may decline. A node that has been written to takes its turn sooner. It spends
    only as MIND_FREE / MIND_PAID allow. The duties that have to be exact (the bond, attestations, one head per
    epoch, the audit) stay with the agent."""
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

    say(f"loading {MIND_MODEL.name} ({torch.get_num_threads()} threads); spending limit: {MIND_LIMIT}; temperature of its wording: {talk.MIND_TEMP:g}")
    s = Session(MIND_MODEL, node, temperature=talk.MIND_TEMP)
    say("loaded; the functions in its template:", json.loads((MIND_MODEL / "hashformer.json").read_text()).get("functions", "?"))
    try:  # how long this node's wallet takes to answer: a wallet with a thousand outputs is what a turn waits for
        t0 = time.time()
        outs = len(net.cli(i, "listunspent", 0, wallet="ord", timeout=120))
        t1 = time.time()
        net.ord_wallet(i, "balance")
        say(f"its wallet has {outs} outputs: the node lists them in {t1 - t0:.1f}s, `ord wallet balance` takes {time.time() - t1:.0f}s")
    except Exception as e:  # noqa: BLE001
        say("its wallet could not be timed:", str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__)
    index_said = False
    time.sleep(random.uniform(0, 20) * (i - 1))  # the minds of one machine do not all speak at once
    while not STOP.exists():
        try:
            h = net.cli(i, "getblockcount")
            net.ord_wait(i, h, timeout=300)
            if node.coins() < 50_000:
                say(f"height {h}: no spendable coins yet (a node's first coins mature 100 blocks after it mines them)")
                time.sleep(30)
                continue
            me = (identity(i) or {}).get("name", f"node {i}")
            if talk_again(say):  # its words and its pacing, if the file has changed (the model stays loaded)
                s.temperature = talk.MIND_TEMP
            pend = [p for p in st.get("pending", []) if h - p[1] <= 30]  # a post that is in no block after 30 is taken for lost
            offer = talk.MIND_OFFERS[st.get("offers", 0) % len(talk.MIND_OFFERS)] if talk.MIND_ACT > 0 and st.get("since_act", 0) + 1 >= talk.MIND_ACT else False
            # what it is asked if it opens a conversation: not what it was asked the last time
            angle = random.choice([a for a in talk.MIND_ANGLES if a != st.get("angle")])
            r = mind_round(s, node, me, h, st.get("passed", []), address=addrs().get(i, ""), act=offer, pending=[p[0] for p in pend], angle=angle,
                           own=talk.MIND_OWN > 0 and st.get("since_own", 0) >= talk.MIND_OWN)
            if r["kind"] == "open":
                st["angle"] = angle
            if r["posted"]:
                st["since_own"] = 0 if r["kind"] == "open" else st.get("since_own", 0) + 1
            calls, spoke = r["calls"], bool(r["posted"])
            st["pending"] = [p for p in pend if p[0] in r["unseen"]] + ([[r["posted"]["id"], h]] if spoke else [])
            if spoke:
                st["since_act"] = 0 if r["offered"] else st.get("since_act", 0) + 1
                st["offers"] = st.get("offers", 0) + bool(r["offered"])
            if not index_said:  # once per start: what the graph index took up of what an earlier process had read
                index_said = True
                try:
                    ix = node._identity().agent.index
                    say(f"its graph index: {len(ix.seen)} inscriptions, {ix.taken_up} of them taken up from what was kept, {len(ix.seen) - ix.taken_up} read now")
                except Exception:  # noqa: BLE001
                    pass
            if r["kind"] != "wait":
                st.update(rounds=st["rounds"] + 1, posts=st["posts"] + spoke, calls=st["calls"] + len(calls), at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if spoke:
                st["said"] = r["answer"][:200]
            elif r["parent"]:  # declined, cut off or refused: that post is not asked about again
                st["passed"] = (st.get("passed", []) + [r["parent"]["id"]])[-200:]
            book.write_text(json.dumps(st))
            to = f" {r['parent']['from']}'s post {r['parent']['number']}" if r["parent"] else ""
            if r["kind"] == "wait":
                say(f"height {h}: no turn ({r['why']})")
            else:
                took = r.get("took", {})  # where the turn's time went: reading the graph, the model, the inscription of the post
                say(f"round {st['rounds']} at height {h} ({r['seconds']:.0f}s the model, {took.get('reading', 0)}s reading the graph"
                    + (f", {took['publishing']}s publishing" if "publishing" in took else "") + f"; {r['kind']}{to}; {r['drawn']} tokens drawn"
                    + (f"; offered: {r['offered']}" if r["offered"] else "") + "): "
                    + ("; ".join(f"{n}({', '.join(f'{k}={v[:120]!r}' for k, v in a.items())}) -> {x[:90]}" for n, a, x, by in calls) or "no call"))
                if r.get("angle"):
                    say(f"   asked: {r['angle']}")
                if r["offered"] and not calls:  # it was offered something and called nothing: what it wrote instead
                    say(f"   to the offer it wrote: {r.get('after', '')[:160]!r}")
                say(f"   {'published' if spoke else 'not published (' + r['why'] + ')'}: {r['answer'][:400]!r}"
                    + (f" with {len(r['posted'].get('att', []))} inscription(s) attached" if spoke and r["posted"].get("att") else ""))
        except Exception as e:
            spoke, r = False, {"kind": "error"}
            say("retry:", str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__)
        gap = talk.MIND_GAP if st["posts"] else min(talk.MIND_GAP, 90)  # until it has spoken once, it tries again soon
        if r["kind"] == "wait":
            gap = min(gap, 60 if r.get("unseen") else 120)  # nothing was computed: it looks again soon
        elif r["kind"] != "error" and not spoke:
            gap = min(gap, 120)  # it declined, or what it wrote was not published: another conversation may be waiting
        t_round, t_next = time.time(), time.time() + gap * random.uniform(0.8, 1.2)
        while time.time() < t_next and not STOP.exists():
            for _ in range(6):
                if STOP.exists():
                    break
                time.sleep(5)
            # written to in the meantime: its turn comes sooner
            if time.time() - t_round >= talk.MIND_QUICK:
                try:
                    if mind_pick(node.threads(), node.height_of, net.cli(i, "getblockcount"), st.get("passed", []))[0] == "answer":
                        break
                except Exception:  # noqa: BLE001  (the node or its ord is busy: the turn comes at its time)
                    pass


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
