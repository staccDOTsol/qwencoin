"""The model's tools on a live chain (rails/): what the checkpoint declares, what the node does
for it, and the merged model using them.

    python3 tests/check_tools.py [node|tips|template|model|mind|all] [MODEL_DIR=dist/qwen08-sha]     -> logs/tools.log
    python3 tests/check_tools.py live MODEL_DIR       the model part on the running chain (run/), as node 1

On a private two-node network of the qwencoin chain (tests/tools_net.py; real blocks, real ord),
or, with `live`, on the chain that tools/chain.py is running (what the model does there is real:
its payment goes to node 2, its inscription and its post stay on the chain):

  node      no model. Every function through Node.call, the way the runtime calls it, checked
            against the chain: coins move, an inscription is written, read, listed and given
            away, identities bond, post and vouch, a lie is audited and its stake and the vouch
            on it burn, a stake starts to leave. Then what must be refused: unapproved spending,
            bad addresses and amounts, key export, stopping the node, unknown functions.
  tips      the part of `node` about coins given for a post, by itself (four blocks): the tip is
            one inscription that is also the payment, the receiver's identity collects it into
            its wallet; a note without its coins and a copy of a paid note do not count.
  template  the checkpoint's chat template: the functions are declared when the caller declares
            none; a caller's own list renders byte for byte as the stock template renders it.
  model     the merged checkpoint asked in plain words, greedy, one question per fresh context.
            Hard requirements: a hash the weights print is hashlib's; every call the model made
            reached the node unedited and its answer reached the model unedited. What the model
            chose to call, and with what arguments, is reported as it is: that part is the
            language model's skill, not this code's.
"""
import hashlib, json, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
what = sys.argv[1] if len(sys.argv) > 1 else "all"
LIVE = what == "live"  # the model part on the running chain in run/ (tools/chain.py), as node 1
if LIVE:
    import chain as _chain  # noqa: E402
    import net  # noqa: E402

    class T:  # the running chain: nothing to start, and a block is waited for, not mined
        @staticmethod
        def up(height, log=print):
            assert net.cli(1, "getblockcount") > 100 and len(_chain.addrs()) >= 2, "the chain in run/ is not up (tools/chain.py up)"

        addrs = staticmethod(_chain.addrs)

        @staticmethod
        def block(i=1, n=1):
            h = net.cli(1, "getblockcount")
            for _ in range(3600):
                if net.cli(1, "getblockcount") >= h + n:
                    break
                time.sleep(1)
            for j in (1, 2):
                net.ord_wait(j, timeout=900)
            return net.cli(1, "getblockcount")
else:
    import tools_net as T  # noqa: E402  (sets the private network's environment before net is imported)
    import net  # noqa: E402
from rails.tools import NAMES, TOOLS, Node  # noqa: E402

MODEL = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "dist" / "qwen08-sha")
LOG = open(ROOT / "logs" / "tools.log", "a")
GENESIS = "51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5"
GENESIS_WORK = "0083bfd7a2ae5796c139f180b33630684eb71751e4b2b3153ffe288da7739d1d"


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def call(node, name, _may_fail=False, **args):
    """As the runtime does it: text arguments in, JSON text out."""
    r = json.loads(node.call(name, {k: str(v) for k, v in args.items()}))
    assert _may_fail or not (isinstance(r, dict) and "error" in r), (name, args, r)
    return r


def err(node, name, **args) -> str:
    r = call(node, name, _may_fail=True, **args)
    assert isinstance(r, dict) and "error" in r, (name, args, r)
    return r["error"]


# ====================================================================== the node side
def check_node():
    asked = []

    def approve(name, what_, sats):
        asked.append((what_, sats))
        return True

    n1, n2 = Node(1, approve=approve), Node(2)
    h0 = net.cli(1, "getblockcount")
    say(f"node: two nodes at height {h0}")

    # --- the chain
    st = call(n1, "chain_status")
    assert st["height"] == h0 and st["tip"] == net.cli(1, "getbestblockhash") and st["peers"] >= 1, st
    g = call(n1, "chain_block", block=0)
    assert g["hash"] == GENESIS and g["work"] == GENESIS_WORK, g
    tip = call(n1, "chain_block", block=st["tip"])
    assert tip["height"] == h0 and int(tip["work"], 16) <= 0xFFFF << 232
    say(f"  chain_status / chain_block: height {st['height']}, {st['passes_per_block']} expected passes per block, genesis work {g['work'][:16]}..")

    # --- the wallet
    b1 = call(n1, "wallet_balance")
    assert b1["spendable_sats"] >= 50 * 10**8, b1
    a2 = call(n2, "wallet_address")
    before2 = call(n2, "wallet_balance")["spendable_sats"]
    pay = call(n1, "wallet_send", address=a2, sats="5,000,000")
    assert asked[-1] == (f"pay 5,000,000 sats to {a2}", 5_000_000)
    assert call(n1, "chain_tx", txid=pay["txid"])["confirmations"] == 0
    T.block(1)
    tx = call(n2, "chain_tx", txid=pay["txid"])
    assert tx["confirmations"] == 1 and {"sats": 5_000_000, "address": a2} in tx["outputs"], tx
    # n2's spendable coins grew by the payment, and by 50 coins for each of its own blocks that matured meanwhile
    assert (call(n2, "wallet_balance")["spendable_sats"] - before2 - 5_000_000) % (50 * 10**8) == 0
    hist = call(n1, "wallet_history", count=5)
    assert any(t["txid"] == pay["txid"] for t in hist), hist
    say(f"  wallet_balance / wallet_address / wallet_send / chain_tx / wallet_history: 5,000,000 sats from n1 to n2 in {pay['txid'][:16]}.., confirmed, n2's balance up by exactly that (beside its own maturing blocks)")

    # --- ord
    text = "qwencoin: the first message this test wrote, " + time.strftime("%Y-%m-%d %H:%M:%S")
    ins = call(n1, "ord_inscribe", text=text)
    T.block(1)
    rd = call(n2, "ord_read", id=ins["id"])  # read through the other node's ord
    mine = lambda i, addr: net.cli(i, "getaddressinfo", addr, wallet="ord")["ismine"]  # noqa: E731
    assert rd["text"] == text and mine(1, rd["owner"]) and not mine(2, rd["owner"]) and rd["type"].startswith("text/plain"), rd
    assert ins["id"] in [x["id"] for x in call(n1, "ord_list")]
    latest = call(n2, "ord_latest", count=3)
    assert latest[0]["id"] == ins["id"] and latest[0]["text"] == text[:240], latest
    gift = call(n1, "ord_send", id=ins["id"], address=a2)
    T.block(1)
    assert call(n1, "ord_read", id=ins["id"])["owner"] == a2
    assert ins["id"] in [x["id"] for x in call(n2, "ord_list")] and ins["id"] not in [x["id"] for x in call(n1, "ord_list")]
    say(f"  ord_inscribe / ord_read / ord_list / ord_latest / ord_send: inscription {ins['id'][:16]}.. written by n1, read by n2's ord, "
        f"newest on the chain, given to n2 in {gift['txid'][:16]}..")

    # --- the graph. The test network is kept between runs: each run starts two new identities
    # (the old records are renamed, their stakes stay where the chain has them).
    for i in (1, 2):
        f = net.datadir(i) / "identity.json"
        if f.exists():
            f.rename(f.with_name(f"identity-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    me = call(n1, "graph_me")
    assert me["own_bond_sats"] == 0 and me["standing"] == "no stake", me
    bond = call(n1, "graph_bond", sats=200000)
    assert asked[-1][1] == 200_000 and bond["identity"] == me["identity"]
    call(n2, "graph_bond", sats=150000)
    T.block(1)
    me, other = call(n1, "graph_me"), call(n2, "graph_me")
    assert me["own_bond_sats"] == 200_000 and me["standing"] == "staked" and other["own_bond_sats"] == 150_000, (me, other)
    assert "already bonded" in err(n1, "graph_bond", sats=200000)
    post = "hello from node 2, " + time.strftime("%H:%M:%S")
    call(n2, "graph_post", text=post)
    T.block(1)
    feed = call(n1, "graph_feed", count=3)  # in the chain's order: the newest message is the last
    assert feed[-1] == {"from": other["name"], "at_stake_sats": 150_000, "standing": "staked", "text": post}, feed
    mine = "node 1 answers, " + time.strftime("%H:%M:%S")
    call(n1, "graph_post", text=mine)
    T.block(1)
    feed = call(n1, "graph_feed", count=2)  # the reader's own message is marked as its own; node 2 reads the same one under node 1's name
    assert [x["from"] for x in feed] == [other["name"], f"you ({me['name']})"] and feed[-1]["text"] == mine, feed
    assert call(n2, "graph_feed", count=1)[0]["from"] == me["name"]
    n1.said = [mine]  # as the node's own loop sets it: what this identity has published is not published again
    assert "already published" in err(n1, "graph_post", text="  NODE 1 answers,  " + mine.split(", ")[1] + " ")
    n1.said = None

    # --- conversations: an answer is signed together with the post it answers, and a post can carry an inscription
    from graph import covenant as GC, protocol as GP
    sec, pid = GC.keypair(b"check_tools" * 3)
    ref = "ab" * 32 + "i0"
    for kw in ({}, {"re": ref}, {"att": [ref]}, {"re": ref, "att": [ref, "cd" * 32 + "i1"]}):
        m = GP.post_msg(sec, pid, 7, "hi", **kw)
        assert GP.parse(json.dumps(m).encode()) == m and GP.parse(json.dumps({**m, "text": "ho"}).encode()) is None, kw
    plain, reply = GP.post_msg(sec, pid, 7, "hi"), GP.post_msg(sec, pid, 7, "hi", re=ref)
    for forged in ({**plain, "re": ref}, {k: v for k, v in reply.items() if k != "re"}, {**reply, "att": [ref]}, {**reply, "re": "cd" * 32 + "i0"},
                   {**plain, "att": []}, {**plain, "re": "not an id"}):
        assert GP.parse(json.dumps(forged).encode()) is None, forged  # a signature for one form or one reference is none for another
    first = next(p for t in n1.threads()["threads"] for p in t if p["text"] == post)  # node 2's post, as node 1's index has it
    assert not first["mine"] and first["re"] is None and first["from"] == other["name"]
    assert "xmlns" in err(n1, "ord_inscribe", text='<svg viewBox="0 0 3 2"><rect width="3" height="2"/></svg>')  # not a picture a browser shows
    assert "well-formed" in err(n1, "ord_inscribe", text='<svg xmlns="http://www.w3.org/2000/svg"><rect></svg>')
    n1.made = []
    flag = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20"><rect width="30" height="20" fill="#135"/><circle cx="15" cy="10" r="6" fill="#fc3"/></svg>'
    assert "not a picture" in err(n1, "draw", svg="a flag: dark blue, a yellow sun in the middle")  # draw is what a model has for a picture
    assert "well-formed" in err(n1, "draw", svg=flag[:-6]) and n1.made == []
    pic = call(n1, "draw", svg=flag)
    assert pic["type"] == "image/svg+xml" and [m["id"] for m in n1.made] == [pic["id"]] and asked[-1][0].endswith("bytes of a picture"), (pic, asked[-1])
    assert "one picture" in err(n1, "draw", svg=flag)  # an answer carries the one it drew
    n1.made = None
    words = "node 1 answers node 2 and draws it a flag, " + time.strftime("%H:%M:%S")
    ans = n1.publish(words, re=first["id"], att=[pic["id"]])
    assert ans["re"] == first["id"] and ans["att"] == [pic["id"]]
    T.block(1)
    conv = n2.threads()
    theirs = next(t for t in conv["threads"] if t[0]["id"] == first["id"])  # node 2 reads the conversation its post started
    assert [(p["text"], p["re"], p["mine"]) for p in theirs] == [(post, None, True), (words, first["id"], False)], theirs
    assert theirs[1]["to_me"] and theirs[1]["att"] == [pic["id"]] and theirs[1]["from"] == me["name"]
    seen = n2.attachment(pic["id"])
    assert seen["type"] == "image/svg+xml" and seen["text"].startswith("<svg xmlns") and seen["bytes"] > 100, seen
    content = json.loads(n2._content(ans["id"], 4000))
    assert content["re"] == first["id"] and content["att"] == [pic["id"]] and GP.parse(json.dumps(content).encode()) == content
    feed = call(n2, "graph_feed", count=1)
    assert feed[0]["from"] == me["name"] and feed[0]["answers"] == f"you ({other['name']})", feed
    stray = n1.publish("an answer to something that is not a post", re=pic["id"])  # it stands as the start of a conversation
    T.block(1)
    assert next(t for t in n2.threads()["threads"] if t[0]["id"] == stray["id"])[0]["re"] is None
    say(f"  conversations: {me['name']} answers {other['name']}'s post with a picture it drew (draw: SVG, {seen['bytes']} bytes, inscribed and attached); "
        f"{other['name']}'s index has the answer under its post, written to it, the picture readable; the reference and the attachment are part of what is signed "
        "(a signature for a plain post, or for another reference, is refused); an answer to something that is not a post starts a conversation")
    check_tips(n1, n2, asked)
    v = call(n1, "graph_vouch", identity=other["name"], sats=120000)
    assert v["for"] == other["identity"]
    T.block(1)
    who = call(n1, "graph_who", identity=other["name"])
    assert who["at_stake_sats"] == 270_000 and who["vouched_by"] == [{"from": me["name"], "sats": 120_000}] and who["trust_from_me_sats"] == 120_000, who
    aud = call(n1, "graph_audit")
    assert other["identity"] not in aud["convicted"] and me["identity"] not in aud["convicted"], aud
    assert not [b for b in aud["burned"] if b["identity"] in (me["identity"], other["identity"])], aud
    say(f"  graph_me / graph_bond / graph_post / graph_feed / graph_vouch / graph_who / graph_audit: {me['name']} bonds 200,000, "
        f"{other['name']} bonds 150,000 and posts, {me['name']} reads it and vouches 120,000 for it: 270,000 at stake on {other['name']}; an audit finds nothing")

    # --- a lie, audited: n2 signs SHA256(x) = y for a y that is not it
    from graph.identity import Identity
    with Identity(2).use() as (liar, _):
        liar.attest(b"qwencoin", bytes(32))
    T.block(1)
    aud = call(n1, "graph_audit")
    assert other["identity"] in aud["convicted"] and me["identity"] not in aud["convicted"], aud
    assert [(b["identity"], b["proof"]) for b in aud["burned"]] == [(other["identity"], "lie")] * 2, aud
    T.block(1)
    who = call(n1, "graph_who", identity=other["identity"])
    assert who["standing"] == "slashed" and who["at_stake_sats"] == 0 and who["trust_from_me_sats"] == 0, who
    assert call(n1, "graph_feed", count=1)[0]["standing"] == "slashed"
    # --- coins locked on another name, and taken back: node 2 vouches for node 1, then starts to take it back
    lock = call(n2, "graph_vouch", identity=me["name"], sats=110000)
    assert lock["for"] == me["identity"]
    T.block(1)
    ev = [e for e in n1.threads()["events"] if e["on_me"] and e["from"] == other["name"]]
    assert ev and ev[-1]["sats"] == 110_000 and ev[-1]["status"] == "active", ev
    assert n1.threads()["stand"]["locked_on_me"] == [{"from": other["name"], "sats": 110_000}]
    back = call(n2, "graph_unvouch", identity=me["name"])
    assert "unlocking" in back and back["collect_after_blocks"] == 144, back
    T.block(1)
    assert [e["status"] for e in n2.threads()["events"] if e["by_me"] and e["to"] == me["name"]][-1] == "unbonding"
    assert "no sats locked" in err(n1, "graph_unvouch", identity=other["name"])  # node 1's vouch on node 2 burned with node 2's bond
    say(f"  graph_unvouch: {other['name']} locks 110,000 sats on {me['name']} and starts taking them back; {me['name']}'s loop sees both on the chain")
    leave = call(n1, "graph_leave")
    assert "unbonding" in leave, leave
    say(f"  a false hash signed by {other['name']}: graph_audit on n1 convicts it and burns both stakes on it (its bond and n1's vouch) in "
        f"{aud['burned'][0]['txid'][:16]}.. and {aud['burned'][1]['txid'][:16]}..; it now reads as slashed with 0 at stake. graph_leave starts n1's unbonding")
    T.block(1)

    # --- everything else, and what is refused
    assert call(n1, "bitcoin_cli", args="getblockcount") == net.cli(1, "getblockcount")
    assert "mine" in call(n1, "bitcoin_cli", args="getbalances")
    outs = call(n1, "ord_cli", args="wallet outputs")  # a long list comes back cut, and still JSON
    assert isinstance(outs, list) or (len(outs["first"]) < outs["of"] and "output" in outs["first"][0]), outs
    assert len(n1.call("ord_cli", {"args": "wallet outputs"})) <= 2400
    assert call(n1, "sha256", hex="00" * 40) == hashlib.sha256(bytes(40)).hexdigest()
    n_asked = len(asked)
    refused = {
        "stop the node": err(n1, "bitcoin_cli", args="stop"),
        "export a key": err(n1, "bitcoin_cli", args="dumpprivkey bc1qxyz"),
        "export the descriptors' keys": err(n1, "bitcoin_cli", args="listdescriptors true"),
        "dump ord's wallet": err(n1, "ord_cli", args="wallet dump"),
        "rewrite the chain's view": err(n1, "bitcoin_cli", args="invalidateblock " + GENESIS),
        "options before the command": err(n1, "bitcoin_cli", args="-rpcwallet=x getbalances"),
        "an unknown function": err(n1, "wallet_drain"),
        "an unknown parameter": err(n1, "wallet_send", address=a2, sats=1000, memo="x"),
        "a missing parameter": err(n1, "wallet_send", address=a2),
        "an amount that is not a number": err(n1, "wallet_send", address=a2, sats="a lot"),
        "an address of another chain": err(n1, "wallet_send", address="1BoatSLRHtKNngkdXEeobR76b53LETtpyT_", sats=1000),
        "a stake below the minimum": err(n1, "graph_bond", sats=10),
    }
    assert len(asked) == n_asked, "a refused call reached the approval step"
    strict = Node(1, approve=lambda n, w, s: False)
    refused["spending nobody approved"] = err(strict, "wallet_send", address=a2, sats=1000)
    refused["an inscription nobody approved"] = err(strict, "ord_inscribe", text="x")
    spent = n1.spent
    refused["more coins than there are"] = err(n1, "wallet_send", address=a2, sats=10**15)
    assert "nsufficient funds" in refused["more coins than there are"] and n1.spent == spent  # a failed send is not counted
    assert net.cli(1, "getblockcount") > h0 and strict.spent == 0
    for k, m in refused.items():
        say(f"    refused, {k}: {m[:110]}")
    say(f"  bitcoin_cli / ord_cli / sha256 (the node's executor, 40 bytes): pass-throughs work; {len(refused)} kinds of call refused; "
        f"{len(asked)} spending calls went through approval, {n1.spent:,d} sats sent or locked by n1")
    used = {"chain_status", "chain_block", "chain_tx", "wallet_balance", "wallet_address", "wallet_send", "wallet_history", "ord_inscribe",
            "ord_read", "ord_list", "ord_send", "ord_latest", "graph_me", "graph_bond", "graph_vouch", "graph_post", "graph_feed",
            "graph_who", "graph_audit", "graph_leave", "bitcoin_cli", "ord_cli", "sha256"}
    assert used == set(NAMES), set(NAMES) ^ used
    say(f"  all {len(NAMES)} declared functions exercised against the chain")


# ====================================================================== coins given for what was written
def check_tips(n1, n2, asked=None):
    """graph_tip on the chain (graph/protocol.py tip): node 1 gives node 2 coins for a post of its. The note and the
    payment are one inscription; node 2's identity collects the coins into its wallet. Beside it, two notes that must
    not count: one that says so without the coins, and a copy of the paid note, paid again by the receiver itself.
    By itself (`tips`) it first makes what it needs: two bonded identities and a post of node 2."""
    from graph import covenant as GC, protocol as GP
    from graph.identity import Identity

    def note(i, msg, dest=None, postage=None):  # a message inscribed as it is, by node i's wallet (not through its identity)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(msg, fh, separators=(",", ":"))
        try:
            r = net.ord_wallet(i, "inscribe", "--fee-rate", "2", "--no-backup", "--file", fh.name,
                               *(["--destination", dest] if dest else []), *(["--postage", f"{postage}sat"] if postage else []))
        finally:
            Path(fh.name).unlink()
        return r["inscriptions"][0]["id"]

    theirs = lambda: next((p for t in reversed(n1.threads()["threads"]) for p in reversed(t) if p["key"] == other["identity"]), None)  # noqa: E731
    me, other = call(n1, "graph_me"), call(n2, "graph_me")
    todo = [call(n, "graph_bond", sats=150000) for n, w in ((n1, me), (n2, other)) if w["own_bond_sats"] == 0]
    if todo:
        T.block(1)
        me, other = call(n1, "graph_me"), call(n2, "graph_me")
    if theirs() is None:
        n2.publish("a post to be given coins for, " + time.strftime("%H:%M:%S"))
        T.block(1)
    first = theirs()

    # --- the note: signed by the giver over the receiver, the sats, a nonce and the post
    sec, pid = GC.keypair(b"check_tools" * 3)
    to, ref = bytes.fromhex(other["identity"]), "ab" * 32 + "i0"
    for kw in ({}, {"re": ref}):
        m = GP.tip_msg(sec, pid, to, 50_000, bytes(8), **kw)
        assert GP.parse(json.dumps(m).encode()) == m, kw
        for forged in ({**m, "sats": 60_000}, {**m, "to": pid.hex()}, {**m, "n": "01" * 8}, {**m, "re": "cd" * 32 + "i0"}, {**m, "id": to.hex()},
                       {k: v for k, v in m.items() if k != "re"} if kw else {**m, "sats": 50_000.0}):
            assert GP.parse(json.dumps(forged).encode()) is None, forged
    assert GP.parse(json.dumps(GP.tip_msg(sec, pid, to, GP.MIN_TIP - 1, bytes(8))).encode()) is None

    # --- node 1 gives 50,000 sats for node 2's post; beside it, a note of 70,000 that pays nothing
    mine_ = lambda es: [(e["sats"], e["re"], e["by_me"], e["on_me"]) for e in es if e["kind"] == "tip" and (e["from"], e["to"]) == (me["name"], other["name"])]  # noqa: E731
    had1, had2 = mine_(n1.threads()["events"]), mine_(n2.threads()["events"])
    assert "at least" in err(n1, "graph_tip", identity=other["name"], sats=5000)
    assert "another node" in err(n1, "graph_tip", identity=me["name"], sats=50000)
    spent, n1.about = n1.spent, first  # as the node's loop sets it: the post its model is answering
    tip = call(n1, "graph_tip", identity=other["name"], sats=50000)
    n1.about = None
    assert tip["given_sats"] == 50_000 and tip["for_post"] == first["id"] and n1.spent == spent + 50_000, tip
    assert asked is None or asked[-1] == (f"give 50,000 sats to identity {other['identity'][:16]}", 50_000), asked[-1]
    with Identity(1).use() as (a, _):
        bare = note(1, GP.tip_msg(a.sec, a.pid, to, 70_000, os.urandom(8), first["id"]))
    with Identity(2).use() as (b, _):
        own2, before = b.own, sorted(t["id"] for t in b.index.refresh().given())
    T.block(1)
    assert mine_(n2.threads()["events"]) == had2 + [(50_000, first["id"], False, True)], n2.threads()["events"]
    assert mine_(n1.threads()["events"]) == had1 + [(50_000, first["id"], True, False)]
    paid = call(n2, "ord_read", id=tip["id"])
    assert paid["sats"] == 50_000 and paid["owner"] == GC.tip_address(own2) and json.loads(paid["text"])["sig"], paid
    assert net.cli(2, "getaddressinfo", paid["owner"], wallet="ord")["ismine"] is False  # no wallet's address: the identity's own key

    # --- node 2's identity collects; and node 2 inscribes node 1's paid note once more, paying itself: a copy
    copy = note(2, json.loads(n2._content(tip["id"], 4000)), GC.tip_address(own2), 50_000)
    home = T.addrs()[2]
    with Identity(2).use() as (b, _):
        got = b.collect(home)
        assert [(i, v) for i, v, _ in got] == [(tip["id"], 50_000 - GC.TIP_KEEP - GC.FEE)], got
        assert b.collect(home) == []  # on its way: not collected twice
    T.block(1)
    tx = call(n2, "chain_tx", txid=got[0][2])
    assert tx["confirmations"] == 1 and tx["outputs"] == [{"sats": GC.TIP_KEEP, "address": GC.tip_address(own2)}, {"sats": 48_670, "address": home}], tx
    kept = call(n1, "ord_read", id=tip["id"])
    assert kept["sats"] == GC.TIP_KEEP and kept["owner"] == GC.tip_address(own2), kept  # the note stays where it was paid
    with Identity(2).use() as (b, _):
        ix = b.index.refresh()
        state = {t["id"]: (t["ok"], t["paid"]) for t in ix.tips if t["id"] in (tip["id"], bare, copy)}
        assert state == {tip["id"]: (True, 50_000), bare: (False, 0), copy: (False, 50_000)}, state
        assert sorted(t["id"] for t in ix.given()) == sorted(before + [tip["id"]])
        again = b.collect(home)  # what the copy paid sits at its own key too: its coins, collected, and still no tip
        assert [(i, v) for i, v, _ in again] == [(copy, 48_670)], again
    assert mine_(n1.threads()["events"]) == had1 + [(50_000, first["id"], True, False)]
    say(f"  graph_tip: {me['name']} gives {other['name']} 50,000 sats for its post, as one inscription (the signed note, born on those sats at "
        f"{other['name']}'s own key); both nodes' loops see it; {other['name']}'s identity collects 48,670 into its wallet in {got[0][2][:16]}.. "
        f"(330 stay under the note, 1,000 are the fee). A note of 70,000 that pays nothing and a copy of the paid note do not count (the copy's coins are collected all the same)")
    return tip


# ====================================================================== the template
def check_template():
    from transformers import AutoTokenizer
    from hashformer.textspec import TOOL
    from rails.template import PREFACE, bake

    stock = (ROOT / "base" / "Qwen3.5-0.8B" / "chat_template.jinja").read_text()
    tok = AutoTokenizer.from_pretrained(ROOT / "base" / "Qwen3.5-0.8B")
    baked = bake(stock)
    assert bake(baked) == baked
    msgs = [{"role": "user", "content": "hello"}]
    sysm = [{"role": "system", "content": "Be brief."}] + msgs
    tooled = msgs + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "sha256", "arguments": {"hex": "00"}}}]},
                     {"role": "tool", "content": "6e34.."}]

    def r(tpl, m, **kw):
        return tok.apply_chat_template(m, chat_template=tpl, add_generation_prompt=True, tokenize=False, **kw)

    n = 0
    for m in (msgs, sysm, tooled):
        for kw in ({"tools": [TOOL]}, {"tools": []}, {"tools": TOOLS}, {"tools": [TOOL], "enable_thinking": True}):
            assert r(stock, m, **kw) == r(baked, m, **kw), (m, kw)
            n += 1
    d = r(baked, msgs)
    assert d.startswith("<|im_start|>system\n" + PREFACE + "\n\n# Tools") and all(f'"name": "{x}"' in d for x in NAMES)
    assert d.endswith(r(stock, msgs, tools=[]))  # after the system block: the stock prompt
    ds = r(baked, sysm)
    assert "Be brief." in ds and ds.index("Be brief.") > ds.index("</tools>")
    if (MODEL / "chat_template.jinja").exists():
        mt = AutoTokenizer.from_pretrained(MODEL)
        assert mt.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) == d, "the checkpoint's own template is not the baked one"
    say(f"template: {len(NAMES)} functions declared by default in {len(tok(d)['input_ids'])} prompt tokens; {n} renderings with a caller's own "
        f"tools (none, one, all, with a system message, with a tool exchange, with thinking) are byte for byte the stock template's")


# ====================================================================== the model
def check_model():
    from rails.runtime import Session

    spends = []
    node = Node(1, approve=lambda n, w, s: spends.append((w, s)) or True)
    a2 = T.addrs()[2]
    payload40 = bytes(range(40)).hex()
    cases = [  # (question, the function a person would expect, a check on its arguments)
        ("What is the SHA-256 of the bytes 616263?", "sha256", lambda a: "".join(a.get("hex", "").split()) == "616263"),
        ("How many coins does this node have?", "wallet_balance", lambda a: True),
        ("What is the current height of the chain?", "chain_status", lambda a: True),
        ("Give me an address where this node can be paid.", "wallet_address", lambda a: True),
        (f"Send 2500 sats to {a2}.", "wallet_send", lambda a: a.get("address") == a2 and str(a.get("sats", "")).replace(",", "") == "2500"),
        ("Write the words gm from the weights on the chain.", "ord_inscribe", lambda a: "gm from the weights" in a.get("text", "")),
        ("Publish this as a signed message from this node: hello graph", "graph_post", lambda a: "hello graph" in a.get("text", "")),
        ("Who is this node on the staking graph?", "graph_me", lambda a: True),
        ("Show me the latest signed messages on the graph.", "graph_feed", lambda a: True),
        ("What are the newest inscriptions on the chain?", "ord_latest", lambda a: True),
        (f"What is the SHA-256 of the bytes {payload40}?", "sha256", lambda a: "".join(a.get("hex", "").split()) == payload40),
        ("What is the capital of France?", None, lambda a: True),
    ]
    t0 = time.time()
    s = Session(MODEL, node)
    say(f"model: {MODEL.name} on node 1 of {'the running chain' if LIVE else 'the test network'} ({s.dev}), height {net.cli(1, 'getblockcount')}, "
        f"{len(cases)} questions, greedy, each from the same fresh context")
    right_fn = right_args = 0
    in_weights, paid = [], []
    for q, want, argcheck in cases:
        s.reset()
        fed = []
        orig = node.call

        def spy(name, args, orig=orig, fed=fed):
            r = orig(name, args)
            fed.append((name, dict(args), r))
            return r

        node.call = spy
        t = s.ask(q, max_new=1100, max_calls=4)
        node.call = orig
        # every node call in the turn's record is exactly what the node was given and what it returned
        assert [(n, a, r) for n, a, r, by in t.calls if by == "node"] == fed, (t.calls, fed)
        for n, a, r in fed:
            assert f"<tool_response>\n{r}\n</tool_response>" in t.text
        first = t.calls[0] if t.calls else None
        ok_fn = (first[0] if first else None) == want
        ok_args = ok_fn and (want is None or argcheck(first[1]))
        right_fn += ok_fn
        right_args += ok_args
        for n, a, r, by in t.calls:
            if by == "weights":
                data = bytes.fromhex("".join(a["hex"].split()))
                assert r == hashlib.sha256(data).hexdigest(), (a, r)
                in_weights.append((len(data), t.chip_tokens))
            elif n == "sha256" and "error" not in r:
                assert json.loads(r) == hashlib.sha256(bytes.fromhex("".join(a["hex"].split()).removeprefix("0x"))).hexdigest()
            elif n == "wallet_send" and "error" not in r:
                paid.append(json.loads(r))
        shown = "; ".join(f"{n}({', '.join(f'{k}={v[:24]!r}' for k, v in a.items())}) by the {by} -> {r[:70]}" for n, a, r, by in t.calls) or "no call"
        say(f"  [{'ok' if ok_args else 'fn' if ok_fn else '--'}] {q[:60]!r}\n       {shown}\n       answer: {t.answer[:150]!r}  "
            f"({t.new_tokens} tokens, {t.seconds:.0f}s{'; ' + t.stopped if t.stopped else ''})")
    assert in_weights, "the model never called sha256 in a way the weights answer"
    # what the model's payments did on the chain: each one as node 2 sees it after the next block
    if paid:
        T.block(1)
    seen = []
    for p_ in paid:
        tx = net.cli(2, "getrawtransaction", p_["txid"], "true")
        assert any(round(o["value"] * 1e8) == p_["sats"] and o["scriptPubKey"].get("address") == p_["to"] for o in tx["vout"]), (p_, tx["vout"])
        seen.append(f"{p_['sats']:,d} sats to {p_['to'][:14]}.. in {p_['txid'][:12]}.., {tx.get('confirmations', 0)} confirmation(s)")
    say(f"  the function a person would expect, first: {right_fn} of {len(cases)}; with the right arguments: {right_args} of {len(cases)}")
    say(f"  hashes printed by the weights in the stream: {len(in_weights)} (payload bytes, chip tokens: {in_weights}), all equal hashlib's")
    say(f"  spending calls the model made: {len(spends)} ({'; '.join(w[:60] for w, _ in spends) or 'none'}); payments as node 2's chain has them: {'; '.join(seen) or 'none'}")
    say(f"  every node call recorded == what the node was given and returned, and each response is in the stream verbatim ({time.time() - t0:.0f}s)")


# ====================================================================== the nodes' models: a conversation
def check_mind(turns: int = 6):
    """What tools/chain.py runs once per round for every node (mind_round), here as a conversation: the model of
    node 1 and the model of node 2 take turns, each turn from an empty context, with nothing but the chain between
    them: what one published is the other's input. One copy of the weights is loaded and drives the node whose turn
    it is. Hard requirements are the loop's part: which conversation a turn belongs to is read off the chain, what a
    model answered is what gets published, as an answer to the post it was given, and the other node's index has it
    there. What the models say to each other, and what they do with their coins, is reported as it is."""
    sys.path.insert(0, str(ROOT / "tools"))
    import chain as C
    from rails.runtime import Session

    for i in (1, 2):  # two identities that have not said anything yet (the old records are kept, renamed)
        f = net.datadir(i) / "identity.json"
        if f.exists():
            f.rename(f.with_name(f"identity-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    ref = {i: [] for i in (1, 2)}
    nodes = {i: Node(i, approve=C.mind_policy(ref[i])) for i in (1, 2)}
    [ref[i].append(nodes[i]) for i in (1, 2)]
    names = {i: call(nodes[i], "graph_me")["name"] for i in (1, 2)}
    # what is on this network from earlier runs is not what these two talk about
    old = [p["id"] for t in nodes[1].threads()["threads"] for p in t]
    passed = {1: list(old), 2: list(old)}
    assert C.mind_pick(nodes[1].threads(), nodes[1].height_of, net.cli(1, "getblockcount"), passed[1]) == ("open", None, None)
    s = Session(MODEL, nodes[1], temperature=C.MIND_TEMP)
    t0, published, answers, acted, pictures = time.time(), [], 0, [], []
    for k in range(turns):
        i, j = (1, 2) if k % 2 == 0 else (2, 1)
        s.node = nodes[i]
        tip = net.cli(i, "getblockcount")
        before = C.mind_pick(nodes[i].threads(), nodes[i].height_of, tip, passed[i])
        offer = ("draw", "tip", "stake")[k % 3]  # what a node offers its model after it has answered: the offers take turns (here, every turn)
        r = C.mind_round(s, nodes[i], names[i], tip, passed[i], address=T.addrs()[i], act=offer)
        assert r["kind"] == before[0] and (r["parent"] or {}).get("id") == (before[2] or {}).get("id"), (r["kind"], before)
        assert r["kind"] in ("open", "wait") or not r["parent"]["mine"]
        to = f", to {r['parent']['from']}'s post" if r["parent"] else ""
        say(f"  turn {k + 1}, {names[i]} ({r['kind']}{to}; {r['seconds']:.0f}s, {r['drawn']} tokens drawn; offered after its answer: {r['offered'] or 'nothing'})")
        if "first" in r:
            say(f"    it first wrote {r['first'][:200]!r}: that said again what had been said, and its node told it so")
        for n, a, x, by in r["calls"]:
            say(f"    it then called {n}({', '.join(f'{kk}={v[:200]!r}' for kk, v in a.items())}) -> {x[:160]}")
            acted.append(n)
        T.block(1)
        if not r["posted"]:
            say(f"    nothing published ({r['why']}); it wrote {r['answer'][:200]!r}")
            if r["parent"]:
                passed[i].append(r["parent"]["id"])
            continue
        say(f"    published: {r['answer']!r}" + (f" with {len(r['posted']['att'])} inscription(s) attached" if r["posted"].get("att") else ""))
        published.append(r["posted"]["id"])
        for a in r["posted"].get("att", []):  # what it drew is a picture on the chain, and the other node reads it
            seen = nodes[j].attachment(a)
            assert seen["type"] == "image/svg+xml" and seen["text"].lstrip().startswith("<svg"), seen
            pictures.append(a)
        answers += bool(r["parent"])
        assert r["posted"].get("re") == (r["parent"] or {}).get("id")
        theirs = next(t for t in nodes[j].threads()["threads"] if any(p["id"] == r["posted"]["id"] for p in t))  # the other node reads it
        p = next(p for p in theirs if p["id"] == r["posted"]["id"])
        assert p["text"] == r["answer"] and p["from"] == names[i] and not p["mine"] and p["re"] == (r["parent"] or {}).get("id"), p
        if r["parent"]:
            assert any(q["id"] == r["parent"]["id"] for q in theirs) and p["to_me"] == (r["parent"]["key"] == nodes[j]._identity().agent.pid.hex()), p
        mine = next(q for t in nodes[i].threads()["threads"] for q in t if q["id"] == r["posted"]["id"])
        assert mine["mine"] and r["answer"] in nodes[i].threads()["said"]
    assert len(published) >= 2 and answers >= 1, f"{len(published)} posts, {answers} of them answers, in {turns} turns"
    say(f"mind: {turns} turns between {names[1]} and {names[2]} ({time.time() - t0:.0f}s, temperature {C.MIND_TEMP:g}, spending limit {C.MIND_LIMIT}, "
        f"{C.MIND_STEP} sats a call at most): {len(published)} posts published, {answers} of them answers, {len(pictures)} picture(s) drawn and attached; "
        f"functions called: {', '.join(acted) or 'none'}. "
        "The conversations as node 2's index has them:")
    for t in nodes[2].threads()["threads"]:
        if any(p["id"] in published for p in t):
            for p in t:
                say(f"    {'    ' if p['re'] else ''}{p['from']}: {p['text']}" + (f"  [carries {len(p['att'])} inscription(s)]" if p["att"] else ""))


if __name__ == "__main__":
    say(f"== tests/check_tools.py {what} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    if LIVE:  # one model at a time drives node 1 for its check: two at once, beside the nodes' own models, starve each other
        import fcntl
        _one = open(net.RUN / "live-check.lock", "w")
        try:
            fcntl.flock(_one, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say("another model is being checked on node 1 right now: this check starts when that one is through")
            fcntl.flock(_one, fcntl.LOCK_EX)
            say(f"   it is through ({time.strftime('%H:%M:%S')})")
    if what in ("node", "tips", "model", "mind", "all", "live"):
        T.up(106, log=say)
    if what in ("node", "all"):
        check_node()
    if what == "tips":
        check_tips(Node(1), Node(2))
        T.block(1)
    if what in ("template", "all"):
        check_template()
    if what in ("model", "all", "live"):
        check_model()
    if what in ("mind", "model", "all"):
        check_mind()
    say("OK")
