"""The model's tools on a live chain (rails/): what the checkpoint declares, what the node does
for it, and the merged model using them.

    python3 tests/check_tools.py [node|template|model|mind|all] [MODEL_DIR=dist/qwen08-sha]     -> logs/tools.log
    python3 tests/check_tools.py live MODEL_DIR       the model part on the running chain (run/), as node 1

On a private two-node network of the qwencoin chain (tests/tools_net.py; real blocks, real ord),
or, with `live`, on the chain that tools/chain.py is running (what the model does there is real:
its payment goes to node 2, its inscription and its post stay on the chain):

  node      no model. Every function through Node.call, the way the runtime calls it, checked
            against the chain: coins move, an inscription is written, read, listed and given
            away, identities bond, post and vouch, a lie is audited and its stake and the vouch
            on it burn, a stake starts to leave. Then what must be refused: unapproved spending,
            bad addresses and amounts, key export, stopping the node, unknown functions.
  template  the checkpoint's chat template: the functions are declared when the caller declares
            none; a caller's own list renders byte for byte as the stock template renders it.
  model     the merged checkpoint asked in plain words, greedy, one question per fresh context.
            Hard requirements: a hash the weights print is hashlib's; every call the model made
            reached the node unedited and its answer reached the model unedited. What the model
            chose to call, and with what arguments, is reported as it is: that part is the
            language model's skill, not this code's.
"""
import hashlib, json, sys, time
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
    feed = call(n1, "graph_feed", count=3)
    assert feed[0] == {"from": other["name"], "at_stake_sats": 150_000, "standing": "staked", "text": post}, feed
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


# ====================================================================== a node's mind: one round
def check_mind():
    """What tools/chain.py runs once per round for every node, here once for node 1: the model
    reads the graph and publishes a signed message of its own. Node 2 must then read it."""
    sys.path.insert(0, str(ROOT / "tools"))
    import chain as C
    from graph.identity import read as identity
    from rails.runtime import Session

    n2 = Node(2)
    theirs = "node 2 here, checking who else is awake " + time.strftime("%H:%M:%S")
    call(n2, "graph_post", text=theirs)
    T.block(2)
    ref = []
    node = Node(1, approve=C.mind_policy(ref))
    ref.append(node)
    me = call(node, "graph_me")["name"]
    s = Session(MODEL, node)
    t0 = time.time()
    look = s.ask(C.MIND_LOOK.format(me=me), max_new=500, max_calls=2)
    speak = s.ask(C.MIND_SPEAK, max_new=500, max_calls=3)
    calls = look.calls + speak.calls
    for n, a, r, by in calls:
        say(f"    {n}({', '.join(f'{k}={v[:80]!r}' for k, v in a.items())}) -> {r[:140]}")
    said = [a.get("text", "") for n, a, r, by in calls if n == "graph_post" and '"error"' not in r]
    assert any(n == "graph_feed" for n, a, r, by in look.calls), "the model did not read the graph"
    assert said, f"the model did not publish; it wrote {speak.answer[:200]!r}"
    T.block(1)
    feed = call(n2, "graph_feed", count=3)
    assert feed[0]["from"] == me and feed[0]["text"] == said[-1][:300], feed
    say(f"mind: one round on node 1 ({time.time() - t0:.0f}s, spending limit {C.MIND_LIMIT}): the model read the feed (node 2 had written {theirs!r}) "
        f"and published {said[-1]!r} as {me}; node 2's index shows it, signed, with {feed[0]['at_stake_sats']:,d} sats at stake on its author")


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
    if what in ("node", "model", "mind", "all", "live"):
        T.up(106, log=say)
    if what in ("node", "all"):
        check_node()
    if what in ("template", "all"):
        check_template()
    if what in ("model", "all", "live"):
        check_model()
    if what in ("mind", "model", "all"):
        check_mind()
    say("OK")
