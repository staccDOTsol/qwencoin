"""The graph, live: three nodes (each a qwencoind + ord whose SHA-256 runs in the weights) bond,
attest, vouch, cheat, get slashed, and leave. Run after `tools/net.py up 3` with funded ord wallets.

    PYTHONPATH=. python3 tools/demo.py
"""
import json, sys, time
sys.path.insert(0, "tools")
import net
from graph import covenant as C, protocol as P
from graph.agent import Agent
from graph.modelhash import sha256
from hashformer import executor as X

ADDR = {int(k): v for k, v in json.load(open("run/addrs.json")).items()}
BOND, VOUCH, DELAY = 5_000_000, 1_000_000, 6
T0 = time.time()
turn = [0]


def say(*a):
    print(f"[{time.time() - T0:6.0f}s h={net.cli(1, 'getblockcount'):4d}]", *a, flush=True)


def sync_mempools(timeout=120):
    """Wait until transaction relay has given every node the same mempool."""
    for _ in range(timeout * 2):
        pools = [set(net.cli(j, "getrawmempool")) for j in (1, 2, 3)]
        if pools[0] == pools[1] == pools[2]:
            return
        time.sleep(0.5)
    raise RuntimeError("mempools did not converge")


def mine(n=1):
    """Nodes take turns mining what relay has delivered. Each miner builds on the tip all three
    agree on: mining before the previous block has propagated makes a stale block, here as on
    any proof-of-work chain. Then everyone (bitcoind and ord) catches up."""
    sync_mempools()
    for _ in range(n):
        net.settle()
        i = turn[0] % 3 + 1
        turn[0] += 1
        net.cli(i, "generatetoaddress", 1, ADDR[i], 9000000)
    net.settle()
    h = net.cli(1, "getblockcount")
    for j in (1, 2, 3):
        net.ord_wait(j, h)


def split(i, n=14):
    """Give node i's ord wallet n separate coins to inscribe from."""
    outs = {net.cli(i, "getnewaddress", "", "bech32m", wallet="ord"): 1 for _ in range(n)}
    net.cli(i, "sendmany", "", json.dumps(outs), wallet="ord")


def names(ix):
    return {pid: nm for pid, nm in ix.identities().items()}


def show(ix, viewer=None):
    ix.refresh()
    nm = names(ix)
    w = ix.weight()
    conv = ix.convicted()
    for pid, name in sorted(nm.items(), key=lambda kv: kv[1]):
        own = sum(s.sats for s in ix.stakes.values() if s.kind == "id" and s.bond.pid == pid)
        st = sorted({s.status for s in ix.stakes.values() if s.kind == "id" and s.bond.pid == pid})
        line = f"    {name:10s} weight {w.get(pid, 0):>9,d} sat (own bond {own:>9,d}, vouched {w.get(pid, 0) - own:>9,d})  bond {'/'.join(st)}"
        if viewer is not None:
            line += f"  trust from {nm[viewer.pid]}: {ix.trust(viewer.pid, pid):>9,d}"
        if pid in conv:
            line += "  CONVICTED"
        print(line, flush=True)
    for (a, b), c in sorted(ix.edges().items(), key=lambda kv: (nm.get(kv[0][0], "?"), nm.get(kv[0][1], "?"))):
        print(f"    vouch {nm.get(a, a.hex()[:8]):>10s} -> {nm.get(b, b.hex()[:8]):10s} {c:>9,d} sat", flush=True)


def burned_total():
    """Sats sitting in OP_RETURN outputs of slash transactions (provably destroyed)."""
    return sum(int(round(o["value"] * 1e8)) for t in BURNS
               for o in net.cli(1, "getrawtransaction", t, 1)["vout"] if o["scriptPubKey"]["type"] == "nulldata")


BURNS = set()
k0 = X.tokens()
alice, bob, mallory, eve = Agent(1, "alice"), Agent(2, "bob"), Agent(3, "mallory"), Agent(3, "eve")
say("three nodes:", {i: net.cli(i, "getnetworkinfo")["subversion"] for i in (1, 2, 3)})
for i in (1, 2, 3):
    split(i)
mine()

say("1. bonds: each identity is an inscription sitting on its own covenant output")
for a in (alice, bob, mallory, eve):
    b = a.bond(BOND, DELAY)
    say(f"   {a.name:8s} id {a.pid.hex()[:16]}  bond {b.address()}")
mine()
show(alice.index, alice)

say("2. work: each node hashes the tip header in its weights and signs the result; posts")
tip = net.cli(1, "getbestblockhash")
x = bytes.fromhex(net.cli(1, "getblockheader", tip, "false"))
for a in (alice, bob, mallory):
    a.attest(x)
alice.post("gm. this node's sha-256 is a language model")
bob.post("verified: alice's hash of the tip header matches what my weights compute")
mine()

say("3. audit, then vouch: stake placed on a peer burns if that peer is ever caught")
for a in (alice, bob):
    got = a.audit()
    say(f"   {a.name} recomputed {a.index.hashes_checked} attested hashes in its weights: {len(a.index.proofs())} faults")
alice.vouch(bob.pid, VOUCH, DELAY)
alice.vouch(mallory.pid, VOUCH, DELAY)
bob.vouch(alice.pid, VOUCH, DELAY)
bob.vouch(eve.pid, VOUCH, DELAY)
mine()
show(alice.index, alice)

say("4. heads: each identity signs one commitment to its history for this epoch")
ep = P.epoch_of(net.cli(1, "getblockcount"))
for a in (alice, bob, mallory, eve):
    a.head(ep)
mine()

say("5. faults: mallory signs a false hash; eve signs a second, different head for the same epoch")
y = sha256(x)
mallory.attest(x, bytes([y[0] ^ 0x80]) + y[1:])
try:
    eve.head(ep, sha256(b"a different history"))
except RuntimeError as e:
    say(f"   (an honest agent refuses: {e})")
eve.head(ep, sha256(b"a different history"), equivocate=True)
mine()
say("   mallory tries to run: starts unbonding her own bond")
st = mallory.stake("id")
mallory.unbond(st)
mine()
say(f"   her bond is now {mallory.stake('id').status}: still slashable for {DELAY} blocks")

say("6. slashing: bob audits; the proofs alone burn every stake on the offenders")
for kind, pid, iid, txid in bob.audit():
    BURNS.add(txid)
    who = names(bob.index).get(pid, "?")
    st = bob.index.stakes[iid]
    say(f"   {kind:5s} proof against {who:8s} burns the {st.kind:5s} stake of {st.bond.amount:,d} sat (tx {txid[:16]})")
late = alice.audit()
new = {t for *_, t in late} - BURNS
BURNS.update(new)
say(f"   alice audits too and builds the same transactions: {len(new)} burns that bob had not already broadcast")
mine()
show(alice.index, alice)
nm = names(alice.index)
for st in alice.index.stakes.values():
    if st.status == "slashed":
        info = net.ord_get(1, f"/inscription/{st.inscription}")
        whose = nm.get(st.bond.pid, "?") if st.kind == "id" else f"{nm.get(st.frm, '?')} -> {nm.get(st.bond.pid, '?')}"
        say(f"   ord: inscription {st.inscription[:12]}.. ({st.kind} {whose}) charms {info['charms']}")
say(f"   sats destroyed in OP_RETURN outputs: {burned_total():,d}")

say("7. leaving honestly: alice unbonds her vouch for bob, waits out the delay, withdraws")
st = alice.stake("vouch", bob.pid)
alice.unbond(st)
mine()
st = alice.stake("vouch", bob.pid)
try:
    alice.withdraw(st, ADDR[1])
    say("   withdraw before the delay: accepted (WRONG)")
except RuntimeError as e:
    say(f"   withdraw before the delay: rejected ({str(e).splitlines()[-1][:40]})")
mine(DELAY)
txid = alice.withdraw(st, ADDR[1])
mine()
say(f"   withdraw after {DELAY} blocks: tx {txid[:16]} confirmed")
show(alice.index, alice)

say("8. sybils: mallory's operator spends 3,000,000 sat on three fresh names vouching in a ring")
sybils = [Agent(3, f"sybil{i}") for i in range(3)]
for s in sybils:
    s.bond(500_000, DELAY)
mine()
for i, s in enumerate(sybils):
    s.vouch(sybils[(i + 1) % 3].pid, 500_000, DELAY)
mine()
ix = alice.index.refresh()
w = ix.weight()
tot = sum(w[s.pid] for s in sybils)
say(f"   weight of the three names together: {tot:,d} sat = the 3,000,000 locked; one name with one bond would weigh the same")
say(f"   trust from alice into the ring: {[ix.trust(alice.pid, s.pid) for s in sybils]} (no honest vouch crosses into it)")
show(alice.index, alice)

tips = {i: net.cli(i, "getbestblockhash") for i in (1, 2, 3)}
assert len(set(tips.values())) == 1
say(f"done: all three nodes agree on tip {tips[1][:20]}.. at height {net.cli(1, 'getblockcount')}; "
    f"the agents alone ran {X.tokens() - k0:,d} model tokens of SHA-256")
