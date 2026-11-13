"""Attacks on the accounting, on the live chain. Sybil resistance is the claim that weight cannot
exceed locked sats, so every way of making one sat count twice (or count without being at risk)
has to fail:

  1a. two stake messages revealed by one transaction into one bond output
  1b. two stake messages inscribed elsewhere, then moved together into one bond-shaped output
  2.  a replay of somebody's signed vouch onto a second bond output
  3.  an id for somebody else's key, signed by the attacker
  4.  an id with no signature at all
  5.  a bond whose "burn" would mostly be a fee, paid to whoever mines the slash
  6.  a bond its owner can leave one block after misbehaving

and an honest vouch has to count, exactly once, on every node.

    PYTHONPATH=. python3 tests/check_sybil.py        (after tools/launch.py; uses the three nodes)
"""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, "tools")
import net
from graph import covenant as C, protocol as P
from graph.agent import Agent
from graph.index import Index
from test_framework import address
from test_framework.messages import COutPoint, CTransaction, CTxIn, CTxInWitness, CTxOut
from test_framework.script import OP_0, OP_CHECKSIG, OP_ENDIF, OP_IF, CScript, TaprootSignatureHash, taproot_construct

ADDR = {int(k): v for k, v in json.load(open("run/addrs.json")).items()}
DELAY = 6
tag = os.urandom(3).hex()  # fresh names, so the test can run again on the same chain


def mine(n=1):
    for _ in range(600):
        if len({frozenset(net.cli(j, "getrawmempool")) for j in (1, 2, 3)}) == 1:
            break
        net.time.sleep(0.5)
    net.cli(1, "generatetoaddress", n, ADDR[1], 9000000)
    net.settle()
    for j in (1, 2, 3):
        net.ord_wait(j, net.cli(1, "getblockcount"))


def write(msg) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write(msg if isinstance(msg, str) else json.dumps(msg, separators=(",", ":")))
    f.close()
    return f.name


def inscribe_raw(node, msg, dest, postage):
    """Inscribe arbitrary content (not necessarily a valid message) onto a bond-shaped output."""
    r = net.ord_wallet(node, "inscribe", "--fee-rate", "2", "--no-backup", "--file", write(msg),
                       "--destination", dest, "--postage", f"{postage}sat")
    return r["inscriptions"][0]["id"]


def move_together(node, msgs, dest, total):
    """Inscribe each message into the attacker's wallet, then spend those outputs together into
    one output at `dest`. Sats flow first in, first out, so every inscribed sat lands in output 0."""
    part = total // len(msgs)
    ids = [net.ord_wallet(node, "inscribe", "--fee-rate", "2", "--no-backup", "--file", write(m),
                          "--postage", f"{part}sat")["inscriptions"][0]["id"] for m in msgs]
    mine()
    ins = []
    for i in ids:
        txid, vout, off = net.ord_get(node, f"/inscription/{i}")["satpoint"].split(":")
        assert off == "0" and net.ord_get(node, f"/output/{txid}:{vout}")["value"] == part
        ins.append({"txid": txid, "vout": int(vout)})
    coin = next(c for c in net.ord_wallet(node, "cardinals") if c["amount"] > 100_000)
    ftxid, fvout = coin["output"].split(":")
    change = net.cli(node, "getrawchangeaddress", "bech32m", wallet="ord")
    outs = [{dest: f"{total / 1e8:.8f}"}, {change: f"{(coin['amount'] - 2000) / 1e8:.8f}"}]
    raw = net.cli(node, "createrawtransaction", json.dumps(ins + [{"txid": ftxid, "vout": int(fvout)}]), json.dumps(outs))
    signed = net.cli(node, "signrawtransactionwithwallet", raw, wallet="ord")
    assert signed["complete"], signed
    net.cli(node, "sendrawtransaction", signed["hex"])
    return ids


def reveal_together(node, msgs, bond):
    """One reveal transaction with an inscription envelope per message, paying the bond output."""
    sec, pub = C.keypair(os.urandom(8))
    ops = [pub, OP_CHECKSIG]
    for m in msgs:
        ops += [OP_0, OP_IF, b"ord", bytes([1]), b"application/json", OP_0,
                json.dumps(m, separators=(",", ":")).encode(), OP_ENDIF]
    info = taproot_construct(pub, [("reveal", CScript(ops))])
    value = bond.amount + 1000
    commit = net.cli(node, "sendtoaddress", address.output_key_to_p2tr(info.output_pubkey, main=True),
                     f"{value / 1e8:.8f}", wallet="ord")
    vout = next(o["n"] for o in net.cli(node, "getrawtransaction", commit, 1)["vout"]
                if o["scriptPubKey"]["hex"] == bytes(info.scriptPubKey).hex())
    tx = CTransaction()
    tx.version = 2
    tx.vin = [CTxIn(COutPoint(int(commit, 16), vout), nSequence=0xFFFFFFFD)]
    tx.vout = [CTxOut(bond.amount, CScript(bond.spk()))]
    lf = info.leaves["reveal"]
    h = TaprootSignatureHash(tx, [CTxOut(value, info.scriptPubKey)], 0, 0, scriptpath=True, leaf_script=lf.script)
    wit = CTxInWitness()
    wit.scriptWitness.stack = [C.sign(sec, h), bytes(lf.script),
                               bytes([lf.version + info.negflag]) + info.internal_pubkey + lf.merklebranch]
    tx.wit.vtxinwit = [wit]
    txid = net.cli(node, "sendrawtransaction", C.tx_hex(tx))
    return [f"{txid}i{k}" for k in range(len(msgs))]


alice, bob, mallory = Agent(1, f"alice-{tag}"), Agent(2, f"bob-{tag}"), Agent(3, f"mallory-{tag}")
alice.bond(600_000, DELAY)
bob.bond(600_000, DELAY)
mine()
bob.vouch(alice.pid, 300_000, DELAY)
mine()
ix = Index(1).refresh()
base = ix.weight()[alice.pid]
print(f"baseline: alice weighs {base:,d} (bond 600,000 + bob's vouch 300,000), trust bob -> alice {ix.trust(bob.pid, alice.pid):,d}")
assert base == 900_000 and ix.trust(bob.pid, alice.pid) == 300_000

# 1. one bond output of 700,000 sat carrying two id messages (1,400,000 claimed), built both ways
b = C.Bond(mallory.pid, mallory.own, 700_000, DELAY)
b2 = C.Bond(mallory.pid, mallory.own, 700_001, DELAY)
born_two = reveal_together(3, [P.id_msg(mallory.sec, b, mallory.name, os.urandom(8)) for _ in range(2)], b)
moved_two = move_together(3, [P.id_msg(mallory.sec, b2, mallory.name, os.urandom(8)) for _ in range(2)], b2.address(), 700_001)
twins = born_two + moved_two

# 2. bob's signed vouch for alice, copied byte for byte onto a second output
vouch = next(s for s in ix.stakes.values() if s.kind == "vouch" and s.frm == bob.pid and s.bond.pid == alice.pid)
replay = inscribe_raw(3, ix._content(vouch.inscription).decode(), vouch.bond.address(), vouch.bond.amount)

# 3. an id for alice's key, signed by mallory and funded by her; 4. the same with no signature
fb = C.Bond(alice.pid, mallory.own, 400_000, DELAY)
forged = inscribe_raw(3, P.id_msg(mallory.sec, fb, "alice (the real one)", os.urandom(8)), fb.address(), 400_000)
bare = {k: v for k, v in P.id_msg(mallory.sec, fb, "alice", os.urandom(8)).items() if k not in ("sig", "n")}
unsigned = inscribe_raw(3, bare, fb.address(), 400_000)

# 5. a bond whose covenant fee is 300,000 sat: a slash would pay that to whoever mines it
#    (the owner's own miner, say) instead of burning it. 6. a bond that can leave after one block.
greedy = C.Bond(mallory.pid, mallory.own, 800_000, DELAY, fee=300_000)
quick = C.Bond(mallory.pid, mallory.own, 800_000, 1)
odd = []
for bond in (greedy, quick):
    msg = {"p": "qwg", "op": "id", "id": bond.pid.hex(), "own": bond.own.hex(), "name": mallory.name,
           "bond": bond.amount, "delay": bond.delay, "fee": bond.fee, "n": "00" * 8,
           "sig": C.sign(mallory.sec, P.id_bytes(bond, mallory.name, bytes(8))).hex()}
    odd.append(inscribe_raw(3, msg, bond.address(), bond.amount))
mine()


def bond_shaped(node, iid, bond) -> bool:
    """The whole of the naive rule: the inscription sits on an output with the bond's script and value."""
    out = net.ord_get(node, "/output/" + net.ord_get(node, f"/inscription/{iid}")["satpoint"].rsplit(":", 1)[0])
    return out["script_pubkey"] == bond.spk().hex() and out["value"] == bond.amount


# every attack does land on chain: a count per bond-shaped inscription would credit all of it
assert all(bond_shaped(1, i, b) for i in born_two) and all(bond_shaped(1, i, b2) for i in moved_two)
assert bond_shaped(1, replay, vouch.bond)
assert bond_shaped(1, forged, fb) and bond_shaped(1, unsigned, fb)
assert bond_shaped(1, odd[0], greedy) and bond_shaped(1, odd[1], quick)
print("on chain: two bond outputs with two ids each, a replayed vouch, two false ids, two rigged bonds; "
      "a count per bond-shaped inscription would credit all of them")

for j in (1, 2, 3):
    ix = Index(j).refresh()
    w = ix.weight()
    assert {ix.stakes[i].status for i in twins} == {"invalid"} and not any(ix.stakes[i].born for i in twins)
    assert ix.stakes[replay].status == "replay" and ix.stakes[vouch.inscription].status == "active"
    assert all(i not in ix.stakes and ix.seen[i] is None for i in (forged, unsigned, *odd))
    assert ix.identities()[alice.pid] == alice.name and mallory.pid not in ix.identities()
    assert w.get(mallory.pid, 0) == 0 and w[alice.pid] == base and ix.trust(bob.pid, alice.pid) == 300_000
    assert sum(s.sats for s in ix.stakes.values() if s.bond.pid in (alice.pid, mallory.pid)) == base
print("1a. two ids revealed into one bond output:  neither is a stake (1,400,000 claimed, 700,000 locked)")
print("1b. two ids moved into one bond output:    neither is a stake; mallory weighs 0")
print(f"2.  bob's vouch replayed on a 2nd output:   'replay'; alice still weighs {base:,d}, trust bob -> alice still 300,000")
print("3.  id for alice's key signed by mallory:   not a message (the signature fails under alice's key)")
print(f"4.  id for alice's key with no signature:   not a message; alice is still named {alice.name!r}")
print("5.  bond with a 300,000 sat covenant fee:   not a message (a slash must burn, not pay a miner)")
print("6.  bond with a one-block delay:            not a message (a stake must outlast its proof)")

# 7. the honest case counts, once, everywhere
mallory.bond(500_000, DELAY)
mine()
mallory.vouch(alice.pid, 250_000, DELAY)
mine()
for j in (1, 2, 3):
    ix = Index(j).refresh()
    assert ix.weight()[alice.pid] == base + 250_000 and ix.weight()[mallory.pid] == 500_000
    assert ix.trust(mallory.pid, alice.pid) == 250_000
print(f"7.  an honest vouch of 250,000:             alice weighs {base + 250_000:,d}, mallory 500,000, on all three nodes")
print("OK: weight never exceeded locked sats")
