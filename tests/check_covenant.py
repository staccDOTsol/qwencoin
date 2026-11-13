"""The slashing covenants on the live chain: every accept / reject below is a consensus decision
of a node whose SHA-256 (sighashes, OP_SHA256, the BIP340 challenges inside CSFS) runs in the weights."""
import json, sys, time
sys.path.insert(0, "tools")
import net
from graph import covenant as C
from graph.modelhash import sha256
from test_framework.messages import CTxOut
from test_framework.script import CScript

N = 1
addr = json.load(open("run/addrs.json"))[str(N)]
def mine(n=1):
    net.cli(N, "generatetoaddress", n, addr, 9000000)
def fund(b):
    txid = net.cli(N, "sendtoaddress", b.address(), b.amount / 1e8, wallet="ord")
    tx = net.cli(N, "getrawtransaction", txid, 1)
    vout = next(o["n"] for o in tx["vout"] if o["scriptPubKey"]["hex"] == b.spk().hex())
    return f"{txid}:{vout}"
def accept(tx, burn=1.0):
    r = net.cli(N, "testmempoolaccept", json.dumps([C.tx_hex(tx)]), 0)[0]
    return r["allowed"], r.get("reject-reason", "")
def send(tx):
    return net.cli(N, "sendrawtransaction", C.tx_hex(tx), 0, 1)

sec, pid = C.keypair(b"t-id"); osec, own = C.keypair(b"t-own")
bonds = [C.Bond(pid, own, 5_000_000 + i, delay=6) for i in range(4)]
ops = [fund(b) for b in bonds]; mine()
print("funded 4 bonds", [o[:12] for o in ops])

# --- lie
b, op = bonds[0], ops[0]
x = bytes.fromhex(net.cli(N, "getblockheader", net.cli(N, "getbestblockhash"), "false"))
y = sha256(x); bad = bytes([y[0] ^ 1]) + y[1:]
honest = b.slash_lie(op, 1, x, y, C.sign(sec, C.hash_msg(x, y)))
print("slash with a TRUE hash claim     ->", accept(honest))
lie = b.slash_lie(op, 1, x, bad, C.sign(sec, C.hash_msg(x, bad)))
forged = b.slash_lie(op, 1, x, bad, C.sign(osec, C.hash_msg(x, bad)))
print("false claim signed by another key ->", accept(forged))
steal = b.slash_lie(op, 1, x, bad, C.sign(sec, C.hash_msg(x, bad)))
steal.vout = [CTxOut(b.amount - b.fee, CScript(bytes.fromhex(net.cli(N, "getaddressinfo", addr, wallet="ord")["scriptPubKey"])))]
print("valid proof, coins sent to slasher ->", accept(steal))
print("valid proof, coins burned          ->", accept(lie)); t1 = send(lie)

# --- equivocation
b, op = bonds[1], ops[1]
ra, rb = sha256(b"history A"), sha256(b"history B")
same = b._spend(op, 1, "equiv", [C.sign(sec, C.head_msg(9, ra)), ra, C.sign(sec, C.head_msg(9, ra)), ra, (9).to_bytes(8, "big")], [C.burn_output(b.amount - b.fee)])
print("two signatures over ONE root      ->", accept(same))
other_epoch = b._spend(op, 1, "equiv", [C.sign(sec, C.head_msg(10, rb)), rb, C.sign(sec, C.head_msg(9, ra)), ra, (9).to_bytes(8, "big")], [C.burn_output(b.amount - b.fee)])
print("roots signed for different epochs ->", accept(other_epoch))
eq = b.slash_equivocation(op, 1, 9, ra, C.sign(sec, C.head_msg(9, ra)), rb, C.sign(sec, C.head_msg(9, rb)))
print("two roots, one epoch              ->", accept(eq)); t2 = send(eq)

# --- unbond, slash during the delay, withdraw
b, op = bonds[2], ops[2]
thief = b.unbond(op, sec); print("unbond signed by a non-owner      ->", accept(thief))
u = b.unbond(op, osec); print("unbond by the owner               ->", accept(u)); tu = send(u)
b3, op3 = bonds[3], ops[3]
u3 = b3.unbond(op3, osec); tu3 = send(u3)
mine()
w = b.withdraw(f"{tu}:0", osec, bytes.fromhex(net.cli(N, "getaddressinfo", addr, wallet="ord")["scriptPubKey"]))
print("withdraw before the delay         ->", accept(w))
late = b3.slash_lie(f"{tu3}:0", 2, x, bad, C.sign(sec, C.hash_msg(x, bad)))
print("slash while unbonding             ->", accept(late)); t3 = send(late)
mine(6)
print("withdraw after the delay          ->", accept(w)); tw = send(w); mine()
for name, t in [("lie burn", t1), ("equiv burn", t2), ("unbonding burn", t3), ("withdraw", tw)]:
    tx = net.cli(N, "getrawtransaction", t, 1)
    o = tx["vout"][0]
    print(f"{name:15s} {t[:16]} confirmations {tx.get('confirmations')} -> {o['value']} to {o['scriptPubKey']['type']} {o['scriptPubKey'].get('asm','')[:40]}")
