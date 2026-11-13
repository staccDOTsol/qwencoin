"""qwg: the social graph's messages. Each is a JSON inscription; ord carries and orders them.

  id      a node's identity, inscribed onto its own bond output. The sats under the inscription
          are the stake; when the bond burns, ord marks the inscription burned. Signed by the
          identity key, so nobody can register (or name) an identity whose key it does not hold.
  vouch   an edge from -> to, inscribed onto a bond that `from` funds and `to`'s faults burn.
          Signed by `from`, so nobody can attribute an edge to an identity that did not make it.
          Both carry a nonce under the signature; a signature counts once (see index.py).
  head    the identity's signed commitment for an epoch. Two different roots for one epoch are
          an equivocation, and the two signatures are the proof that burns its stake.
  attest  the identity's signed claim SHA256(x) = y, computed in its weights. A false claim is a
          proof that burns its stake; the chain's own OP_SHA256 is the judge.
  post    signed text. A post may answer another post ("re": the inscription id of the post it answers) and may
          carry inscriptions ("att": the ids of up to 4 inscriptions, a picture its author drew, say). Both are
          signed with the text, under another tag than a plain post's, so a signature for one cannot be passed off
          as a signature for the other. Posts and the answers to them are the graph's conversations.

Signatures are BIP340 over the exact bytes the covenant scripts check (covenant.head_msg and
covenant.hash_msg), so a message that convicts is already in the form consensus verifies.
"""
import json

from . import covenant as C
from .modelhash import sha256

TAG_ID = b"qwg/id"
TAG_VOUCH = b"qwg/vouch"
TAG_POST = b"qwg/post"
TAG_POST2 = b"qwg/post2"  # a post that answers another or carries inscriptions
MAX_ATT = 4
EPOCH_BLOCKS = 12
# What a stake must look like to count. A covenant's fee goes to whoever mines the slash, so a
# bond that named its own fee could hand the "burned" coins to its owner's miner; and a bond with
# a one-block delay could leave before anyone had read the proof.
MIN_BOND = 100_000  # sats: the fixed fees are at most 2% of a stake
MIN_DELAY, MAX_DELAY = 6, 0xFFFF  # blocks a stake stays slashable after its owner starts to leave


def epoch_of(height: int) -> int:
    return height // EPOCH_BLOCKS


def _bond_fields(b: C.Bond) -> dict:
    return {"bond": b.amount, "delay": b.delay, "fee": b.fee}


def _bond_bytes(b: C.Bond) -> bytes:
    return b.pid + b.own + b.amount.to_bytes(8, "big") + b.delay.to_bytes(4, "big") + b.fee.to_bytes(4, "big")


def bond_of(m: dict) -> C.Bond:
    """The bond an id / vouch message describes (whose address must hold the inscription).
    Raises unless it is one this graph counts: the standard fee, a real stake, a real delay."""
    pid = bytes.fromhex(m["id"] if m["op"] == "id" else m["to"])
    own = bytes.fromhex(m["own"])
    amount, delay, fee = m["bond"], m["delay"], m["fee"]
    if not all(type(v) is int for v in (amount, delay, fee)) or len(pid) != 32 or len(own) != 32:
        raise ValueError("malformed bond")
    if fee != C.FEE or amount < MIN_BOND or not MIN_DELAY <= delay <= MAX_DELAY:
        raise ValueError("not a bond this graph counts")
    return C.Bond(pid, own, amount, delay, fee)


def id_bytes(b: C.Bond, name: str, nonce: bytes) -> bytes:
    assert len(nonce) == 8
    return TAG_ID + _bond_bytes(b) + nonce + sha256(name.encode())


def id_msg(sec: bytes, b: C.Bond, name: str, nonce: bytes) -> dict:
    return {"p": "qwg", "op": "id", "id": b.pid.hex(), "own": b.own.hex(), "name": name, **_bond_fields(b),
            "n": nonce.hex(), "sig": C.sign(sec, id_bytes(b, name, nonce)).hex()}


def vouch_bytes(frm: bytes, b: C.Bond, nonce: bytes) -> bytes:
    assert len(nonce) == 8
    return TAG_VOUCH + frm + _bond_bytes(b) + nonce


def vouch_msg(frm_sec: bytes, frm: bytes, b: C.Bond, nonce: bytes) -> dict:
    return {"p": "qwg", "op": "vouch", "from": frm.hex(), "to": b.pid.hex(), "own": b.own.hex(),
            **_bond_fields(b), "n": nonce.hex(), "sig": C.sign(frm_sec, vouch_bytes(frm, b, nonce)).hex()}


def head_msg(sec: bytes, pid: bytes, epoch: int, root: bytes) -> dict:
    return {"p": "qwg", "op": "head", "id": pid.hex(), "epoch": epoch, "root": root.hex(),
            "sig": C.sign(sec, C.head_msg(epoch, root)).hex()}


def attest_msg(sec: bytes, pid: bytes, x: bytes, y: bytes) -> dict:
    return {"p": "qwg", "op": "attest", "id": pid.hex(), "x": x.hex(), "y": y.hex(),
            "sig": C.sign(sec, C.hash_msg(x, y)).hex()}


def inscription_id(x) -> bool:
    """An inscription id as ord writes it: the reveal transaction's id, "i", the inscription's index in it."""
    return (type(x) is str and len(x) >= 66 and x[64] == "i" and x[65:].isdigit() and len(x) <= 70
            and all(c in "0123456789abcdef" for c in x[:64]))


def post_bytes(pid: bytes, seq: int, text: str, re: str | None = None, att=()) -> bytes:
    if re is None and not att:
        return TAG_POST + pid + seq.to_bytes(8, "big") + sha256(text.encode())
    assert (re is None or inscription_id(re)) and len(att) <= MAX_ATT and all(inscription_id(a) for a in att)
    return (TAG_POST2 + pid + seq.to_bytes(8, "big") + sha256(text.encode()) + sha256((re or "").encode())
            + sha256(",".join(att).encode()))


def post_msg(sec: bytes, pid: bytes, seq: int, text: str, re: str | None = None, att=()) -> dict:
    return {"p": "qwg", "op": "post", "id": pid.hex(), "seq": seq, "text": text, **({"re": re} if re is not None else {}),
            **({"att": list(att)} if att else {}), "sig": C.sign(sec, post_bytes(pid, seq, text, re, att)).hex()}


def parse(content: bytes):
    """A well-formed qwg message with a valid signature where one is required, else None."""
    try:
        m = json.loads(content)
        if not isinstance(m, dict) or m.get("p") != "qwg":
            return None
        op = m["op"]
        if op == "id":
            b = bond_of(m)
            msg = id_bytes(b, m["name"], bytes.fromhex(m["n"]))
            return m if C.verify(b.pid, bytes.fromhex(m["sig"]), msg) else None
        if op == "vouch":
            b, frm = bond_of(m), bytes.fromhex(m["from"])
            msg = vouch_bytes(frm, b, bytes.fromhex(m["n"]))
            return m if C.verify(frm, bytes.fromhex(m["sig"]), msg) else None
        pid, sig = bytes.fromhex(m["id"]), bytes.fromhex(m["sig"])
        if op == "head":
            msg = C.head_msg(int(m["epoch"]), bytes.fromhex(m["root"]))
        elif op == "attest":
            msg = C.hash_msg(bytes.fromhex(m["x"]), bytes.fromhex(m["y"]))
        elif op == "post":
            att = m.get("att", [])
            if (type(m["text"]) is not str or ("re" in m and not inscription_id(m["re"])) or type(att) is not list
                    or not 0 < len(att) <= MAX_ATT and "att" in m or not all(inscription_id(a) for a in att)):
                return None
            msg = post_bytes(pid, int(m["seq"]), m["text"], m.get("re"), att)
        else:
            return None
        return m if C.verify(pid, sig, msg) else None
    except Exception:
        return None
