"""Bonds and vouches: taproot covenants that burn on a proof of fault.

A bond locks sats under three script leaves (the internal key is the NUMS point, so there is no
key path):

  unbond  <own> CHECKSIGVERIFY <T_unbond> CTV
          the owner starts leaving; CTV forces the coins into the unbonding output below
  equiv   two signatures by <id> over two different roots for one epoch -> <T_burn> CTV
  lie     a signature by <id> over (x, y) with y != SHA256(x)           -> <T_burn> CTV

and the unbonding output is

  withdraw  <delay> CSV DROP <own> CHECKSIG
  equiv, lie as above

`id` is the identity whose behaviour the coins answer for; `own` is who gets them back. For an
identity bond both belong to the node. For a vouch, `id` is the peer vouched for and `own` the
voucher: the voucher's coins burn if the peer is caught.

T_burn fixes the spending transaction to a single OP_RETURN output carrying the whole amount
(minus a fixed fee): the sats are destroyed, nobody is paid, so there is nothing to gain by
slashing yourself or front-running a slash, and anyone holding the proof can broadcast it at no
cost. The proofs are checked by consensus (CHECKSIGFROMSTACK, CAT, SHA256), not by an indexer:
the node that verifies them computes its SHA-256 in the weights.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Bitcoin Core's own Python transaction, script and key code: from the source tree, or the copy a kit ships
for _p in (ROOT / "upstream" / "bitcoin" / "test" / "functional", ROOT / "vendor"):
    if (_p / "test_framework").is_dir():
        sys.path.insert(0, str(_p))
        break

from test_framework import address, key, messages, script  # noqa: E402
from test_framework.messages import COutPoint, CTransaction, CTxIn, CTxInWitness, CTxOut  # noqa: E402
from test_framework.script import (  # noqa: E402
    OP_2DUP, OP_CAT, OP_CHECKSEQUENCEVERIFY, OP_CHECKSIG, OP_CHECKSIGFROMSTACK, OP_CHECKSIGVERIFY,
    OP_CHECKTEMPLATEVERIFY, OP_DROP, OP_DUP, OP_EQUAL, OP_EQUALVERIFY, OP_FROMALTSTACK, OP_NOT,
    OP_RETURN, OP_ROT, OP_SHA256, OP_SIZE, OP_SWAP, OP_TOALTSTACK, OP_TUCK, OP_VERIFY, CScript,
    TaprootSignatureHash, taproot_construct)

from . import modelhash  # noqa: E402

modelhash.install(messages, key, script, address)  # the builder's SHA-256 runs in the weights

NUMS = bytes.fromhex(key.H_POINT)
TAG_HEAD = b"qwg/head"  # head message: TAG_HEAD || epoch (8 bytes, big endian) || root (32)
TAG_HASH = b"qwg/hash"  # attestation:  TAG_HASH || y (32) || x  (the claim SHA256(x) = y)
BURN_NOTE = b"qwg:slashed"
SEQ = 0xFFFFFFFD
MAX_X = 80  # tapscript relay policy caps a witness item at 80 bytes: enough for a block header
FEE = 1000  # sats each covenant transaction leaves to the miner; everything else of a slash burns


def head_msg(epoch: int, root: bytes) -> bytes:
    assert len(root) == 32
    return TAG_HEAD + epoch.to_bytes(8, "big") + root


def hash_msg(x: bytes, y: bytes) -> bytes:
    assert len(y) == 32 and len(x) <= MAX_X
    return TAG_HASH + y + x


def sign(seckey: bytes, msg: bytes) -> bytes:
    return key.sign_schnorr(seckey, msg)


def verify(pub: bytes, sig: bytes, msg: bytes) -> bool:
    """BIP340 verification of a message of any length (what CHECKSIGFROMSTACK does)."""
    from test_framework.crypto import secp256k1

    if len(pub) != 32 or len(sig) != 64:
        return False
    P = secp256k1.GE.from_bytes_xonly(pub)
    r, s = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")
    if P is None or r >= secp256k1.FE.SIZE or s >= key.ORDER:
        return False
    e = int.from_bytes(key.TaggedHash("BIP0340/challenge", sig[:32] + pub + msg), "big") % key.ORDER
    R = secp256k1.GE.mul((s, secp256k1.G), (-e, P))
    return not R.infinity and R.y.is_even() and r == R.x


def template(outputs: list) -> bytes:
    """BIP119 template hash of the one-input transaction with exactly these outputs."""
    tx = CTransaction()
    tx.version = 2
    tx.vin = [CTxIn(COutPoint(0, 0), nSequence=SEQ)]
    tx.vout = outputs
    return tx.get_standard_template_hash(0)


def burn_output(amount: int) -> CTxOut:
    return CTxOut(amount, CScript([OP_RETURN, BURN_NOTE]))


def equiv_script(pid: bytes, t_burn: bytes) -> CScript:
    # witness: sig_b root_b sig_a root_a epoch
    return CScript([
        OP_SIZE, 8, OP_EQUALVERIFY, TAG_HEAD, OP_SWAP, OP_CAT,  # tag || epoch
        OP_TUCK, OP_SWAP, OP_SIZE, 32, OP_EQUALVERIFY, OP_DUP, OP_TOALTSTACK, OP_CAT,
        OP_ROT, OP_SWAP, pid, OP_CHECKSIGFROMSTACK, OP_VERIFY,  # sig_a over tag||epoch||root_a
        OP_SWAP, OP_SIZE, 32, OP_EQUALVERIFY,
        OP_DUP, OP_FROMALTSTACK, OP_EQUAL, OP_NOT, OP_VERIFY,  # root_b != root_a
        OP_CAT, pid, OP_CHECKSIGFROMSTACK, OP_VERIFY,  # sig_b over tag||epoch||root_b
        t_burn, OP_CHECKTEMPLATEVERIFY])


def lie_script(pid: bytes, t_burn: bytes) -> CScript:
    # witness: sig x y
    return CScript([
        OP_SIZE, 32, OP_EQUALVERIFY, OP_2DUP, OP_SWAP, OP_SHA256, OP_EQUAL, OP_NOT, OP_VERIFY,
        TAG_HASH, OP_SWAP, OP_CAT, OP_SWAP, OP_CAT,  # tag || y || x
        pid, OP_CHECKSIGFROMSTACK, OP_VERIFY,
        t_burn, OP_CHECKTEMPLATEVERIFY])


@dataclass(frozen=True)
class Bond:
    pid: bytes  # x-only key whose faults burn the coins
    own: bytes  # x-only key that can take them back
    amount: int  # sats locked
    delay: int = 144  # blocks between starting to unbond and withdrawing
    fee: int = FEE  # fixed fee of each covenant transaction

    # --- the unbonding output (second stage)
    def _stage2(self):
        t_burn = template([burn_output(self.amount - 2 * self.fee)])
        return taproot_construct(NUMS, [
            ("withdraw", CScript([self.delay, OP_CHECKSEQUENCEVERIFY, OP_DROP, self.own, OP_CHECKSIG])),
            ("equiv", equiv_script(self.pid, t_burn)),
            ("lie", lie_script(self.pid, t_burn))])

    # --- the bond output (first stage)
    def _stage1(self):
        t_burn = template([burn_output(self.amount - self.fee)])
        t_unbond = template([CTxOut(self.amount - self.fee, self._stage2().scriptPubKey)])
        return taproot_construct(NUMS, [
            ("unbond", CScript([self.own, OP_CHECKSIGVERIFY, t_unbond, OP_CHECKTEMPLATEVERIFY])),
            ("equiv", equiv_script(self.pid, t_burn)),
            ("lie", lie_script(self.pid, t_burn))])

    def info(self, stage: int = 1):
        return self._stage1() if stage == 1 else self._stage2()

    def spk(self, stage: int = 1) -> bytes:
        return bytes(self.info(stage).scriptPubKey)

    def address(self, stage: int = 1) -> str:
        return address.output_key_to_p2tr(self.info(stage).output_pubkey, main=True)

    def value(self, stage: int = 1) -> int:
        return self.amount if stage == 1 else self.amount - self.fee

    def stage_of(self, spk_hex: str):
        """1 or 2 if this scriptPubKey is the bond's or its unbonding output's, else None."""
        for st in (1, 2):
            if self.spk(st).hex() == spk_hex:
                return st
        return None

    # --- spends
    def _spend(self, outpoint: str, stage: int, leaf: str, stack: list, outputs: list, seq=SEQ):
        info = self.info(stage)
        lf = info.leaves[leaf]
        txid, vout = outpoint.split(":")
        tx = CTransaction()
        tx.version = 2
        tx.vin = [CTxIn(COutPoint(int(txid, 16), int(vout)), nSequence=seq)]
        tx.vout = outputs
        control = bytes([lf.version + info.negflag]) + info.internal_pubkey + lf.merklebranch
        wit = CTxInWitness()
        wit.scriptWitness.stack = stack + [bytes(lf.script), control]
        tx.wit.vtxinwit = [wit]
        return tx

    def slash_equivocation(self, outpoint: str, stage: int, epoch: int, root_a: bytes, sig_a: bytes,
                           root_b: bytes, sig_b: bytes) -> CTransaction:
        """Burn the coins: two signed heads of `pid` for one epoch. No key is needed."""
        assert root_a != root_b
        stack = [sig_b, root_b, sig_a, root_a, epoch.to_bytes(8, "big")]
        return self._spend(outpoint, stage, "equiv", stack, [burn_output(self.value(stage) - self.fee)])

    def slash_lie(self, outpoint: str, stage: int, x: bytes, y: bytes, sig: bytes) -> CTransaction:
        """Burn the coins: `pid` signed SHA256(x) = y and it is not. No key is needed."""
        return self._spend(outpoint, stage, "lie", [sig, x, y], [burn_output(self.value(stage) - self.fee)])

    def unbond(self, outpoint: str, own_seckey: bytes) -> CTransaction:
        """Stage 1 -> stage 2: the owner starts the unbonding delay."""
        out = CTxOut(self.amount - self.fee, CScript(self.spk(2)))
        tx = self._spend(outpoint, 1, "unbond", [], [out])
        lf = self.info(1).leaves["unbond"]
        spent = [CTxOut(self.amount, CScript(self.spk(1)))]
        h = TaprootSignatureHash(tx, spent, 0, 0, scriptpath=True, leaf_script=lf.script)
        tx.wit.vtxinwit[0].scriptWitness.stack.insert(0, sign(own_seckey, h))
        return tx

    def withdraw(self, outpoint: str, own_seckey: bytes, dest_spk: bytes) -> CTransaction:
        """Stage 2 -> the owner, once the delay has passed."""
        out = CTxOut(self.amount - 2 * self.fee, CScript(dest_spk))
        tx = self._spend(outpoint, 2, "withdraw", [], [out], seq=self.delay)
        lf = self.info(2).leaves["withdraw"]
        spent = [CTxOut(self.amount - self.fee, CScript(self.spk(2)))]
        h = TaprootSignatureHash(tx, spent, 0, 0, scriptpath=True, leaf_script=lf.script)
        tx.wit.vtxinwit[0].scriptWitness.stack.insert(0, sign(own_seckey, h))
        return tx


# --- tips: coins given to an identity. They are paid to a taproot output whose key is the identity's own key (the `own`
# of its bond, which is on the chain): nobody else can spend it, and the identity's node collects it into its wallet.
TIP_KEEP = 330  # sats that stay under the tip's note when the rest is collected (the note keeps an output of its own)


def tip_spk(own: bytes) -> bytes:
    assert len(own) == 32
    return bytes(CScript([1, own]))  # OP_1 <own>: segwit v1, the key itself is the output key


def tip_address(own: bytes) -> str:
    return address.output_key_to_p2tr(own, main=True)


def collect_tip(outpoint: str, value: int, own_seckey: bytes, own: bytes, dest_spk: bytes, fee: int = FEE) -> CTransaction:
    """A tip's coins to the receiver's wallet (key path, signed by its own key). The first TIP_KEEP sats, with the
    note on them, go back to the tip output's own script; the rest, less the fee, to dest_spk."""
    assert value > TIP_KEEP + fee
    txid, vout = outpoint.split(":")
    tx = CTransaction()
    tx.version = 2
    tx.vin = [CTxIn(COutPoint(int(txid, 16), int(vout)), nSequence=SEQ)]
    tx.vout = [CTxOut(TIP_KEEP, CScript(tip_spk(own))), CTxOut(value - TIP_KEEP - fee, CScript(dest_spk))]
    h = TaprootSignatureHash(tx, [CTxOut(value, CScript(tip_spk(own)))], 0, 0)
    wit = CTxInWitness()
    wit.scriptWitness.stack = [sign(own_seckey, h)]
    tx.wit.vtxinwit = [wit]
    return tx


def keypair(seed: bytes | None = None) -> tuple:
    sec = key.generate_privkey() if seed is None else modelhash.sha256(b"qwg/key" + seed)
    pub, _ = key.compute_xonly_pubkey(sec)
    return sec, pub


def tx_hex(tx: CTransaction) -> str:
    return tx.serialize_with_witness().hex()
