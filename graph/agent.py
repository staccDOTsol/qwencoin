"""A node's agent: its identity on the graph and what it does there.

Each agent drives one node (qwencoind + ord). It bonds, posts, attests hashes it computed in
its weights, publishes a signed head per epoch, vouches for peers, and audits everyone: it
recomputes every attested hash in its own weights and checks every head for equivocation. A
fault it finds becomes a transaction that burns the offender's bond and every vouch placed on
the offender. The proof is all it needs; consensus does the rest.
"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402

from . import covenant as C  # noqa: E402
from . import protocol as P  # noqa: E402
from .index import Index  # noqa: E402
from .modelhash import sha256  # noqa: E402


@dataclass
class Agent:
    node: int
    name: str
    index: Index = None
    seq: int = 0
    log: list = field(default_factory=list)  # what this identity has published, in order
    secret: bytes = b""  # a real identity's keys come from a secret kept on its node (tools/chain.py);
    # with none (the tests' named characters) they come from the name alone

    def __post_init__(self) -> None:
        seed = self.secret + self.name.encode()
        self.sec, self.pid = C.keypair(seed)  # identity key: signs heads, hashes
        self.own_sec, self.own = C.keypair(seed + b"/own")  # takes stake back
        self.index = Index(self.node)

    # ------------------------------------------------------------------ publishing
    def inscribe(self, msg: dict, dest: str | None = None, postage: int | None = None) -> dict:
        """Inscribe a message. With no destination it goes onto the identity's own key (the address a tip for it is
        paid to), on the smallest output there is: what an identity publishes is its own, and it stays out of the
        node's wallet. (ord's wallet looks at every output it holds for every command: with a head and an
        attestation an epoch landing in it, a wallet took minutes to answer after a day.)"""
        net.ord_wait(self.node)
        if dest is None:
            dest, postage = C.tip_address(self.own), postage or C.TIP_KEEP
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(msg, f, separators=(",", ":"))
        # --no-backup: ord's recovery-key import makes the node rescan the last two hours of
        # blocks, which on a chain this young is every block, once per inscription
        args = ["inscribe", "--fee-rate", "2", "--no-backup", "--file", f.name]
        if dest:
            args += ["--destination", dest]
        if postage:
            args += ["--postage", f"{postage}sat"]
        r = net.ord_wallet(self.node, *args)
        Path(f.name).unlink()
        if msg["op"] in ("head", "attest", "post", "tip"):
            self.log.append(msg)
        return r["inscriptions"][0]

    def bond(self, amount: int, delay: int = 144) -> C.Bond:
        b = C.Bond(self.pid, self.own, amount, delay)
        self.inscribe(P.id_msg(self.sec, b, self.name, os.urandom(8)), b.address(), amount)
        return b

    def vouch(self, to: bytes, amount: int, delay: int = 144) -> C.Bond:
        b = C.Bond(to, self.own, amount, delay)
        self.inscribe(P.vouch_msg(self.sec, self.pid, b, os.urandom(8)), b.address(), amount)
        return b

    def attest(self, x: bytes, y: bytes | None = None) -> dict:
        """Publish SHA256(x) = y. y is computed in the weights unless the caller overrides it."""
        return self.inscribe(P.attest_msg(self.sec, self.pid, x, sha256(x) if y is None else y))

    def post(self, text: str, re: str | None = None, att=()) -> dict:
        """Publish signed text; re: the inscription id of the post it answers; att: ids of inscriptions it carries."""
        self.seq += 1
        return self.inscribe(P.post_msg(self.sec, self.pid, self.seq, text, re, att))

    def tip(self, to: bytes, sats: int, re: str | None = None) -> dict:
        """Give sats to identity `to` (for the post `re`, if given): the signed note, inscribed onto an output of
        exactly those sats that only `to`'s own key can spend."""
        own = self.index.refresh().owns().get(to)
        if own is None:
            raise RuntimeError("that identity has no bond on the graph: there is no key of its own to pay to")
        return self.inscribe(P.tip_msg(self.sec, self.pid, to, sats, os.urandom(8), re), C.tip_address(own), sats)

    def collect(self, dest_address: str) -> list:
        """The coins given to this identity that still sit where they were paid: each note's coins to the node's
        wallet (the note stays, on TIP_KEEP sats, where it is). Whatever was paid to this identity's own key under a
        note is collected, a note that does not count as a tip included (the coins are this identity's all the
        same). Returns [(tip inscription, sats, txid)]."""
        ix, done = self.index.refresh(), []
        spk = bytes.fromhex(net.cli(self.node, "getaddressinfo", dest_address, wallet="ord")["scriptPubKey"])
        gone = self.__dict__.setdefault("_collected", set())  # notes whose coins are known to be collected
        for t in ix.tips:
            if t["to"] != self.pid or not t.get("paid") or t["id"] in gone:
                continue
            info = net.ord_get(self.node, f"/inscription/{t['id']}")
            outpoint = info["satpoint"].rsplit(":", 1)[0]
            out = net.ord_get(self.node, f"/output/{outpoint}")
            if out["script_pubkey"] != C.tip_spk(self.own).hex() or out["value"] <= C.TIP_KEEP + C.FEE:
                gone.add(t["id"])  # collected (what is left is the note's own output)
                continue
            if out.get("spent"):
                continue  # on its way: the collecting transaction is not in a block yet
            tx = C.collect_tip(outpoint, out["value"], self.own_sec, self.own, spk)
            try:
                txid = net.cli(self.node, "sendrawtransaction", C.tx_hex(tx))
            except RuntimeError as e:
                if any(w in str(e) for w in ("already", "missing", "spent", "conflict")):
                    continue  # on its way already
                raise
            done.append((t["id"], out["value"] - C.TIP_KEEP - C.FEE, txid))
        return done

    def root(self) -> bytes:
        """Commitment to everything this identity has published so far."""
        h = bytes(32)
        for m in self.log:
            h = sha256(h + sha256(json.dumps(m, sort_keys=True).encode()))
        return h

    def signed(self, epoch: int) -> bool:
        """This identity has signed a head for that epoch (its own record on disk says so)."""
        book = net.datadir(self.node) / f"qwg-heads-{self.name}.json"
        return book.exists() and str(epoch) in json.loads(book.read_text())

    def head(self, epoch: int | None = None, root: bytes | None = None, equivocate: bool = False) -> dict:
        """Sign this epoch's head. An honest agent signs one root per epoch and remembers it on
        disk: a second, different signature for the same epoch is all it takes to burn its stake."""
        if epoch is None:
            epoch = P.epoch_of(net.cli(self.node, "getblockcount"))
        root = self.root() if root is None else root
        book = net.datadir(self.node) / f"qwg-heads-{self.name}.json"
        signed = json.loads(book.read_text()) if book.exists() else {}
        prior = signed.get(str(epoch))
        if prior is not None and prior != root.hex() and not equivocate:
            raise RuntimeError(f"{self.name} already signed a head for epoch {epoch}; refusing to equivocate")
        if prior is None:
            signed[str(epoch)] = root.hex()
            book.write_text(json.dumps(signed))
        return self.inscribe(P.head_msg(self.sec, self.pid, epoch, root))

    # ------------------------------------------------------------------ auditing
    def audit(self) -> list:
        """Slash every live stake on every identity the chain's data convicts. Returns the
        (kind, offender, stake inscription, txid) of each burn this node broadcast."""
        ix = self.index.refresh()
        done = []
        for proof in ix.proofs():
            for st in ix.stakes.values():
                if st.bond.pid != proof.pid or not st.live:
                    continue
                tx = ix.slash_tx(st, proof)
                try:
                    txid = net.cli(self.node, "sendrawtransaction", C.tx_hex(tx), 0, st.bond.amount / 1e8)
                except RuntimeError as e:
                    if any(w in str(e) for w in ("already", "missing", "spent", "conflict", "insufficient fee")):
                        continue  # another auditor got there first, or the stake is mid-move:
                    raise         # the next audit finds it at its new location
                done.append((proof.kind, proof.pid, st.inscription, txid))
        return done

    # ------------------------------------------------------------------ leaving
    def stake(self, kind: str, to: bytes | None = None):
        self.index.refresh()
        for st in self.index.stakes.values():
            if st.kind == kind and st.bond.own == self.own and st.live and (to is None or st.bond.pid == to):
                return st
        return None

    def unbond(self, st) -> str:
        assert st.stage == 1, "only a bond that is not already unbonding can start to"
        return net.cli(self.node, "sendrawtransaction", C.tx_hex(st.bond.unbond(st.outpoint, self.own_sec)))

    def withdraw(self, st, dest_address: str) -> str:
        assert st.stage == 2, "withdraw spends the unbonding output"
        spk = bytes.fromhex(net.cli(self.node, "getaddressinfo", dest_address, wallet="ord")["scriptPubKey"])
        return net.cli(self.node, "sendrawtransaction", C.tx_hex(st.bond.withdraw(st.outpoint, self.own_sec, spk)))

    def reclaim(self, dest_address: str) -> list:
        """What this identity is taking back, once it may have it. A stake of its own (its bond, or coins it locked
        on another identity) that has been unbonding for its whole delay is withdrawn, the covenant's last step,
        onto the identity's own key: the stake's inscription stays there. What a withdrawal has put there then goes
        on to the node's wallet, all but the TIP_KEEP sats the inscription sits on. Two transactions, in two blocks.
        (Until this was done for it, a stake somebody had started to take back stayed "being taken back" for good:
        nobody came for it after the delay.) Returns [(step, stake inscription, sats, txid)], step "withdrawn" or
        "collected"."""
        ix, done, own_spk = self.index.refresh(), [], C.tip_spk(self.own)
        spk = bytes.fromhex(net.cli(self.node, "getaddressinfo", dest_address, wallet="ord")["scriptPubKey"])
        for st in list(ix.stakes.values()):
            if st.bond.own != self.own:
                continue
            if st.live and st.stage == 2:
                out = net.cli(self.node, "gettxout", *st.outpoint.split(":"))  # (its mempool counts: a withdrawal on its way is not sent again)
                if not out or out["confirmations"] < st.bond.delay:
                    continue  # still inside its delay
                tx, step, sats = st.bond.withdraw(st.outpoint, self.own_sec, own_spk), "withdrawn", st.bond.amount - 2 * st.bond.fee
            elif st.status == "exited" and st.at and not st.burned:
                out = net.cli(self.node, "gettxout", *st.at.split(":"))
                value = round(out["value"] * 1e8) if out else 0
                if not out or out["scriptPubKey"]["hex"] != own_spk.hex() or value <= C.TIP_KEEP + C.FEE:
                    continue  # not on this identity's own key, or collected already
                tx, step, sats = C.collect_tip(st.at, value, self.own_sec, self.own, spk), "collected", value - C.TIP_KEEP - C.FEE
            else:
                continue
            try:
                txid = net.cli(self.node, "sendrawtransaction", C.tx_hex(tx))
            except RuntimeError as e:
                if any(w in str(e) for w in ("already", "missing", "spent", "conflict", "non-BIP68-final")):
                    continue  # on its way already, or a block short of its delay
                raise
            done.append((step, st.inscription, sats, txid))
        return done
