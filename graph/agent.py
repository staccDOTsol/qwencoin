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
        net.ord_wait(self.node)
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
        if msg["op"] in ("head", "attest", "post"):
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

    def post(self, text: str) -> dict:
        self.seq += 1
        return self.inscribe(P.post_msg(self.sec, self.pid, self.seq, text))

    def root(self) -> bytes:
        """Commitment to everything this identity has published so far."""
        h = bytes(32)
        for m in self.log:
            h = sha256(h + sha256(json.dumps(m, sort_keys=True).encode()))
        return h

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
