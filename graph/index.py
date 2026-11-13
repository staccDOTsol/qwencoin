"""The graph, read from a node's own ord and bitcoind.

Nothing here is trusted state: stakes are checked against the chain (the inscription must sit on
exactly the covenant output its message describes), signatures are verified, and every attested
hash is recomputed in the weights. Scores are functions of sats that are provably at risk.

Sybil resistance. An identity's weight is the sats that burn if it misbehaves: its own bonds
plus the vouches others have locked on it. Each locked sat sits in one output and counts once,
so splitting a budget across any number of identities, or vouching in circles, creates no
weight: an attacker's total weight is its locked capital, however many names it wears. For a
personal view, `trust` is the max flow over vouch edges from the viewer to the target, which is
capped by the vouches honest identities have placed on the attacker's side.

"Counts once" is enforced here, not assumed. A stake is an id or vouch inscription that
  - was born on its bond: its reveal transaction carries that one inscription and its first
    output is the bond output, with the bond's value. Two messages revealed into one output, or
    inscriptions made elsewhere and moved into a bond-shaped output together, would each claim
    the same sats; neither is a stake.
  - is the earliest of the inscriptions born that way with its signature. Stake messages are
    signed over a nonce, so a second inscription with the same signature is a replay of
    somebody's message onto other coins. It never counts, even after the first has left.
    (Getting ahead of an honest reveal means locking the full amount under the victim's key.)
  - still sits, alone, on the bond output or on the unbonding output the covenant forces.
All three are read from the chain, so every node's index agrees. An id is signed by the identity
key itself, and a bond that names its own fee, a dust stake or a one-block delay is not a
message at all (protocol.bond_of).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402

from . import covenant as C  # noqa: E402
from . import protocol as P  # noqa: E402
from .modelhash import sha256  # noqa: E402


@dataclass
class Stake:
    inscription: str
    kind: str  # "id" | "vouch"
    bond: C.Bond
    frm: bytes | None  # the voucher's identity (vouch only)
    name: str = ""
    number: int = 0  # ord's inscription number: the chain's order of inscriptions
    sig: bytes = b""
    born: bool = False  # its reveal put it, alone, on its bond output
    outpoint: str = ""
    stage: int | None = None
    status: str = "unknown"  # active | unbonding | slashed | exited | invalid | replay

    @property
    def live(self) -> bool:
        return self.status in ("active", "unbonding")

    @property
    def sats(self) -> int:
        return self.bond.value(self.stage) if self.live else 0


@dataclass
class Proof:
    kind: str  # "equiv" | "lie"
    pid: bytes
    data: dict


@dataclass
class Index:
    node: int
    seen: dict = field(default_factory=dict)  # inscription id -> parsed message or None
    stakes: dict = field(default_factory=dict)  # inscription id -> Stake
    heads: dict = field(default_factory=dict)  # (pid, epoch) -> {root: sig}
    attests: list = field(default_factory=list)  # (pid, x, y, sig, checked)
    posts: list = field(default_factory=list)
    hashes_checked: int = 0

    # ------------------------------------------------------------------ scan
    def refresh(self) -> "Index":
        net.ord_wait(self.node)
        page = 0
        while True:
            r = net.ord_get(self.node, f"/inscriptions/{page}")
            for iid in r["ids"]:
                if iid not in self.seen:
                    self._ingest(iid)
            if not r.get("more"):
                break
            page += 1
        first = {}  # signature -> the earliest inscription born on a bond with it
        for st in self.stakes.values():
            if st.born and (st.sig not in first or st.number < first[st.sig].number):
                first[st.sig] = st
        for st in self.stakes.values():
            if not st.born:
                st.stage, st.status = None, "invalid"
            elif first[st.sig] is not st:
                st.stage, st.status = None, "replay"
            else:
                self._locate(st)
        return self

    def _content(self, iid: str) -> bytes:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{net.ordport(self.node)}/content/{iid}", timeout=30) as r:
            return r.read()

    def _ingest(self, iid: str) -> None:
        m = P.parse(self._content(iid))
        self.seen[iid] = m
        if m is None:
            return
        pid = bytes.fromhex(m["id"] if m["op"] != "vouch" else m["to"])
        if m["op"] in ("id", "vouch"):
            frm = bytes.fromhex(m["from"]) if m["op"] == "vouch" else None
            bond = P.bond_of(m)
            number = net.ord_get(self.node, f"/inscription/{iid}")["number"]
            reveal = net.ord_get(self.node, f"/tx/{iid[:64]}")
            out0 = reveal["transaction"]["output"][0]
            born = (iid[64:] == "i0" and reveal["inscription_count"] == 1
                    and out0["script_pubkey"] == bond.spk(1).hex() and out0["value"] == bond.amount)
            self.stakes[iid] = Stake(iid, m["op"], bond, frm, m.get("name", ""), number,
                                     bytes.fromhex(m["sig"]), born)
        elif m["op"] == "head":
            self.heads.setdefault((pid, int(m["epoch"])), {})[bytes.fromhex(m["root"])] = bytes.fromhex(m["sig"])
        elif m["op"] == "attest":
            x, y = bytes.fromhex(m["x"]), bytes.fromhex(m["y"])
            ok = sha256(x) == y  # recomputed in this node's weights
            self.hashes_checked += 1
            self.attests.append((pid, x, y, bytes.fromhex(m["sig"]), ok))
        elif m["op"] == "post":
            # ord's inscription number is the chain's order of inscriptions: the order of the feed
            number = net.ord_get(self.node, f"/inscription/{iid}")["number"]
            self.posts.append((pid, int(m["seq"]), m["text"], number))

    def _locate(self, st: Stake) -> None:
        """Where a stake's inscription sits now (its confirmed location): on the bond, on the
        unbonding output, burned out of one of them by a slash, or gone with its owner."""
        if st.status == "slashed":
            return  # a burn is final
        info = net.ord_get(self.node, f"/inscription/{st.inscription}")
        outpoint = info["satpoint"].rsplit(":", 1)[0]
        if "burned" in info.get("charms", []):
            # slashed if the burn spent a bond output: the covenant leaves no other way to burn one
            tx = net.ord_get(self.node, f"/tx/{outpoint[:64]}")["transaction"]
            ptxid, pvout = tx["input"][0]["previous_output"].split(":")
            prev = net.ord_get(self.node, f"/tx/{ptxid}")["transaction"]["output"][int(pvout)]
            slashed = len(tx["input"]) == 1 and st.bond.stage_of(prev["script_pubkey"]) is not None
            st.stage, st.outpoint, st.status = None, outpoint, "slashed" if slashed else "exited"
            return
        out = net.ord_get(self.node, f"/output/{outpoint}")
        stage = st.bond.stage_of(out["script_pubkey"])
        if stage and out["value"] == st.bond.value(stage) and out["inscriptions"] == [st.inscription]:
            st.outpoint, st.stage = outpoint, stage
            st.status = "active" if stage == 1 else "unbonding"
        else:
            st.stage, st.status = None, "exited"

    def feed(self, count: int | None = None) -> list:
        """The signed posts, newest first: (identity, seq, text). Call refresh first."""
        out = [(pid, seq, text) for pid, seq, text, number in sorted(self.posts, key=lambda p: -p[3])]
        return out if count is None else out[:count]

    # ------------------------------------------------------------------ faults
    def proofs(self) -> list:
        """Every fault the chain's data proves: equivocations and false hash attestations."""
        out = []
        for (pid, epoch), roots in self.heads.items():
            if len(roots) >= 2:
                (ra, sa), (rb, sb) = sorted(roots.items())[:2]
                out.append(Proof("equiv", pid, {"epoch": epoch, "root_a": ra, "sig_a": sa, "root_b": rb, "sig_b": sb}))
        for pid, x, y, sig, ok in self.attests:
            if not ok:
                out.append(Proof("lie", pid, {"x": x, "y": y, "sig": sig}))
        return out

    def convicted(self) -> set:
        return {p.pid for p in self.proofs()}

    def slash_tx(self, st: Stake, proof: Proof):
        d = proof.data
        if proof.kind == "equiv":
            return st.bond.slash_equivocation(st.outpoint, st.stage, d["epoch"], d["root_a"], d["sig_a"],
                                              d["root_b"], d["sig_b"])
        return st.bond.slash_lie(st.outpoint, st.stage, d["x"], d["y"], d["sig"])

    # ------------------------------------------------------------------ scores
    def identities(self) -> dict:
        """pid -> name, for every identity with an id stake. The earliest id names it."""
        names = {}
        for s in sorted(self.stakes.values(), key=lambda s: s.number):
            if s.kind == "id" and s.status not in ("invalid", "replay"):
                names.setdefault(s.bond.pid, s.name)
        return names

    def weight(self) -> dict:
        """pid -> sats that burn if pid misbehaves (own bonds plus vouches placed on it)."""
        w = {pid: 0 for pid in self.identities()}
        for s in self.stakes.values():
            if s.live:
                w[s.bond.pid] = w.get(s.bond.pid, 0) + s.sats
        return w

    def edges(self) -> dict:
        """(from, to) -> live vouch sats."""
        e = {}
        for s in self.stakes.values():
            if s.kind == "vouch" and s.live:
                e[s.frm, s.bond.pid] = e.get((s.frm, s.bond.pid), 0) + s.sats
        return e

    def trust(self, src: bytes, dst: bytes) -> int:
        """Max flow of vouch stake from src to dst; 0 for a convicted identity."""
        if dst in self.convicted():
            return 0
        if src == dst:
            return self.weight().get(dst, 0)
        cap = dict(self.edges())
        flow = 0
        while True:
            prev, queue = {src: None}, [src]
            while queue and dst not in prev:
                u = queue.pop(0)
                for (a, b), c in cap.items():
                    if a == u and c > 0 and b not in prev:
                        prev[b] = a
                        queue.append(b)
            if dst not in prev:
                return flow
            path, v = [], dst
            while prev[v] is not None:
                path.append((prev[v], v))
                v = prev[v]
            push = min(cap[e] for e in path)
            for a, b in path:
                cap[a, b] -= push
                cap[b, a] = cap.get((b, a), 0) + push
            flow += push
