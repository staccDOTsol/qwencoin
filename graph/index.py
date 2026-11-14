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

What a node has verified it remembers. Reading an inscription (its content, its signature, for an
attestation the hash recomputed in the weights, for a stake or a tip the transaction it was born
in) is done once; the result for every inscription buried KEEP_DEPTH blocks or more is kept beside
the node (qwg-index.json) and taken up again by the next process, if the chain still has the block
it was kept under and the code that read it is the code here. Without that, every start of an
agent, a model's loop or the pages verified the whole history again: a quarter of an hour with a
thousand inscriptions, and longer with every block. Where a stake's inscription sits now is never
kept: it is looked up at every refresh. QWG_INDEX_KEEP=0: nothing is kept or taken up.
"""
import json, os, sys, time, zlib
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


KEEP_DEPTH = 24  # blocks under which what was read of an inscription is kept between runs


def reading() -> str:
    """A fingerprint of the code that reads an inscription: what another version read is not taken up."""
    c = 0
    for n in ("index.py", "protocol.py", "covenant.py"):
        c = zlib.crc32((Path(__file__).parent / n).read_bytes(), c)
    return f"{c:08x}"


@dataclass
class Index:
    node: int
    seen: dict = field(default_factory=dict)  # inscription id -> parsed message or None
    records: dict = field(default_factory=dict)  # inscription id -> what reading it gave (_read), as it is kept
    taken_up: int = 0  # how many of them came from the file kept beside the node
    _kept: list = field(default_factory=lambda: [False, 0, 0.0])  # looked for the file; records in it; when it was last written
    stakes: dict = field(default_factory=dict)  # inscription id -> Stake
    heads: dict = field(default_factory=dict)  # (pid, epoch) -> {root: sig}
    attests: list = field(default_factory=list)  # (pid, x, y, sig, checked)
    posts: list = field(default_factory=list)  # (pid, seq, text, number, inscription id, id of the post it answers or None, ids it carries)
    # {"id", "number", "from", "to", "sats", "re", "ok", "paid"}: ok once it is seen to be paid as it says; paid: the sats its
    # note was born on at the receiver's own key (whatever the note says: they are the receiver's to collect)
    tips: list = field(default_factory=list)
    hashes_checked: int = 0

    # ------------------------------------------------------------------ scan
    def refresh(self) -> "Index":
        net.ord_wait(self.node)
        if not self._kept[0]:
            self._kept[0] = True
            self._take_up()
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
        # A tip counts if it was paid: its note was born, alone, on an output of exactly its sats that only the
        # receiver's own key can spend, and it is the first note with its signature born that way. (The receiver's
        # own key is the `own` of its bond, so this is decided once the receiver's bond is known.)
        owns, seen = self.owns(), set()
        for t in sorted(self.tips, key=lambda t: t["number"]):
            if t["ok"] is None and t["to"] in owns:
                reveal = net.ord_get(self.node, f"/tx/{t['id'][:64]}")
                out0 = reveal["transaction"]["output"][0]
                t["paid"] = out0["value"] if t["id"][64:] == "i0" and out0["script_pubkey"] == C.tip_spk(owns[t["to"]]).hex() else 0
                t["ok"] = bool(t["paid"] == t["sats"] and reveal["inscription_count"] == 1)
                self.records[t["id"]].update(paid=t["paid"], tip_ok=t["ok"])  # as read from the chain (a copy is told apart below)
            sig = self.seen[t["id"]]["sig"]
            if t["ok"] and sig in seen:
                t["ok"] = False  # the same signed note inscribed a second time
            if t["ok"]:
                seen.add(sig)
        self._keep()
        return self

    # ------------------------------------------------------------------ what was read, kept between runs
    def _file(self) -> Path:
        return net.datadir(self.node) / "qwg-index.json"

    def _take_up(self) -> None:
        """What an earlier process read of the inscriptions that were buried then, if the chain still has the block
        it was kept under."""
        if os.environ.get("QWG_INDEX_KEEP") == "0":
            return
        try:
            k = json.loads(self._file().read_text())
            if k["reading"] != reading() or net.cli(self.node, "getblockhash", k["anchor"][0]) != k["anchor"][1]:
                return
            for rec in k["records"]:
                if rec["id"] not in self.seen:
                    self._apply(rec)
            self.taken_up = self._kept[1] = len(k["records"])
        except Exception:  # noqa: BLE001  (no file, another chain, a block that is not there: everything is read again)
            self.seen.clear(), self.records.clear(), self.stakes.clear(), self.heads.clear(), self.attests.clear()
            self.posts.clear(), self.tips.clear()
            self.hashes_checked = self.taken_up = 0

    def _keep(self) -> None:
        """Write what was read of every inscription buried KEEP_DEPTH blocks or more, when there is more of it than
        the file has (at most every two minutes)."""
        if os.environ.get("QWG_INDEX_KEEP") == "0" or time.time() - self._kept[2] < 120:
            return
        try:
            anchor = net.cli(self.node, "getblockcount") - KEEP_DEPTH
            if anchor < 1:
                return
            deep = sorted((r for r in self.records.values() if r.get("height") is not None and r["height"] <= anchor),
                          key=lambda r: (r["number"], r["id"]))
            if len(deep) <= self._kept[1]:
                return
            self._kept[2] = time.time()
            try:  # another process of this node may have written more in the meantime
                if len(json.loads(self._file().read_text()).get("records", [])) >= len(deep):
                    return
            except Exception:  # noqa: BLE001
                pass
            tmp = self._file().with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"reading": reading(), "anchor": [anchor, net.cli(self.node, "getblockhash", anchor)], "records": deep},
                                      separators=(",", ":")))
            tmp.replace(self._file())
            self._kept[1] = len(deep)
        except Exception:  # noqa: BLE001  (the node is busy, the disk is full: nothing is kept this time)
            pass

    def _content(self, iid: str) -> bytes:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{net.ordport(self.node)}/content/{iid}", timeout=30) as r:
            return r.read()

    def _ingest(self, iid: str) -> None:
        self._apply(self._read(iid))

    def _read(self, iid: str) -> dict:
        """One inscription, read and verified: {"id", "number", "height", "m": the message or None} and, for a stake,
        "born" (its reveal put it, alone, on its bond output), for an attestation, "ok" (the hash, recomputed in
        this node's weights, is the one it claims)."""
        m = P.parse(self._content(iid))
        info = net.ord_get(self.node, f"/inscription/{iid}")
        rec = {"id": iid, "number": info["number"], "height": info.get("height"), "m": m}
        if m is None:
            return rec
        if m["op"] in ("id", "vouch"):
            bond = P.bond_of(m)
            reveal = net.ord_get(self.node, f"/tx/{iid[:64]}")
            out0 = reveal["transaction"]["output"][0]
            rec["born"] = bool(iid[64:] == "i0" and reveal["inscription_count"] == 1
                               and out0["script_pubkey"] == bond.spk(1).hex() and out0["value"] == bond.amount)
        elif m["op"] == "attest":
            rec["ok"] = bool(sha256(bytes.fromhex(m["x"])) == bytes.fromhex(m["y"]))  # recomputed in this node's weights
        return rec

    def _apply(self, rec: dict) -> None:
        """Take what was read of an inscription into the index (nothing is asked of the node here)."""
        iid, m, number = rec["id"], rec["m"], rec["number"]
        self.records[iid] = rec
        self.seen[iid] = m
        if m is None:
            return
        pid = bytes.fromhex(m["id"] if m["op"] != "vouch" else m["to"])
        if m["op"] in ("id", "vouch"):
            frm = bytes.fromhex(m["from"]) if m["op"] == "vouch" else None
            self.stakes[iid] = Stake(iid, m["op"], P.bond_of(m), frm, m.get("name", ""), number, bytes.fromhex(m["sig"]), rec["born"])
        elif m["op"] == "head":
            self.heads.setdefault((pid, int(m["epoch"])), {})[bytes.fromhex(m["root"])] = bytes.fromhex(m["sig"])
        elif m["op"] == "attest":
            self.hashes_checked += 1
            self.attests.append((pid, bytes.fromhex(m["x"]), bytes.fromhex(m["y"]), bytes.fromhex(m["sig"]), rec["ok"]))
        elif m["op"] == "tip":
            self.tips.append({"id": iid, "number": number, "from": pid, "to": bytes.fromhex(m["to"]), "sats": m["sats"], "re": m.get("re"),
                              "ok": rec.get("tip_ok"), "paid": rec.get("paid", 0)})
        elif m["op"] == "post":
            # ord's inscription number is the chain's order of inscriptions: the order of the feed
            self.posts.append((pid, int(m["seq"]), m["text"], number, iid, m.get("re"), tuple(m.get("att", []))))

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
        out = [(p[0], p[1], p[2]) for p in sorted(self.posts, key=lambda p: -p[3])]
        return out if count is None else out[:count]

    def threads(self) -> list:
        """The posts as conversations, in the chain's order: [{"root": inscription id, "posts": [post, ..]}], a post
        being {"id", "pid", "seq", "text", "number", "re", "att"}. A post answers another only if that one is a post earlier
        on the chain; an answer to anything else stands as the start of a conversation of its own (re None there).
        The conversations are ordered by their latest post, the one written in last at the end. Call refresh first."""
        by, root, out = {}, {}, {}
        for pid, seq, text, number, iid, re, att in sorted(self.posts, key=lambda p: p[3]):
            if iid in by:
                continue  # one inscription, one post
            re = re if re in by else None
            by[iid] = {"id": iid, "pid": pid, "seq": seq, "text": text, "number": number, "re": re, "att": list(att)}
            root[iid] = root[re] if re else iid
            out.setdefault(root[iid], []).append(by[iid])
        return sorted(({"root": r, "posts": ps} for r, ps in out.items()), key=lambda t: t["posts"][-1]["number"])

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

    def owns(self) -> dict:
        """pid -> the identity's own key (the `own` of its earliest id stake): where a tip for it is paid."""
        out = {}
        for s in sorted(self.stakes.values(), key=lambda s: s.number):
            if s.kind == "id" and s.status not in ("invalid", "replay"):
                out.setdefault(s.bond.pid, s.bond.own)
        return out

    def given(self) -> list:
        """The tips that count, in the chain's order."""
        return sorted((t for t in self.tips if t["ok"]), key=lambda t: t["number"])

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
