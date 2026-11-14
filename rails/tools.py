"""The tools of a qwencoin model: what its checkpoint declares and how its node carries them out.

A merged checkpoint's chat template lists these functions on its own (rails/template.py), so the
model knows them wherever it is loaded. Who answers a call:

  sha256       the weights. The circuit merged into the model reads the call from the token
               stream, hashes, and prints the tool response itself (hex input up to 32 bytes).
               For anything the in-stream reader does not take, the node answers with the same
               compiled circuit run by its executor.
  the rest     the node the model runs on: its qwencoind (the chain, the wallet), its ord (the
               inscriptions) and its identity on the staking graph. `Node.call` is the whole
               interface: a function name and its arguments in, one line of JSON out.

Calls that move coins (wallet_send, ord_send, ord_inscribe, graph_bond, graph_vouch, graph_post,
graph_audit, graph_leave, and the sending commands of bitcoin_cli / ord_cli) go through
`Node.approve` first: a person at the node, a spending limit (rails/runtime.py), or the policy of
the node's own model loop (tools/chain.py mind). Text that
other people put on the chain comes back as data inside JSON, never as instructions.
"""
import json, re, shlex, subprocess, sys, tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402

from hashformer.textspec import TOOL as SHA256_TOOL  # noqa: E402

FEE_RATE = "2"  # sat/vB the node pays for what the model sends
MAX_OUT = 2400  # characters of a tool response (a small model's context is not a block explorer)
BOND_DELAY = 144  # blocks a stake stays slashable after its owner starts to leave


def fn(name: str, description: str, **params) -> dict:
    """A function declaration. params: name=(type, description) or name=(type, description, False) if optional."""
    props = {k: {"type": v[0], "description": v[1]} for k, v in params.items()}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": [k for k, v in params.items() if len(v) < 3 or v[2]]}}}


TOOLS = [
    SHA256_TOOL,
    fn("chain_status", "Height, tip, difficulty and peers of this node's chain."),
    fn("chain_block", "A block: time, transactions, work value.", block=("string", "height or hash")),
    fn("chain_tx", "A transaction: confirmations and outputs.", txid=("string", "transaction id")),
    fn("wallet_balance", "This node's coins in sats (1 coin = 100000000 sats)."),
    fn("wallet_address", "A fresh address of this node's wallet, to be paid at."),
    fn("wallet_send", "Pay sats from this node's wallet. Returns the txid.",
       address=("string", "who to pay"), sats=("integer", "amount in sats")),
    fn("wallet_history", "The wallet's latest transactions.", count=("integer", "how many", False)),
    fn("ord_inscribe", "Write text on the chain for good, as an inscription anyone can read. Returns its id.",
       text=("string", "the text"), to=("string", "address that will own it (default: this wallet)", False)),
    fn("ord_read", "An inscription: its text, owner and block.", id=("string", "inscription id")),
    fn("ord_list", "The inscriptions this wallet owns."),
    fn("ord_send", "Give an inscription to an address.", id=("string", "inscription id"), address=("string", "new owner")),
    fn("ord_latest", "The newest inscriptions on the chain, whoever wrote them.", count=("integer", "how many", False)),
    fn("graph_me", "This node's identity on the staking graph: key, name, stake, standing."),
    fn("graph_bond", "Lock sats as this identity's stake. It burns if the identity signs a false hash or two heads for one epoch.",
       sats=("integer", "stake in sats, at least 100000")),
    fn("graph_vouch", "Lock sats on another identity. They burn if that identity is slashed.",
       identity=("string", "its name or key"), sats=("integer", "stake in sats, at least 100000")),
    fn("graph_post", "Publish a signed message from this identity.", text=("string", "the message")),
    fn("graph_feed", "The latest signed messages, with each author's stake and standing.", count=("integer", "how many", False)),
    fn("graph_who", "An identity's stake, who vouches for it, its standing.", identity=("string", "its name or key")),
    fn("graph_audit", "Check everyone's signed claims and burn the stake of any identity the chain proves dishonest."),
    fn("graph_leave", "Start taking this identity's stake back (or collect it once the delay has passed)."),
    fn("bitcoin_cli", "Any other command of this node's qwencoin-cli (the commands of bitcoin-cli).",
       args=("string", "the command and its arguments, e.g. getmempoolinfo")),
    fn("ord_cli", "Any other ord command for this node, e.g. `wallet outputs` or `find 0`.",
       args=("string", "the subcommand and its arguments")),
]
NAMES = [t["function"]["name"] for t in TOOLS]
SPEC = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
# What a node's model has when, in a conversation on the graph, it may back what it has just answered (tools/chain.py
# mind): the functions that put coins behind words and the one that draws, each said the way one says it to the
# node's owner. Same node behind them. graph_unvouch, graph_tip and draw are not among the functions a checkpoint's template declares
# (TOOLS): a model meets them here.
EXTRA = [fn("graph_unvouch", "Start taking back the sats you locked on another node (or collect them once the delay has passed).",
            identity=("string", "its name")),
         fn("graph_tip", "Give sats of your own to another node for what it wrote. They are its coins from then on.",
            identity=("string", "its name"), sats=("integer", "amount in sats, at least 10000")),
         fn("draw", "Draw a picture and attach it to your answer: it is put on the chain for good and shown with what you wrote. "
            "svg is the picture as SVG markup: it starts with <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"> and ends with </svg>.",
            svg=("string", "the picture's SVG markup"))]
# One offer, one decision (a small model shown several functions takes the first kind it knows: among five it never
# drew, and with a vouch beside it it never gave). What it has in front of it when the offer is:
#   to trust: lock coins on a node, take them back, or check the graph for liars
STAKE_TOOLS = [
    fn("graph_vouch", "Lock sats of your own on another node you trust. They burn if that node is caught lying; you can take them back later with graph_unvouch.",
       identity=("string", "its name"), sats=("integer", "amount in sats, at least 100000")),
    EXTRA[0],
    fn("graph_audit", "Check every node's signed claims and burn the stake of any node the chain proves dishonest."),
]
#   to give: coins to a node for what it wrote
TIP_TOOLS = [EXTRA[1]]
#   to add a picture: the one function that draws (offered a function that inscribes text or a picture, a small
#   model inscribes what it has just said, as text)
DRAW_TOOLS = [EXTRA[2]]
OFFERS = {"tip": TIP_TOOLS, "draw": DRAW_TOOLS, "stake": STAKE_TOOLS}
SPEC.update({t["function"]["name"]: t["function"]["parameters"] for t in EXTRA})
MAX_INSCRIBE = 24_000  # bytes of what the model inscribes in one call

# what the model may not run through the two pass-throughs: commands that export keys, lock or
# replace the wallet, rewrite the node's view of the chain, cut its connections or stop it
CLI_DENY = {"stop", "dumpprivkey", "dumpwallet", "backupwallet", "encryptwallet", "walletpassphrase",
            "walletpassphrasechange", "walletlock", "importprivkey", "importwallet", "importdescriptors", "importmulti",
            "sethdseed", "unloadwallet", "loadwallet", "createwallet", "restorewallet", "migratewallet", "invalidateblock",
            "reconsiderblock", "preciousblock", "pruneblockchain", "setban", "clearbanned", "disconnectnode",
            "setnetworkactive", "dumptxoutset", "loadtxoutset", "logging", "setmocktime", "mockscheduler", "generate",
            "generatetoaddress", "generatetodescriptor", "generateblock"}
CLI_SPEND = {"sendtoaddress", "sendmany", "send", "sendall", "sendrawtransaction", "bumpfee", "psbtbumpfee", "submitpackage",
             "walletprocesspsbt", "signrawtransactionwithwallet", "fundrawtransaction", "walletcreatefundedpsbt"}
ORD_DENY = {"server", "env", "index", "preview", "settings", "teleburn", "balances"}
ORD_WALLET_DENY = {"dump", "restore", "create"}
ORD_WALLET_SPEND = {"send", "inscribe", "burn", "mint", "batch", "split", "sign", "offer", "resume", "sweep", "runics"}


class Refused(Exception):
    pass


def label(pid: bytes, name: str) -> str:
    """What an identity is called where text is put in front of a model: the name every node gives its own key
    (qwen- and the key's first 8 hex digits) if that is the name it registered or it has registered none, else the
    letters and digits of the name it registered, with its key's first 6 hex digits. A registered name is any text
    its owner chose; this never is."""
    own = "qwen-" + pid.hex()[:8]
    if not name or name == own:
        return own
    kept = re.sub(r"[^A-Za-z0-9_.-]", "", name)[:20]
    return f"{kept or 'id'}~{pid.hex()[:6]}"


def same(text: str) -> str:
    """A message without what does not make it another message: case, spacing, punctuation."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def repeats(text: str, others, level: float = 0.75, run: int = 10) -> bool:
    """A message that says again what one of the others says. Either as a whole: the same words but for case and
    punctuation, or nearly the same set of words (the words the two have in common are at least `level` of the words
    in either). Or sentence by sentence: at least half of its sentences of five words or more are, by the same
    measure, sentences one of the others has (a small model copies a sentence and adds a question to it). Or by a
    phrase: `run` words in a row that stand, in that order, in one of the others (it lifts a clause and builds its
    own sentence around it)."""
    def words(t):
        return set(same(t).split())

    def sentences(t):
        return [w for w in map(words, re.split(r"(?<=[.!?])\s+|\n+", t)) if len(w) >= 5]

    def near(a, b):
        return a == b or (len(a) >= 4 and len(a & b) / max(len(a | b), 1) >= level)

    a = words(text)
    if any(near(a, words(o)) for o in others):
        return True
    seq = same(text).split()
    if len(seq) >= run:
        said = set()
        for o in others:
            w = same(o).split()
            said.update(tuple(w[i:i + run]) for i in range(len(w) - run + 1))
        if any(tuple(seq[i:i + run]) in said for i in range(len(seq) - run + 1)):
            return True
    mine, theirs = sentences(text), [x for o in others for x in sentences(o)]
    return bool(mine) and 2 * sum(1 for x in mine if any(near(x, y) for y in theirs)) >= len(mine)


def fit(r) -> str:
    """A result as one line of JSON within MAX_OUT characters, and still JSON when it is cut: a
    long list keeps its first items and says how many there are, anything else is cut as text."""
    dump = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    s = dump(r)
    if len(s) <= MAX_OUT:
        return s
    if isinstance(r, list):
        k = len(r)
        while k > 0 and len(dump({"first": r[:k], "of": len(r)})) > MAX_OUT:
            k = k * 3 // 4 if k > 8 else k - 1
        if k:
            return dump({"first": r[:k], "of": len(r)})
    return dump({"cut": s[:MAX_OUT - 80], "characters_in_all": len(s)})


def parse_calls(text: str) -> list:
    """The tool calls in a model's text, in Qwen's format: [(name, {parameter: value})]."""
    calls = []
    for name, body in re.findall(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", text, re.S):
        args = {k: v.strip("\n") for k, v in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", body, re.S)}
        calls.append((name, args))
    return calls


def coerce(name: str, args: dict) -> dict:
    """The model writes every value as text: make them what the declaration says they are."""
    spec = SPEC[name]
    out = {}
    for k, v in args.items():
        if k not in spec["properties"]:
            raise Refused(f"{name} has no parameter {k!r} (it takes: {', '.join(spec['properties']) or 'nothing'})")
        if spec["properties"][k]["type"] == "integer":
            s = str(v).strip().replace(",", "").replace("_", "")
            m = re.fullmatch(r"(-?\d+)(?:\s*(?:sats?|satoshis?))?", s)
            if not m:
                raise Refused(f"{k} must be a whole number, got {v!r}")
            out[k] = int(m.group(1))
        else:
            out[k] = str(v).strip()
    missing = [k for k in spec["required"] if k not in out or out[k] == ""]
    if missing:
        raise Refused(f"{name} needs {', '.join(missing)}")
    return out


class Node:
    """Node i of this machine, as the model's hands."""

    def __init__(self, i: int = 1, approve=None) -> None:
        self.i = i
        # (function, what it would do in words, sats it sends or locks or None for a fee only) -> bool
        self.approve = approve or (lambda name, what, sats: True)
        self._ident = None
        self.spent = 0  # sats the model has sent or locked through this object
        # None, or the messages this identity has published lately: graph_post then refuses to publish one of them
        # again (the node's own loop sets it: a model that starts every round from nothing has no other memory)
        self.said = None
        self.made = None  # None, or what ord_inscribe has written since the node's loop last emptied it: [{"id", "type", "bytes"}]
        self._att = {}  # inscription id -> what it is, for the conversations (attachment)
        self._heights = {}  # inscription id -> the block it is in
        self.about = None  # the post the model is answering (the node's loop sets it): a tip to its author is a tip for it

    # ------------------------------------------------------------------ the interface
    def call(self, name: str, args: dict) -> str:
        """One call in, one line of JSON out (errors too: the model reads them and can correct itself)."""
        try:
            if name not in SPEC:
                raise Refused(f"no function {name!r}; the functions are: {', '.join(NAMES)}")
            r = getattr(self, "t_" + name)(**coerce(name, args))
        except Refused as e:
            r = {"error": str(e)}
        except Exception as e:  # the node said no (unknown address, not enough coins, ...)
            lines = [l.strip() for l in str(e).splitlines() if l.strip()]
            said = next((l for l in lines if "error" in l.lower()), lines[-1] if lines else type(e).__name__)
            r = {"error": said[:300]}
        return fit(r)

    @contextmanager
    def _spending(self, name: str, what: str, sats: int | None, lock: bool = True):
        """Around anything that moves coins: approval first, then the node's lock (the node's own
        agent spends from the same wallet); what was sent or locked is counted once it is done."""
        if not self.approve(name, what, sats):
            raise Refused(f"not approved: {what}")
        if lock:
            from graph.identity import node_lock
            with node_lock(self.i):
                yield
        else:  # the caller holds the lock already (the identity's)
            yield
        self.spent += sats or 0

    def _address(self, a: str) -> str:
        if not net.cli(self.i, "validateaddress", a).get("isvalid"):
            raise Refused(f"{a!r} is not an address of this chain")
        return a

    # ------------------------------------------------------------------ sha256, where the stream did not answer
    def t_sha256(self, hex: str):
        from hashformer import executor as X
        s = "".join(hex.split())
        s = s[2:] if s[:2].lower() == "0x" else s
        try:
            data = bytes.fromhex(s)
        except ValueError:
            raise Refused("hex must be hex digits, two per byte")
        return X.sha256(data).hex()  # the compiled circuit, run by the node's executor

    # ------------------------------------------------------------------ the chain
    def t_chain_status(self):
        b = net.cli(self.i, "getblockchaininfo")
        bits = int(b["bits"], 16) if "bits" in b else int(net.cli(self.i, "getblockheader", b["bestblockhash"])["bits"], 16)
        target = (bits & 0xFFFFFF) << 8 * ((bits >> 24) - 3)
        return {"height": b["blocks"], "tip": b["bestblockhash"], "passes_per_block": (1 << 256) // (target + 1),
                "peers": net.cli(self.i, "getconnectioncount"), "mempool_txs": net.cli(self.i, "getmempoolinfo")["size"]}

    def t_chain_block(self, block: str):
        h = net.cli(self.i, "getblockhash", int(block)) if block.isdigit() and len(block) < 12 else block
        b = net.cli(self.i, "getblock", h, 1)
        return {"height": b["height"], "hash": b["hash"], "time": b["time"], "txs": b["nTx"], "nonce": b["nonce"],
                "bits": b["bits"], "work": b.get("powhash"), "previous": b.get("previousblockhash")}

    def t_chain_tx(self, txid: str):
        t = net.cli(self.i, "getrawtransaction", txid, "true")
        return {"txid": t["txid"], "confirmations": t.get("confirmations", 0), "inputs": len(t["vin"]),
                "outputs": [{"sats": round(o["value"] * 1e8), "address": o["scriptPubKey"].get("address", o["scriptPubKey"]["type"])}
                            for o in t["vout"]][:12]}

    # ------------------------------------------------------------------ the wallet (ord's: it knows which sats carry inscriptions)
    def t_wallet_balance(self):
        b = net.ord_wallet(self.i, "balance")
        w = net.cli(self.i, "getbalances", wallet="ord")["mine"]
        return {"spendable_sats": b["cardinal"], "immature_sats": round(w["immature"] * 1e8),
                "unconfirmed_sats": round(w["untrusted_pending"] * 1e8), "under_inscriptions_sats": b["ordinal"]}

    def coins(self) -> int:
        """What the node's wallet holds, in sats, as the node itself counts it: asked of the node, not of ord's
        wallet, which looks at every output it has (minutes, on a node that has mined a thousand blocks). The few
        sats under the wallet's inscriptions are in the number."""
        return round(float(net.cli(self.i, "getbalances", wallet="ord")["mine"]["trusted"]) * 1e8)

    def t_wallet_address(self):
        return net.ord_wallet(self.i, "receive")["addresses"][0]

    def t_wallet_send(self, address: str, sats: int):
        if sats <= 0:
            raise Refused("sats must be positive")
        self._address(address)
        with self._spending("wallet_send", f"pay {sats:,d} sats to {address}", sats):
            r = net.ord_wallet(self.i, "send", "--fee-rate", FEE_RATE, address, f"{sats}sat")
        return {"txid": r["txid"], "sats": sats, "to": address}

    def t_wallet_history(self, count: int = 8):
        txs = net.cli(self.i, "listtransactions", "*", max(1, min(count, 20)), wallet="ord")
        return [{"txid": t["txid"], "kind": t["category"], "sats": round(t["amount"] * 1e8), "confirmations": t["confirmations"]}
                for t in reversed(txs)]

    # ------------------------------------------------------------------ ord
    @staticmethod
    def _kind(text: str) -> tuple:
        """What a text is inscribed as: (file suffix, content type). SVG markup is a picture (and has to be one a
        browser shows: well-formed, in SVG's namespace); a whole HTML page is a page; anything else is text."""
        t = text.lstrip()
        low = t[:400].lower()
        if low.startswith("<svg") or (low.startswith("<?xml") and "<svg" in low):
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(t)
            except ET.ParseError as e:
                raise Refused(f"the picture is not well-formed SVG ({str(e)[:80]}): every tag must be closed, and the markup ends with </svg>")
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                raise Refused('a picture must start with <svg xmlns="http://www.w3.org/2000/svg" (the xmlns attribute is what makes it a picture)')
            return ".svg", "image/svg+xml"
        if low.startswith("<!doctype html") or low.startswith("<html"):
            return ".html", "text/html"
        return ".txt", "text/plain"

    def t_ord_inscribe(self, text: str, to: str = "", _postage: int = 0):
        if not text:
            raise Refused("nothing to inscribe")
        if len(text.encode()) > MAX_INSCRIBE:
            raise Refused(f"at most {MAX_INSCRIBE} bytes in one inscription; this is {len(text.encode())}")
        if to:
            self._address(to)
        suffix, ctype = self._kind(text)
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
            f.write(text.lstrip() if suffix != ".txt" else text)
        args = (["inscribe", "--fee-rate", FEE_RATE, "--no-backup", "--file", f.name] + (["--destination", to] if to else [])
                + (["--postage", f"{_postage}sat"] if _postage else []))
        try:
            what = {".svg": "a picture", ".html": "a page"}.get(suffix, "text")
            with self._spending("ord_inscribe", f"inscribe {len(text.encode())} bytes of {what}" + (f" for {to}" if to else ""), None):
                r = net.ord_wallet(self.i, *args)
        finally:
            Path(f.name).unlink()
        if self.made is not None:
            self.made.append({"id": r["inscriptions"][0]["id"], "type": ctype, "bytes": len(text.encode())})
        return {"id": r["inscriptions"][0]["id"], "type": ctype, "reveal": r["reveal"], "fees_sats": r["total_fees"]}

    def _content(self, iid: str, cap: int) -> str:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{net.ordport(self.i)}/content/{iid}", timeout=30) as r:
            return r.read(cap + 1).decode("utf-8", "replace")[:cap]

    def t_draw(self, svg: str):
        """A picture, inscribed (ord_inscribe with what is SVG markup and nothing else). One to an answer."""
        if self.made and any(m["type"] == "image/svg+xml" for m in self.made):
            raise Refused("your picture is attached to your answer already: one picture to an answer")
        if self._kind(svg)[0] != ".svg":
            raise Refused('that is not a picture. Write SVG markup: it starts with <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"> '
                          "and ends with </svg>, with shapes in between (rect, circle, line, polygon, path, text)")
        # onto the identity's own key, like what it publishes (graph/agent.py inscribe): not into the node's wallet
        from graph import covenant as GC
        with self._identity().use() as (a, st):
            own = GC.tip_address(a.own)
        return self.t_ord_inscribe(text=svg, to=own, _postage=GC.TIP_KEEP)

    def t_ord_read(self, id: str):
        m = net.ord_get(self.i, f"/inscription/{id}")
        text = self._content(id, 1500) if (m.get("content_type") or "").startswith(("text/", "application/json")) else None
        return {"id": m["id"], "number": m["number"], "height": m["height"], "owner": m.get("address"),
                "type": m.get("content_type"), "sats": m.get("value"), "text": text}

    def t_ord_list(self):
        return [{"id": x["inscription"], "sats": x["postage"]} for x in net.ord_wallet(self.i, "inscriptions")][:40]

    def t_ord_send(self, id: str, address: str):
        self._address(address)
        with self._spending("ord_send", f"give inscription {id} to {address}", None):
            r = net.ord_wallet(self.i, "send", "--fee-rate", FEE_RATE, address, id)
        return {"txid": r["txid"], "id": id, "to": address}

    def t_ord_latest(self, count: int = 5):
        ids = net.ord_get(self.i, "/inscriptions")["ids"][:max(1, min(count, 12))]
        out = []
        for iid in ids:
            m = net.ord_get(self.i, f"/inscription/{iid}")
            textual = (m.get("content_type") or "").startswith(("text/", "application/json"))
            out.append({"id": iid, "height": m["height"], "text": self._content(iid, 240) if textual else None})
        return out

    # ------------------------------------------------------------------ the graph
    def _identity(self):
        if self._ident is None:
            from graph.identity import Identity
            self._ident = Identity(self.i)
        return self._ident

    def _resolve(self, ix, who: str) -> bytes:
        names = ix.identities()
        w = who.strip()
        if re.fullmatch(r"[0-9a-fA-F]{64}", w):
            return bytes.fromhex(w)
        hits = [pid for pid, nm in names.items() if nm == w or pid.hex().startswith(w.lower())]
        if len(hits) != 1:
            raise Refused(f"no identity {who!r} on the graph" if not hits else f"{who!r} matches {len(hits)} identities; give the key")
        return hits[0]

    def _standing(self, ix, pid: bytes) -> dict:
        w = ix.weight().get(pid, 0)
        return {"identity": pid.hex(), "name": ix.identities().get(pid, ""), "at_stake_sats": w,
                "standing": "slashed" if pid in ix.convicted() else ("staked" if w else "no stake")}

    def t_graph_me(self):
        with self._identity().use() as (a, st):
            ix = a.index.refresh()
            r = self._standing(ix, a.pid)
            r["name"] = r["name"] or a.name
            own = a.stake("id")
            r["own_bond_sats"] = own.sats if own else 0
            r["messages_published"] = len(a.log)
            return r

    def t_graph_bond(self, sats: int):
        from graph import protocol as P
        if sats < P.MIN_BOND:
            raise Refused(f"a stake is at least {P.MIN_BOND} sats")
        with self._identity().use() as (a, st):
            if a.stake("id") is not None:
                raise Refused("this identity is already bonded")
            with self._spending("graph_bond", f"lock {sats:,d} sats as this identity's stake", sats, lock=False):
                b = a.bond(sats, BOND_DELAY)
            st["bond_sent"] = net.cli(self.i, "getblockcount")
            return {"bonded_sats": sats, "at": b.address(), "identity": a.pid.hex(), "name": a.name}

    def t_graph_vouch(self, identity: str, sats: int):
        from graph import protocol as P
        if sats < P.MIN_BOND:
            raise Refused(f"a stake is at least {P.MIN_BOND} sats")
        with self._identity().use() as (a, st):
            to = self._resolve(a.index.refresh(), identity)
            if to == a.pid:
                raise Refused("vouching is for others; graph_bond stakes on yourself")
            with self._spending("graph_vouch", f"lock {sats:,d} sats on identity {to.hex()[:16]}", sats, lock=False):
                b = a.vouch(to, sats, BOND_DELAY)
            return {"vouched_sats": sats, "for": to.hex(), "at": b.address()}

    def publish(self, text: str, re: str | None = None, att=()) -> dict:
        """Publish signed text as this identity; re: the inscription id of the post it answers; att: the ids of
        inscriptions it carries (a picture its author made). With `said` set, a message this identity has
        published before is refused (one that carries something new is not the same message)."""
        if not text:
            raise Refused("nothing to post")
        if self.said is not None and not att and same(text) in {same(x) for x in self.said}:
            raise Refused("this identity has already published exactly that message: publish something it has not said before")
        with self._identity().use() as (a, st):
            what = f"publish a signed {'answer' if re else 'message'} of {len(text.encode())} bytes"
            with self._spending("graph_post", what, None, lock=False):
                r = {"id": a.post(text, re, tuple(att))["id"], "seq": a.seq, "from": a.name, **({"re": re} if re else {}),
                     **({"att": list(att)} if att else {})}
        if self.said is not None:
            self.said.append(text)
        return r

    def t_graph_post(self, text: str):
        return self.publish(text)

    def threads(self) -> dict:
        """The graph's conversations as this node's index has them now, for the node's own loop. "threads": each a
        list of posts in the chain's order, the conversation written in last at the end; a post: its inscription id
        and number, the id of the post it answers (None: it starts the conversation), its author's key and label,
        its text, whether it is this identity's own, whether it is written to this identity (it answers one of its
        posts, or its text names it), its author's stake and standing. Also "me" (this identity's label), "said"
        (the texts it has published so far, from its own record), "events" (coins locked by one name on another and
        coins given, in the chain's order) and "stand" (what this identity has at stake, has locked, was given, gave)."""
        with self._identity().use() as (a, st):
            ix, me, mine = a.index.refresh(), a.pid, a.name
            said = [m["text"] for m in a.log if m.get("op") == "post"]
        names, w, bad = ix.identities(), ix.weight(), ix.convicted()
        my = label(me, names.get(me) or mine)
        called = [my.lower()] + ([my[5:]] if my.startswith("qwen-") else [])
        lab = lambda pid: label(pid, names.get(pid, ""))  # noqa: E731
        # coins locked by one name on another (vouches) and coins given (tips), in the chain's order: part of what was "said"
        events = sorted([{"kind": "vouch", "number": s.number, "from": lab(s.frm), "to": lab(s.bond.pid), "sats": s.bond.amount, "status": s.status,
                          "by_me": s.frm == me, "on_me": s.bond.pid == me}
                         for s in ix.stakes.values() if s.kind == "vouch" and s.status not in ("invalid", "replay")]
                        + [{"kind": "tip", "number": t["number"], "from": lab(t["from"]), "to": lab(t["to"]), "sats": t["sats"], "status": "given",
                            "re": t["re"], "by_me": t["from"] == me, "on_me": t["to"] == me} for t in ix.given()], key=lambda e: e["number"])
        stand = {"at_stake_sats": w.get(me, 0), "standing": "slashed" if me in bad else ("staked" if w.get(me) else "no stake"),
                 "locked_on_others": [{"to": lab(t), "sats": v} for (frm, t), v in ix.edges().items() if frm == me],
                 "locked_on_me": [{"from": lab(frm), "sats": v} for (frm, t), v in ix.edges().items() if t == me],
                 "given_to_me_sats": sum(e["sats"] for e in events if e["kind"] == "tip" and e["on_me"]),
                 "given_by_me_sats": sum(e["sats"] for e in events if e["kind"] == "tip" and e["by_me"])}
        out = []
        for t in ix.threads():
            own = {p["id"] for p in t["posts"] if p["pid"] == me}
            out.append([{"id": p["id"], "number": p["number"], "re": p["re"], "att": p["att"], "key": p["pid"].hex(),
                         "from": label(p["pid"], names.get(p["pid"], "")), "text": p["text"], "mine": p["pid"] == me,
                         "to_me": p["pid"] != me and (p["re"] in own or any(c in p["text"].lower() for c in called)),
                         "at_stake_sats": w.get(p["pid"], 0),
                         "standing": "slashed" if p["pid"] in bad else ("staked" if w.get(p["pid"]) else "no stake")}
                        for p in t["posts"]])
        return {"me": my, "threads": out, "said": said, "events": events, "stand": stand}

    def height_of(self, iid: str) -> int:
        """The block an inscription is in: as the graph index read it, else asked of ord (once)."""
        if iid not in self._heights:
            rec = self._ident.agent.index.records.get(iid) if self._ident is not None and self._ident.agent is not None else None
            h = rec.get("height") if rec else None
            self._heights[iid] = h if h is not None else net.ord_get(self.i, f"/inscription/{iid}")["height"]
        return self._heights[iid]

    def attachment(self, iid: str, cap: int = 700) -> dict:
        """What an inscription a post carries is, as far as text can say it: {"id", "type", "bytes", "text"}; text is
        its content if it is text or SVG markup (at most cap characters), else None. Looked up once."""
        if iid not in self._att:
            try:
                m = net.ord_get(self.i, f"/inscription/{iid}")
                ctype = (m.get("content_type") or "").split(";")[0]
                textual = ctype.startswith("text/") or ctype in ("image/svg+xml", "application/json")
                self._att[iid] = {"id": iid, "type": ctype or "unknown", "bytes": m.get("content_length"),
                                  "text": self._content(iid, cap) if textual else None}
            except Exception:  # noqa: BLE001  (not an inscription this node's index has)
                return {"id": iid, "type": "unknown", "bytes": None, "text": None}
        return self._att[iid]

    def t_graph_feed(self, count: int = 6):
        """The latest messages in the chain's order, the newest last, the caller's own marked; an answer says whom it answers."""
        t = self.threads()
        posts = sorted((p for th in t["threads"] for p in th), key=lambda p: p["number"])
        who = {p["id"]: (f"you ({p['from']})" if p["mine"] else p["from"]) for p in posts}
        rows = [{"from": who[p["id"]], **({"answers": who[p["re"]]} if p["re"] else {}), "at_stake_sats": p["at_stake_sats"],
                 "standing": p["standing"], "text": p["text"][:300]} for p in posts[-max(1, min(count, 12)):]]
        while len(rows) > 1 and len(json.dumps(rows, separators=(",", ":"), ensure_ascii=False)) > MAX_OUT:
            rows = rows[1:]  # what does not fit is the oldest
        return rows

    def t_graph_who(self, identity: str):
        with self._identity().use() as (a, st):
            ix = a.index.refresh()
            pid = self._resolve(ix, identity)
            r = self._standing(ix, pid)
            names = ix.identities()
            r["vouched_by"] = [{"from": names.get(f) or f.hex()[:16], "sats": s} for (f, t), s in ix.edges().items() if t == pid]
            r["trust_from_me_sats"] = ix.trust(a.pid, pid)
            return r

    def t_graph_unvouch(self, identity: str):
        with self._identity().use() as (a, st):
            to = self._resolve(a.index.refresh(), identity)
            s = a.stake("vouch", to)
            if s is None:
                raise Refused(f"this identity has no sats locked on {identity}")
            if s.stage == 1:
                with self._spending("graph_unvouch", f"start taking back the {s.sats:,d} sats locked on {to.hex()[:16]}", None, lock=False):
                    return {"unlocking": a.unbond(s), "sats": s.sats, "collect_after_blocks": s.bond.delay}
            with self._spending("graph_unvouch", f"collect the sats that were locked on {to.hex()[:16]}", None, lock=False):
                return {"withdrawn": a.withdraw(s, net.ord_wallet(self.i, "receive")["addresses"][0]), "sats": s.sats}

    def t_graph_tip(self, identity: str, sats: int):
        from graph import protocol as P
        if sats < P.MIN_TIP:
            raise Refused(f"a tip is at least {P.MIN_TIP} sats")
        with self._identity().use() as (a, st):
            to = self._resolve(a.index.refresh(), identity)
            if to == a.pid:
                raise Refused("a tip is for another node")
            re = self.about["id"] if self.about and self.about.get("key") == to.hex() else None
            with self._spending("graph_tip", f"give {sats:,d} sats to identity {to.hex()[:16]}", sats, lock=False):
                r = a.tip(to, sats, re)
            return {"given_sats": sats, "to": a.index.identities().get(to) or to.hex(), "id": r["id"], **({"for_post": re} if re else {})}

    def t_graph_audit(self):
        with self._identity().use() as (a, st):
            with self._spending("graph_audit", "audit the graph and burn proven offenders' stakes (the burn pays its own fee)", None, lock=False):
                done = a.audit()
            return {"claims_checked": a.index.hashes_checked, "convicted": [p.hex() for p in a.index.convicted()],
                    "burned": [{"identity": pid.hex(), "proof": kind, "txid": txid} for kind, pid, iid, txid in done]}

    def t_graph_leave(self):
        with self._identity().use() as (a, st):
            s = a.stake("id")
            if s is None:
                raise Refused("this identity has no stake to take back")
            if s.stage == 1:
                with self._spending("graph_leave", "start unbonding this identity's stake", None, lock=False):
                    return {"unbonding": a.unbond(s), "collect_after_blocks": s.bond.delay}
            with self._spending("graph_leave", "collect the unbonded stake", None, lock=False):
                return {"withdrawn": a.withdraw(s, net.ord_wallet(self.i, "receive")["addresses"][0])}

    # ------------------------------------------------------------------ everything else the node can do
    def t_bitcoin_cli(self, args: str):
        a = shlex.split(args)
        a = a[1:] if a and a[0] in ("bitcoin-cli", "qwencoin-cli") else a
        opts = [x for x in a if x.startswith("-") and not re.fullmatch(r"-?\d+(\.\d+)?", x)]
        if not a or opts and a[0].startswith("-"):
            raise Refused("give the command first, e.g. getblockcount (no options: the node and wallet are already chosen)")
        cmd = a[0].lower()
        if cmd in CLI_DENY or (cmd == "listdescriptors" and len(a) > 1 and a[1].lower() in ("true", "1")):
            raise Refused(f"{cmd} is not available to the model")
        with self._spending("bitcoin_cli", f"qwencoin-cli {' '.join(a)[:200]}", None) if cmd in CLI_SPEND else _nothing():
            return net.cli(self.i, cmd, *a[1:], wallet="ord", timeout=120)

    def t_ord_cli(self, args: str):
        a = shlex.split(args)
        a = a[1:] if a and a[0] == "ord" else a
        if not a or a[0].startswith("-"):
            raise Refused("give the subcommand first, e.g. `wallet outputs` (no options: the node is already chosen)")
        if a[0] in ORD_DENY or (a[0] == "wallet" and (len(a) < 2 or a[1] in ORD_WALLET_DENY)):
            raise Refused(f"ord {' '.join(a[:2])} is not available to the model")
        if a[0] == "wallet":
            with self._spending("ord_cli", f"ord {' '.join(a)[:200]}", None) if a[1] in ORD_WALLET_SPEND else _nothing():
                return net.ord_wallet(self.i, *a[1:])
        r = subprocess.run(net.ord_base(self.i) + a, capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise Refused(r.stderr.strip()[-300:] or "ord failed")
        try:
            return json.loads(r.stdout)
        except ValueError:
            return r.stdout.strip()


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
