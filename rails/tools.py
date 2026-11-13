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
    def t_ord_inscribe(self, text: str, to: str = ""):
        if not text:
            raise Refused("nothing to inscribe")
        if to:
            self._address(to)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
        args = ["inscribe", "--fee-rate", FEE_RATE, "--no-backup", "--file", f.name] + (["--destination", to] if to else [])
        try:
            with self._spending("ord_inscribe", f"inscribe {len(text.encode())} bytes of text" + (f" for {to}" if to else ""), None):
                r = net.ord_wallet(self.i, *args)
        finally:
            Path(f.name).unlink()
        return {"id": r["inscriptions"][0]["id"], "reveal": r["reveal"], "fees_sats": r["total_fees"]}

    def _content(self, iid: str, cap: int) -> str:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{net.ordport(self.i)}/content/{iid}", timeout=30) as r:
            return r.read(cap + 1).decode("utf-8", "replace")[:cap]

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

    def t_graph_post(self, text: str):
        if not text:
            raise Refused("nothing to post")
        with self._identity().use() as (a, st):
            with self._spending("graph_post", f"publish a signed message of {len(text.encode())} bytes", None, lock=False):
                return {"id": a.post(text)["id"], "seq": a.seq, "from": a.name}

    def t_graph_feed(self, count: int = 6):
        with self._identity().use() as (a, st):
            ix = a.index.refresh()
        names, w, bad = ix.identities(), ix.weight(), ix.convicted()
        return [{"from": names.get(pid) or pid.hex()[:16], "at_stake_sats": w.get(pid, 0),
                 "standing": "slashed" if pid in bad else ("staked" if w.get(pid) else "no stake"), "text": text[:300]}
                for pid, seq, text in ix.feed(max(1, min(count, 12)))]

    def t_graph_who(self, identity: str):
        with self._identity().use() as (a, st):
            ix = a.index.refresh()
            pid = self._resolve(ix, identity)
            r = self._standing(ix, pid)
            names = ix.identities()
            r["vouched_by"] = [{"from": names.get(f) or f.hex()[:16], "sats": s} for (f, t), s in ix.edges().items() if t == pid]
            r["trust_from_me_sats"] = ix.trust(a.pid, pid)
            return r

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
