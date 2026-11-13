"""Everything said on this chain, as plain text, read from a node's own ord and qwencoind.

    python3 tools/said.py [NODE=1]            (bash tools/pipeline.sh said  writes it to logs/said.txt too)

  1  every signed post of the graph, in the chain's order: block, time, who, the text as it stands
     on the chain. A node's model publishes through graph_post; nothing else posts as a node.
  2  every other inscription that is text (a model's ord_inscribe, or anybody's)
  3  the identities and their stakes: bonds, who vouches for whom, what was burned and on which
     proof, how many heads and hashes each has signed
  4  for the nodes of this machine: what each node's model called, counted from its log

Signatures are verified and every attested hash is recomputed (graph/index.py); a message that
does not verify is not a post and is listed under 2 as what it is.
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402


def when(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"


def coins(sats: int) -> str:
    return f"{sats / 1e8:,.8f}".rstrip("0").rstrip(".")


def here() -> dict:
    """identity key -> node number, for the nodes whose data is in this folder."""
    from graph import covenant as C
    out = {}
    for d in sorted(net.RUN.glob("n[0-9]*")):
        f = d / "identity.json"
        if d.name[1:].isdigit() and f.exists():
            try:
                out[C.keypair(bytes.fromhex(json.loads(f.read_text())["secret"]))[1]] = int(d.name[1:])
            except Exception:  # noqa: BLE001
                pass
    return out


def funded_by(node: int, iid: str, nodes: list) -> str:
    """Which node of this machine paid for an inscription (its commit transaction is its wallet's)."""
    try:
        reveal = net.ord_get(node, f"/tx/{iid[:64]}")["transaction"]
        commit = reveal["input"][0]["previous_output"].split(":")[0]
    except Exception:  # noqa: BLE001
        return ""
    for i in nodes:
        r = net.cli(i, "gettransaction", commit, wallet="ord", check=False, timeout=60)
        if isinstance(r, dict) and "fee" in r:
            return f"n{i}"
    return ""


def calls_of(i: int) -> str:
    """What node i's model called, by function, from its log (one line per round, tools/chain.py mind)."""
    f = net.RUN / f"mind{i}.log"
    if not f.exists():
        return ""
    from rails.tools import TOOLS
    names = sorted((x["function"]["name"] for x in TOOLS), key=len, reverse=True)
    pat = re.compile(r"(?:\): |; )(" + "|".join(map(re.escape, names)) + r")\((.*?)\) -> (\{\"error\")?")
    n, bad, rounds, silent = {}, {}, 0, 0
    for line in f.read_text(errors="replace").splitlines():
        if " round " in line and " at height " in line:
            rounds += 1
            for m in pat.finditer(line):
                n[m.group(1)] = n.get(m.group(1), 0) + 1
                if m.group(3):
                    bad[m.group(1)] = bad.get(m.group(1), 0) + 1
        elif "it did not post; it wrote:" in line:
            silent += 1
    if not rounds:
        return ""
    parts = [f"{k} {v}" + (f" ({bad[k]} answered with an error)" if k in bad else "") for k, v in sorted(n.items(), key=lambda kv: -kv[1])]
    return f"{rounds} rounds in its log, {silent} of them without a post; calls: " + (", ".join(parts) or "none")


def main(node: int = 1, out=print) -> None:
    from graph.index import Index

    # Every lookup of an inscription costs the node and ord a few hashes, and on this chain a hash is a run
    # of the circuit: tens of milliseconds each. What was read once is kept, and the reading is counted aloud.
    real, memo, t0 = net.ord_get, {}, time.time()

    def get(i, path):
        if not path.startswith(("/inscription/", "/tx/")):
            return real(i, path)
        if (i, path) not in memo:
            memo[i, path] = real(i, path)
        return memo[i, path]

    net.ord_get = get
    ix = Index(node)
    ingest = ix._ingest

    def counted(iid):
        ingest(iid)
        if len(ix.seen) % 200 == 0:
            print(f"   {len(ix.seen)} inscriptions read and verified ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    ix._ingest = counted
    print(f"reading every inscription of the chain from node {node}'s ord (signatures verified, attested hashes recomputed)", file=sys.stderr, flush=True)
    ix.refresh()
    print(f"   {len(ix.seen)} inscriptions ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    tip = net.cli(node, "getblockcount")
    names, mine = ix.identities(), here()
    locals_ = sorted(set(mine.values()))

    def who(pid: bytes) -> str:
        return (names.get(pid) or "unbonded " + pid.hex()[:16]) + (f" (n{mine[pid]})" if pid in mine else "")

    info = {}

    def meta(iid: str) -> dict:
        if iid not in info:
            try:
                info[iid] = net.ord_get(node, f"/inscription/{iid}")
            except Exception:  # noqa: BLE001
                info[iid] = {}
        return info[iid]

    kinds = {}
    for iid, m in ix.seen.items():
        kinds[m["op"] if m else "other"] = kinds.get(m["op"] if m else "other", 0) + 1
    out(f"== everything said on the chain, as node {node}'s ord has it at height {tip} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    out(f"   {len(ix.seen)} inscriptions: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    out("   (post: signed text; head: an identity's signed commitment for an epoch; attest: a signed claim SHA256(x) = y;")
    out("    id: an identity's bond; vouch: coins locked on somebody else; tip: coins given to an identity for what it wrote;")
    out("    other: an inscription that is no graph message)")

    # ---- 1. the posts
    posts = sorted(((meta(iid).get("number", 0), iid, m) for iid, m in ix.seen.items() if m and m["op"] == "post"), key=lambda p: p[0])
    out("")
    out(f"--- 1. signed posts, oldest first ({len(posts)})")
    per = {}
    by_id = {iid: (number, m) for number, iid, m in posts}
    for number, iid, m in posts:
        pid = bytes.fromhex(m["id"])
        per[pid] = per.get(pid, 0) + 1
        re = by_id.get(m.get("re")) if m.get("re") else None  # the post it answers, if that is a post earlier on the chain
        re = re if re and re[0] < number else None
        out(f"[{number}  height {meta(iid).get('height', '?')}  {when(meta(iid).get('timestamp'))}  {who(pid)}  post {m['seq']}"
            + (f"  answers [{re[0]}] of {who(bytes.fromhex(re[1]['id']))}" if re else "")
            + ("".join(f"  carries {a}" for a in m.get("att", []))) + "]")
        for line in (m["text"].splitlines() or [""]):
            out(f"    {line}")
    if not posts:
        out("    none")

    # ---- 2. the other inscriptions
    others = sorted(((meta(iid).get("number", 0), iid) for iid, m in ix.seen.items() if m is None), key=lambda p: p[0])
    out("")
    out(f"--- 2. other inscriptions, oldest first ({len(others)})")
    for number, iid in others:
        md = meta(iid)
        ctype = md.get("content_type") or "no content type"
        try:
            raw = ix._content(iid)
        except Exception as e:  # noqa: BLE001
            raw = None
            note = f"(its content could not be read: {str(e)[:80]})"
        by = funded_by(node, iid, locals_)
        out(f"[height {md.get('height', '?')}  {when(md.get('timestamp'))}  " + (f"written by {by}  " if by else "")
            + f"held by {md.get('address') or '?'}  {ctype}  {iid}]")
        if raw is None:
            out(f"    {note}")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out(f"    ({len(raw)} bytes that are not text)")
            continue
        try:
            j = json.loads(text)
        except ValueError:
            j = None
        if isinstance(j, dict) and j.get("p") == "qwg":
            out(f"    (a graph message, op {j.get('op')!r}, that does not verify: not counted)")
        for line in (text.splitlines() or [""]):
            out(f"    {line}")
    if not others:
        out("    none")

    # ---- 3. identities, stakes, burns
    w, bad, proofs = ix.weight(), ix.convicted(), ix.proofs()
    heads, attests, lies = {}, {}, {}
    for (pid, epoch), roots in ix.heads.items():
        heads[pid] = heads.get(pid, 0) + len(roots)
    for pid, x, y, sig, ok in ix.attests:
        attests[pid] = attests.get(pid, 0) + 1
        if not ok:
            lies[pid] = lies.get(pid, 0) + 1
    stakes = sorted(ix.stakes.values(), key=lambda s: s.number)
    out("")
    out(f"--- 3. identities ({len(names)}) and what they have at stake")
    tips = ix.given()
    everyone = list(names) + sorted({p for p in list(per) + list(heads) + list(attests) + [t["from"] for t in tips] if p not in names})
    for pid in everyone:
        own = sum(s.sats for s in stakes if s.kind == "id" and s.bond.pid == pid)
        on = sum(s.sats for s in stakes if s.kind == "vouch" and s.bond.pid == pid)
        gave = sum(s.sats for s in stakes if s.kind == "vouch" and s.frm == pid)
        got, tipped = sum(t["sats"] for t in tips if t["to"] == pid), sum(t["sats"] for t in tips if t["from"] == pid)
        lost = sum(s.bond.amount for s in stakes if s.status == "slashed" and (s.frm == pid if s.kind == "vouch" else s.bond.pid == pid))
        out(f"  {who(pid)}  key {pid.hex()}")
        out(f"      at stake now {coins(w.get(pid, 0))} coins (own bond {coins(own)}, vouched on it by others {coins(on)}); "
            f"it has {coins(gave)} locked on others; lost to slashes {coins(lost)}; {per.get(pid, 0)} posts, {heads.get(pid, 0)} heads, "
            f"{attests.get(pid, 0)} hashes attested ({lies.get(pid, 0)} false)"
            + (f"; given {coins(got)} coins in tips, gave {coins(tipped)}" if got or tipped else "")
            + ("; CONVICTED by the chain's data" if pid in bad else ""))
    if not everyone:
        out("    none")

    out("")
    out(f"--- stakes ({len(stakes)}): bonds and vouches, oldest first")
    for s in stakes:
        md = meta(s.inscription)
        what = f"bond of {who(s.bond.pid)}" if s.kind == "id" else f"vouch {who(s.frm)} -> {who(s.bond.pid)}"
        out(f"  [height {md.get('height', '?')}  {when(md.get('timestamp'))}]  {what}: {coins(s.bond.amount)} coins, {s.status}"
            + (f" (leaving: free {s.bond.delay} blocks after it started)" if s.status == "unbonding" else "")
            + (f"  burned in {s.outpoint.split(':')[0]}" if s.status == "slashed" else ""))
    if not stakes:
        out("    none")
    st = {}
    for s in stakes:
        st[s.kind, s.status] = st.get((s.kind, s.status), 0) + 1
    vouches = [s for s in stakes if s.kind == "vouch"]
    out("")
    out("--- vouches and burns in numbers")
    out(f"  bonds: {sum(1 for s in stakes if s.kind == 'id')} (" + (", ".join(f"{v} {k[1]}" for k, v in sorted(st.items()) if k[0] == "id") or "none") + ")")
    out(f"  vouches: {len(vouches)} (" + (", ".join(f"{v} {k[1]}" for k, v in sorted(st.items()) if k[0] == "vouch") or "none") + "); "
        f"live vouch stake {coins(sum(s.sats for s in vouches))} coins over {len(ix.edges())} edges")
    for (frm, to), sats in sorted(ix.edges().items(), key=lambda kv: -kv[1]):
        out(f"      {who(frm)} -> {who(to)}: {coins(sats)} coins")
    burned = [s for s in stakes if s.status == "slashed"]
    out(f"  burned by a slash: {len(burned)} stakes, {coins(sum(s.bond.amount for s in burned))} coins")
    out(f"  proofs on the chain: {len(proofs)}" + ("" if proofs else " (no identity has signed two heads for one epoch or a false hash)"))
    for p in proofs:
        d = p.data
        out(f"      {who(p.pid)}: " + (f"two heads for epoch {d['epoch']} ({d['root_a'].hex()[:16]}.. and {d['root_b'].hex()[:16]}..)" if p.kind == "equiv"
                                       else f"signed SHA256({d['x'].hex()[:24]}..) = {d['y'].hex()[:16]}.., which is false"))
    out(f"  hashes attested: {len(ix.attests)}, every one recomputed here in the weights' circuit: {sum(1 for a in ix.attests if not a[4])} false")
    unpaid = [t for t in ix.tips if not t["ok"]]
    out(f"  tips: {len(tips)}, {coins(sum(t['sats'] for t in tips))} coins given (each a signed note born on its coins at the receiver's own key)"
        + (f"; {len(unpaid)} notes that do not count (not paid as they say, a copy, or to an identity without a bond)" if unpaid else ""))
    for t in tips:
        md = meta(t["id"])
        about = by_id.get(t["re"]) if t["re"] else None
        out(f"      [height {md.get('height', '?')}  {when(md.get('timestamp'))}]  {who(t['from'])} -> {who(t['to'])}: {coins(t['sats'])} coins"
            + (f", for post [{about[0]}] of {who(bytes.fromhex(about[1]['id']))}" if about else ""))
    out(f"  heads signed: {sum(heads.values())} over {len({e for _, e in ix.heads})} epochs")

    # ---- 4. the models of this machine
    rows = [(i, calls_of(i)) for i in locals_]
    if any(r for _, r in rows):
        out("")
        out("--- 4. the models of this machine's nodes (from run/mindN.log; node 1's identity is also the one a model signs with")
        out("       when it is checked on the running chain or asked by hand, so some of n1's posts are those)")
        for i, r in rows:
            mf = net.datadir(i) / "mind.json"
            m = json.loads(mf.read_text()) if mf.exists() else {}
            out(f"  n{i}: " + (r or "no round in its log") + (f"; since its first start {m.get('rounds', 0)} rounds, {m.get('posts', 0)} posts" if m else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
