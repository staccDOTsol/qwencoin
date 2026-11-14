"""What the graph index keeps between runs (graph/index.py: qwg-index.json) gives the index that reading the whole
chain again gives. On the private two-node network of the tool tests (tests/tools_net.py), whose chain has bonds,
vouches, burns, posts with answers and pictures, tips paid and unpaid:

    python3 tests/check_keep.py

  1. an index that keeps nothing and takes nothing up (QWG_INDEX_KEEP=0) reads every inscription: the reference
  2. an index that keeps: its first refresh reads everything and writes the file
  3. a new index takes the file up and reads only what is not buried yet: it must be the reference in everything
     that is derived (stakes and where they stand, heads, attestations, posts and conversations, tips, weights,
     edges, proofs), and it must have asked the node for far less
  4. a file kept under a block the chain does not have, or read by other code, is not taken up
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tools")]
import tools_net as T  # noqa: E402
import net  # noqa: E402
from graph import index as I  # noqa: E402


def state(ix) -> dict:
    return {
        "seen": sorted(ix.seen), "messages": sum(1 for m in ix.seen.values() if m),
        "stakes": sorted((s.inscription, s.kind, s.status, s.stage, s.outpoint, s.born, s.number, s.sats) for s in ix.stakes.values()),
        "heads": sorted((p.hex(), e, sorted(r.hex() for r in roots)) for (p, e), roots in ix.heads.items()),
        "attests": sorted((p.hex(), x.hex(), y.hex(), ok) for p, x, y, sig, ok in ix.attests),
        "posts": sorted((n, i, p.hex(), seq, text, re, att) for p, seq, text, n, i, re, att in ix.posts),
        "tips": sorted((t["number"], t["id"], t["ok"], t["paid"], t["sats"], t["re"]) for t in ix.tips),
        "given": [t["id"] for t in ix.given()],
        "weight": sorted((p.hex(), v) for p, v in ix.weight().items()),
        "edges": sorted((a.hex(), b.hex(), v) for (a, b), v in ix.edges().items()),
        "names": sorted((p.hex(), n) for p, n in ix.identities().items()),
        "threads": [(t["root"], [x["id"] for x in t["posts"]]) for t in ix.threads()],
        "proofs": sorted((p.kind, p.pid.hex()) for p in ix.proofs()),
        "checked": ix.hashes_checked,
    }


def counted(ix) -> list:
    n = [0]
    real = ix._read
    ix._read = lambda iid: (n.__setitem__(0, n[0] + 1), real(iid))[1]
    return n


def main():
    T.up(106, log=print)
    node, f = 1, net.datadir(1) / "qwg-index.json"
    for attempt in range(5):
        h = net.cli(node, "getblockcount")
        os.environ["QWG_INDEX_KEEP"] = "0"
        ref = I.Index(node)
        n_ref = counted(ref)
        t0 = time.time()
        ref.refresh()
        t_ref = time.time() - t0
        del os.environ["QWG_INDEX_KEEP"]
        f.unlink(missing_ok=True)
        first = I.Index(node)
        first.refresh()
        assert f.exists(), "nothing was kept"
        kept = json.loads(f.read_text())
        again = I.Index(node)
        n_again = counted(again)
        t0 = time.time()
        again.refresh()
        t_again = time.time() - t0
        if net.cli(node, "getblockcount") == h:
            break
        print(f"a block came while the indexes were read (height {h}): once more")
    else:
        sys.exit("the chain did not hold still")
    a, b, c = state(ref), state(first), state(again)
    for k in a:
        assert a[k] == b[k] == c[k], f"{k}: the index that took the file up differs\n  read: {a[k]}\n  kept: {c[k]}"
    buried = sum(1 for r in ref.records.values() if r["height"] <= h - I.KEEP_DEPTH)
    assert len(kept["records"]) == buried == again.taken_up and kept["anchor"][0] == h - I.KEEP_DEPTH, (len(kept["records"]), buried, again.taken_up)
    assert n_ref[0] == len(ref.seen) and n_again[0] == len(ref.seen) - buried, (n_ref, n_again, buried)
    print(f"height {h}: {len(a['seen'])} inscriptions ({a['messages']} graph messages: {len(a['stakes'])} stakes, {len(a['posts'])} posts, {len(a['tips'])} tip notes "
          f"of which {len(a['given'])} count, {len(a['attests'])} attestations, {len(a['proofs'])} proofs)")
    print(f"  read whole: {n_ref[0]} inscriptions read and verified in {t_ref:.1f}s")
    print(f"  kept: {buried} of them, buried {I.KEEP_DEPTH} blocks or more, under block {kept['anchor'][0]} ({kept['anchor'][1][:16]}..)")
    print(f"  taken up: the same index in everything derived, {n_again[0]} inscriptions read in {t_again:.1f}s")
    # a file from another chain, or read by other code, is left alone
    for what, change in (("kept under a block this chain does not have", lambda k: k.update(anchor=[k["anchor"][0], "00" * 32])),
                         ("read by other code", lambda k: k.update(reading="00000000"))):
        k = json.loads(f.read_text())
        change(k)
        f.write_text(json.dumps(k))
        ix = I.Index(node)
        n = counted(ix)
        ix._keep = lambda: None
        ix.refresh()
        assert ix.taken_up == 0 and n[0] == len(ix.seen) >= len(ref.seen), (what, ix.taken_up, n[0], len(ix.seen))
        print(f"  a file {what}: not taken up, everything read again ({n[0]} inscriptions)")
    f.unlink()
    I.Index(node).refresh()  # the file as it should be, for whatever runs next
    print("OK")


if __name__ == "__main__":
    main()
