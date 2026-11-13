"""ord takes in an orphan when a reorganisation falls inside one batch of its indexer, and the
node's tooling (tools/net.py: ord_sick, ord_rebuild, ord_heal) gets the index right again.

    QWEN_SEEDS=none QWEN_RUN=<folder> QWEN_PORT=9733 QWEN_ORD_PORT=19080 python3 tests/check_ord_orphan.py

Real node, real ord. Node 1 of QWEN_RUN has to hold two branches (getchaintips: a valid fork next
to the active chain, the active one longer): a node that mined by itself and then met the network
has them. The reorganisation is put inside the batch by stopping ord's process (SIGSTOP) while it
reads the shorter branch, switching the node to the longer one and letting ord go on: what a
block that arrives in the wrong millisecond does on a running chain.

ord checks that a block links to the one before it against the index as committed. Inside a
batch nothing is committed yet, so the blocks of two branches go in one after the other. The
index then has blocks the node no longer has, and lacks the outputs of the blocks that replaced
them: its indexer dies at the first transaction that spends one of those
(`assertion failed: !self.index.have_full_utxo_index()`), the server goes on answering, and
every wallet command waits for an index that never moves again.

The ord of this project checks the link inside a batch too (patches/ord-0.29.1-qwencoin.patch)
and drops a batch whose blocks do not link. On that ord the test shows the batch being dropped
and the index coming out right; on ord 0.29.1 without the check it shows the fault and the
repair by the tooling. logs/ord-orphan.log has both runs.
"""
import os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402


def say(*a):
    print(*a, flush=True)


def differing(lo, hi):
    """Heights lo..hi at which ord's index has another block than the node."""
    return [h for h in range(lo, hi + 1) if net.ord_get(1, f"/r/blockhash/{h}") != net.cli(1, "getblockhash", h)]


def into_batch(y1, long, after, tries=20):
    """ord indexes from nothing while the node goes from its short branch to its long one; ord is
    stopped once it has asked for `after` blocks. Returns (blocks of the short branch it had read,
    how it ended, the heights at which its index differs from the node)."""
    log = net.datadir(1) / "debug.log"
    os.environ["RUST_LOG"] = "warn"  # ord says why an update of its index was given up
    for t in range(tries):
        net.ord_stop(1)
        (net.RUN / "ord1" / "index.redb").unlink(missing_ok=True)
        net.cli(1, "invalidateblock", y1)  # the node on its short branch
        short = net.cli(1, "getblockcount")
        net.cli(1, "logging", '["rpc"]')  # the node writes down every request: that is how ord's reading is followed
        pos = log.stat().st_size
        net.ord_up(1)
        pid = int((net.RUN / "ord1" / "server.pid").read_text())
        read, t0 = 0, time.time()
        with open(log, "rb") as f:
            f.seek(pos)
            buf = b""
            while time.time() - t0 < 30 and read < after:
                buf += f.read()
                read = buf.count(b"method=getblock ")
            os.kill(pid, signal.SIGSTOP)
            time.sleep(0.5)  # the node finishes the answer it was at
            read = (buf + f.read()).count(b"method=getblock ")
        net.cli(1, "logging", "[]", '["rpc"]')
        net.cli(1, "reconsiderblock", y1)  # the node back on its long branch, ord still stopped in mid-batch
        for _ in range(600):
            if net.cli(1, "getblockcount") == long:
                break
            time.sleep(0.2)
        assert net.cli(1, "getblockcount") == long
        os.kill(pid, signal.SIGCONT)
        how, t0 = None, time.time()
        while how is None and time.time() - t0 < 600:
            time.sleep(1)
            try:
                at = net.ord_height(1)
            except Exception:  # noqa: BLE001
                continue
            dropped = [l.strip() for l in net.ord_since_start(1).splitlines() if "does not link" in l]
            if net.ord_panic(1):
                how = "its indexer died"
            elif at >= long:
                how = "it indexed to the node's tip"
            elif at == short and time.time() - t0 > 20:
                how = "it committed the short branch whole"
            elif dropped and time.time() - t0 > 20:
                how = f"it dropped the batch and stays at height {at}, on what it had committed of the short branch"
        off = differing(1, min(net.ord_height(1), long))
        say(f"   attempt {t + 1}: ord stopped after reading {read} blocks of the short branch ({short + 1} with genesis); let go "
            f"on the long one: {how}; its index differs from the node at {len(off)} heights"
            + (f" ({off[0]}..{off[-1]})" if off else ""))
        if dropped:
            say(f"      ord: {dropped[0][-260:]}")
        if (off or dropped or how == "its indexer died") and how != "it committed the short branch whole":
            return read, how, off
    raise SystemExit("the reorganisation did not land inside a batch in any attempt")


if __name__ == "__main__":
    say(f"== tests/check_ord_orphan.py ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    net.up(1)
    tips = net.cli(1, "getchaintips")
    inv = [t for t in tips if t["status"] == "invalid"]
    if inv:  # left by a run of this test that was cut short
        net.cli(1, "reconsiderblock", inv[0]["hash"])
        time.sleep(2)
        tips = net.cli(1, "getchaintips")
    act = next(t for t in tips if t["status"] == "active")
    forks = [t for t in tips if t["status"] in ("valid-fork", "valid-headers") and t["height"] < act["height"] and t["branchlen"] >= 6]
    assert forks, f"node 1 has one branch only: {tips}"
    x = max(forks, key=lambda t: t["branchlen"])
    fork = x["height"] - x["branchlen"]
    long, y1 = act["height"], net.cli(1, "getblockhash", fork + 1)
    say(f"node 1: active chain of {long} blocks, a branch of {x['branchlen']} blocks that leaves it after height {fork}")

    # ---- 1. the fault, and what is seen of it
    after = int(os.environ.get("ORPHAN_AFTER") or x["branchlen"] // 2)
    read, how, off = into_batch(y1, long, after)
    if not off and how != "its indexer died":  # this ord checks the link inside a batch
        assert how == "it indexed to the node's tip" and not differing(0, long) and net.ord_sick(1, 200) is None and not net.ord_panic(1)
        say(f"this ord dropped the batch whose blocks did not link and read the chain again: every height 0..{long} "
            "the index has the node's block; net.ord_sick: None")
        say("OK")
        sys.exit(0)
    at = net.ord_height(1)
    say(f"ord's index: height {at} of {long}; {how}" + ("" if off else
        " (inside the batch that read both branches: nothing of it was committed, so what the index holds is sound)"))
    died = net.ord_panic(1)
    if died:
        say(f"   its log: {died}")
    for line in net.ord_autopsy(1):
        say(f"   {line}")
    why = net.ord_sick(1, 200)
    say(f"net.ord_sick: {why}")
    assert why and (why.startswith(net.SURE) or not died), why
    time.sleep(6)
    assert net.ord_sick(1, 200) == why  # it does not get over it
    say("six seconds later: the same" + ("" if why.startswith(net.SURE) else
                                         " (a reorganisation deeper than ord's savepoints: the index is rebuilt after a minute of that)"))

    # ---- 2. any process that waits for the index gets it rebuilt
    t0 = time.time()
    net.ord_wait(1, timeout=900)
    aside = sorted(p.name for p in (net.RUN / "ord1").glob("index-set-aside-*"))
    say(f"net.ord_wait: the index is at the node's tip after {time.time() - t0:.0f}s; set aside: {aside or 'nothing (its server was started again, the index kept)'}")
    off = differing(0, long)
    assert not off and net.ord_sick(1, 200) is None and not net.ord_panic(1), (off, net.ord_sick(1, 200))
    say(f"every height 0..{long}: the index has the node's block; net.ord_sick: None")
    if aside:
        say(f"   the record kept with the index that was set aside:")
        for line in (net.RUN / "ord1" / [a for a in aside if a.endswith(".txt")][-1]).read_text().splitlines():
            say(f"      {line[:400]}")

    # ---- 3. the same fault found at start (chain.py up, pipeline.sh refresh)
    if "again" in sys.argv:
        into_batch(y1, long, after)
        net.ord_stop(1)  # as `chain.py stop-helpers` leaves it
        import subprocess
        env = dict(os.environ, MINERS="0", AGENTS="0", MINDS="0")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "chain.py"), "up", "1"], env=env, capture_output=True, text=True, timeout=1800)
        say("chain.py up 1 on that index:")
        for line in (r.stdout + r.stderr).splitlines():
            say(f"   {line[:300]}")
        assert r.returncode == 0, r.returncode
        assert not differing(0, long) and net.ord_sick(1, 200) is None and not net.ord_panic(1)
        say(f"after it: every height 0..{long} the node's block; net.ord_sick: None ("
            + ("the tooling rebuilt the index" if "set aside and built again" in r.stdout else
               "started again, ord found its committed tip off the node's chain and went back to a savepoint by itself") + ")")
    say("OK")
