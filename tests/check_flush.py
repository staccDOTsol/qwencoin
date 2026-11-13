"""A node that is killed hard keeps what it wrote to disk, and it writes its chain state once a
day unless told to: tools/chain.py tells it every ten minutes (`gettxoutsetinfo none`).

    QWEN_RUN=<folder> QWEN_PORT=9733 QWEN_ORD_PORT=19080 python3 tests/check_flush.py

On node 1 of QWEN_RUN (a private chain; it mines two blocks twice):
  1  blocks mined, the node killed (SIGKILL), started again: its log says which height its chain
     state on disk was at ("Loaded best chain"): the height from before the blocks. What it had
     of them in its block index (written hourly) it connects again; the rest it has to be sent
  2  the same with the call tools/chain.py makes while mining: the height with the blocks
  3  the node's miner itself (tools/chain.py mine, told to make the call every 10 s instead of
     every 600): blocks it mined are in the chain state on disk after the node is killed
"""
import os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402


def say(*a):
    print(*a, flush=True)


def kill_hard():
    r = subprocess.run(["pgrep", "-f", "--", f"-datadir={net.datadir(1)}"], capture_output=True, text=True)
    for pid in r.stdout.split():
        os.kill(int(pid), signal.SIGKILL)
    time.sleep(2)


def loaded() -> int:
    """The height of the chain state the node found on disk when it last started."""
    lines = [l for l in (net.datadir(1) / "debug.log").read_text(errors="replace").splitlines() if "Loaded best chain" in l]
    return int(re.search(r"height=(\d+)", lines[-1]).group(1))


if __name__ == "__main__":
    say(f"== tests/check_flush.py ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    net.up(1)
    def address():  # a wallet of the node's own to mine to (after a restart none is loaded)
        if not net.cli(1, "listwallets"):
            if not isinstance(net.cli(1, "loadwallet", "flushtest", check=False), dict):
                net.cli(1, "createwallet", "flushtest", check=False)
        return net.cli(1, "getnewaddress", wallet=net.cli(1, "listwallets")[0])

    addr = address()
    h0 = net.cli(1, "getblockcount")
    net.cli(1, "generatetoaddress", 2, addr, 100000)
    h1 = net.cli(1, "getblockcount")
    assert h1 == h0 + 2, (h0, h1)
    kill_hard()
    net.up(1)
    a = loaded()
    now = net.cli(1, "getblockcount")
    say(f"1 two blocks mined on height {h0}, the node killed, started again: the chain state it found on disk was at height {a}; "
        + (f"it connected the rest again from its block files and is at {now}" if now == h1 else
           f"it is at {now}: the two blocks are gone with it (their entries in the block index were not on disk either)"))
    assert a <= h0, a
    net.cli(1, "generatetoaddress", 2, addr, 100000)
    h2 = net.cli(1, "getblockcount")
    t0 = time.time()
    info = net.cli(1, "gettxoutsetinfo", "none")
    took = time.time() - t0
    kill_hard()
    net.up(1)
    b = loaded()
    say(f"2 two more blocks (height {h2}), then `gettxoutsetinfo none` ({took:.1f}s, {info['txouts']} outputs), the node killed, started again: "
        f"the chain state on disk was at height {b}")
    assert b == h2, (b, h2)

    stop = net.RUN / "STOP"  # a network that was taken down leaves it: a miner would end at once
    had = stop.exists()
    stop.unlink(missing_ok=True)
    miner = subprocess.Popen([sys.executable, str(ROOT / "tools" / "chain.py"), "mine", "1"], env=dict(os.environ, QWEN_FLUSH_EVERY="10"),
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    try:
        t0 = time.time()
        while time.time() - t0 < 600 and net.cli(1, "getblockcount") < h2 + 2:
            time.sleep(1)
        h3 = net.cli(1, "getblockcount")
        assert h3 >= h2 + 2, (h2, h3)
        time.sleep(25)  # two of its calls at the least after those blocks
    finally:
        miner.terminate()
        miner.wait(timeout=60)
        if had:
            stop.write_text("")
    h4 = net.cli(1, "getblockcount")
    kill_hard()
    net.up(1)
    c = loaded()
    say(f"3 the node's miner (tools/chain.py mine 1, QWEN_FLUSH_EVERY=10) for {time.time() - t0:.0f}s: it mined from {h2} to {h4}; "
        f"the node killed, started again: the chain state on disk was at height {c}")
    assert c >= h3, (c, h3)
    say("OK")
