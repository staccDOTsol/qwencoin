"""ord's wallet under what a fast chain and several users of one node do to it (tools/net.py).

    python3 tests/check_ord_busy.py        on the private two-node network of tests/tools_net.py

1. Several processes use node 1's wallet at once. Started bare, `ord wallet` refuses all but one
   ("failed to open wallet database"); through net.ord_wallet every command is carried out, also
   while bare commands (a process that does not take the lock) keep hammering the same wallet.
2. A block has just arrived. With ord's default polling (5 s) its index is behind and a bare
   wallet command is refused ("blocks behind `bitcoind`"); net.ord_wallet waits for the index and
   carries the command out. Started by net.ord_up the server polls every half second.
"""
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import tools_net as T  # noqa: E402  (sets the test network's folder and ports)

net = T.net


def bare(i, *args):
    cmd = net.ord_base(i) + ["wallet", "--server-url", f"http://127.0.0.1:{net.ordport(i)}"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def server(i, old: bool):
    """Node i's ord server, started as before (ord's default polling) or as net.ord_up starts it."""
    subprocess.run(["pkill", "-f", "--", f"--data-dir {net.RUN}/ord{i} --index-sats server"])
    if old:
        while net.ord_running(i):  # it closes its index first
            time.sleep(0.5)
        log = open(net.RUN / f"ord{i}" / "server.log", "a")
        p = subprocess.Popen(net.ord_base(i) + ["server", "--http-port", str(net.ordport(i)), "--address", "127.0.0.1"],
                             stdout=log, stderr=log, start_new_session=True)
        (net.RUN / f"ord{i}" / "server.pid").write_text(str(p.pid))
    else:  # at once, while the one before may still be closing: net.ord_ensure has to get it up all the same
        net.ord_ensure(i)
    net.ord_wait(i, timeout=600)


if __name__ == "__main__":
    print(f"== tests/check_ord_busy.py ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    a = T.up(106)
    print(f"test network at height {net.cli(1, 'getblockcount')}")

    # ---- 1. one wallet, many users
    with ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(lambda k: bare(1, "balance"), range(16)))
    refused = sum(1 for r in rs if r.returncode and net.ORD_OPEN in r.stderr)
    other = [r.stderr[-200:] for r in rs if r.returncode and net.ORD_OPEN not in r.stderr and net.ORD_BEHIND not in r.stderr]
    print(f"16 bare `ord wallet balance` at once on node 1: {refused} refused with {net.ORD_OPEN!r}")
    assert refused and not other, other
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:  # threads here; the lock is a file lock, so processes are held the same way
        got = list(ex.map(lambda k: net.ord_wallet(1, "balance" if k % 2 else "receive"), range(16)))
    assert all(isinstance(g, dict) for g in got), got
    print(f"16 through net.ord_wallet at once: all carried out ({time.time() - t0:.0f}s)")
    stop = []

    def hammer():
        n = 0
        while not stop:
            bare(1, "balance")
            n += 1
            time.sleep(0.2)
        return n

    with ThreadPoolExecutor(6) as ex:
        hs = [ex.submit(hammer) for _ in range(2)]
        t0 = time.time()
        try:
            got = list(ex.map(lambda k: net.ord_wallet(1, "receive"), range(8)))
        finally:
            stop.append(1)
        n = sum(h.result() for h in hs)
    assert all(isinstance(g, dict) and g.get("addresses") for g in got), got
    print(f"8 through net.ord_wallet while 2 loops of bare commands held the wallet ({n} of them): all carried out ({time.time() - t0:.0f}s)")

    # ---- 2. a block has just arrived
    server(1, old=True)
    behind = 0
    for k in range(3):
        T.net.cli(1, "generatetoaddress", 1, a[1], 10_000_000)
        r = bare(1, "balance")
        behind += int(bool(r.returncode) and net.ORD_BEHIND in r.stderr)
        assert not r.returncode or net.ORD_BEHIND in r.stderr, r.stderr[-300:]
        T.net.cli(1, "generatetoaddress", 1, a[1], 10_000_000)
        t0 = time.time()
        b = net.ord_wallet(1, "balance")
        assert isinstance(b, dict) and "cardinal" in b, b
        print(f"  block {net.cli(1, 'getblockcount')}: a bare command right after the block before it "
              f"{'was refused (index behind)' if r.returncode else 'got through'}; net.ord_wallet right after this one: carried out in {time.time() - t0:.1f}s")
    print(f"ord server with its default polling: {behind} of 3 bare commands refused right after a block; net.ord_wallet: 3 of 3 carried out")
    assert behind, "the index was never behind: nothing was tested"
    server(1, old=False)
    late = 0
    for k in range(3):
        T.net.cli(1, "generatetoaddress", 1, a[1], 10_000_000)
        time.sleep(1.0)
        late += int(net.ord_get(1, "/status")["height"] < net.cli(1, "getblockcount"))
    print(f"ord server as net.ord_up starts it (polling every 500 ms): index behind one second after a block in {late} of 3")
    assert late == 0
    net.settle(len(T.NODES), 300)
    print("OK")
