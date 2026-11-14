"""A private two-node network of the qwencoin chain for the tool tests (work/tools-net, its own
ports), kept between runs: mining its first hundred blocks is most of the cost.

    python3 tests/tools_net.py up [HEIGHT=106]     nodes, their ords and wallets; mines to HEIGHT if below
    python3 tests/tools_net.py down

Both nodes mine until HEIGHT (the first coinbases are then spendable); after that a block is
mined only when a test asks for one (`block`).
"""
import json, os, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NET = ROOT / "work" / "tools-net"
os.environ.update(QWEN_RUN=str(NET), QWEN_PORT="38733", QWEN_ORD_PORT="38080", QWEN_BIND="127.0.0.1", QWEN_SEEDS="none")  # private: nobody elsewhere is dialed
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import net  # noqa: E402

NODES = (1, 2)


def addrs() -> dict:
    f = NET / "addrs.json"
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def up(height: int = 106, log=print) -> dict:
    NET.mkdir(parents=True, exist_ok=True)
    net.up(len(NODES))
    a = addrs()
    for i in NODES:
        net.ord_ensure(i)
    for i in NODES:
        net.ord_wait(i, timeout=900)
        net.ord_wallet_ready(i, log)
        if i not in a:
            a[i] = net.ord_wallet(i, "receive")["addresses"][0]
            (NET / "addrs.json").write_text(json.dumps(a))
    t0, h0 = time.time(), net.cli(1, "getblockcount")
    if h0 < height:
        if h0 < 3:  # the first coinbases to mature are node 1's
            net.cli(1, "generatetoaddress", 3 - h0, a[1], 10_000_000)
            net.settle(len(NODES), 300)
        log(f"mining from height {h0} to {height} on both nodes")

        def miner(i):
            while net.cli(i, "getblockcount") < height:
                net.cli(i, "generatetoaddress", 1, a[i], 24)

        ts = [threading.Thread(target=miner, args=(i,)) for i in NODES]
        [t.start() for t in ts]
        while any(t.is_alive() for t in ts):
            time.sleep(30)
            log(f"  height {net.cli(1, 'getblockcount')} ({time.time() - t0:.0f}s)")
        net.settle(len(NODES), 300)
        log(f"mined {net.cli(1, 'getblockcount') - h0} blocks in {time.time() - t0:.0f}s")
    for i in NODES:
        net.ord_wait(i, timeout=1800)
    for _ in range(120):  # the two find each other a moment after they start
        if all(net.cli(i, "getconnectioncount") >= 1 for i in NODES):
            break
        time.sleep(1)
    if any(net.cli(i, "getmempoolinfo")["size"] for i in NODES):  # what an earlier run left unconfirmed is confirmed before this one begins
        left = level()
        net.cli(1, "generatetoaddress", 1, a[1], 10_000_000)
        net.settle(len(NODES), 300)
        for i in NODES:
            net.ord_wait(i, timeout=600)
        log(f"what an earlier run had left in the mempools ({left} transaction(s) one node had and the other not) is in block {net.cli(1, 'getblockcount')}")
    return a


def level() -> int:
    """Each node is handed what the other's mempool has and its own has not, parents first. (A node relays a
    transaction when it first takes it in: what a node that was stopped kept in its mempool, it does not offer
    again, and the other never hears of it.) Returns how many were handed over."""
    pools, n = {j: net.cli(j, "getrawmempool", "true") for j in NODES}, 0
    for j in NODES:
        for k in NODES:
            for t in sorted((t for t in pools[j] if k != j and t not in pools[k]), key=lambda t: pools[j][t]["ancestorcount"]):
                try:
                    net.cli(k, "sendrawtransaction", net.cli(j, "getrawtransaction", t))
                    n += 1
                except RuntimeError:  # it spends what something in the other's mempool spends: it stays where it is
                    pass
    return n


def block(i: int = 1, n: int = 1) -> int:
    """Mine n blocks on node i (confirming what is in its mempool) and wait until both ords have them."""
    a = addrs()
    for k in range(240):  # what one node sent has to reach the miner's mempool first (relay takes seconds)
        if len({tuple(sorted(net.cli(j, "getrawmempool"))) for j in NODES}) == 1:
            break
        if k % 20 == 19 and not level():  # ten seconds and still not the same: handed over; nothing to hand over: they will not be
            break
        time.sleep(0.5)
    net.cli(i, "generatetoaddress", n, a[i], 10_000_000)
    net.settle(len(NODES), 300)
    for j in NODES:
        net.ord_wait(j, timeout=600)
    return net.cli(i, "getblockcount")


def down():
    import subprocess
    subprocess.run(["pkill", "-f", "--", f"--data-dir {NET}/ord"])
    net.down()


if __name__ == "__main__":
    if sys.argv[1] == "up":
        up(int(sys.argv[2]) if len(sys.argv) > 2 else 106)
        print({i: (net.cli(i, "getblockcount"), net.ord_wallet(i, "balance")["cardinal"]) for i in NODES})
    else:
        down()
