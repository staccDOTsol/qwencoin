"""From nothing to the live graph: wipe run/, start three nodes and their ords, mine the chain
from genesis until every node holds spendable coins, then run the scenario (tools/demo.py).

    PYTHONPATH=. python3 tools/launch.py [--keep]

Every block is mined and validated with SHA-256 in the weights; expect about 13 minutes on 2 idle cores.
"""
import json, shutil, subprocess, sys, time
sys.path.insert(0, "tools")
import net

if "--keep" not in sys.argv:
    net.down()
    subprocess.run(["pkill", "-f", f"--data-dir {net.RUN}/ord"])  # this network's ords only
    time.sleep(3)
    shutil.rmtree(net.RUN, ignore_errors=True)
t0 = time.time()
net.up(3)
for i in (1, 2, 3):
    net.ord_up(i)
time.sleep(5)
addrs = {}
for i in (1, 2, 3):
    net.ord_wait(i)
    net.ord_wallet(i, "create")
    addrs[i] = net.ord_wallet(i, "receive")["addresses"][0]
json.dump(addrs, open(net.RUN / "addrs.json", "w"))
print("genesis", net.cli(1, "getblockhash", 0), "sha256:", open(net.datadir(1) / "debug.log").read().split("Using the '")[1].split("'")[0], flush=True)
for i in (1, 2, 3):  # each node mines its own stake, on the tip all three agree on
    net.settle()
    net.cli(i, "generatetoaddress", 6, addrs[i], 9000000)
for k in range(10):  # then the 100 blocks that mature it
    net.settle()
    net.cli(1 + k % 3, "generatetoaddress", 10, addrs[1 + k % 3], 9000000)
    net.settle()
    print(f"height {net.cli(1, 'getblockcount')} ({time.time() - t0:.0f}s)", flush=True)
for i in (1, 2, 3):
    net.ord_wait(i)
subprocess.run([sys.executable, "tools/demo.py"], check=True)
