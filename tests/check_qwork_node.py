"""The weights-bound proof of work in running nodes (qwencoind on the qwencoin chain).

    python3 tests/check_qwork_node.py [BLOCKS=8]        -> logs/qwork-node.log

A private network of this chain in work/qwork-net (its own ports, nothing shared with run/):

  1  a node does not start without the table, with a file that is not one, or with a table in
     which one weight differs
  2  the genesis block: its hash, and its work value as the node reports it, as the exact numpy
     reference computes it, and as the constant in the source says
  3  blocks mined by the nodes (two miners): every node ends on the same tip; for every block
     the reference recomputes the work value from the raw header, it equals the node's, it is
     under the target, and the block's hash (hashlib) is what the node calls it
  4  native hashing does not count: blocks built on the tip and ground the way Bitcoin is mined
     (SHA-256d of the header under the target, hashlib) are refused as high-hash, by submitblock
     and by submitheader; the tip does not move
  5  a block mined outside the node (template from the node, nonces tried with the C kernel,
     the winner checked with the reference) is accepted and relayed
  6  that block with another nonce is refused
  7  a node restarted keeps the chain; a node restarted with -reindex re-evaluates every block's
     work and arrives at the same tip
  8  a new node synchronises from genesis from the others

The node's SHA-256 runs in the compiled weights; this script's is hashlib. The two sides agree
on every header hash, seed and work value or the test fails.
"""
import hashlib, os, shutil, struct, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NET = ROOT / "work" / "qwork-net"
os.environ.update(QWEN_RUN=str(NET), QWEN_PORT="28733", QWEN_BIND="127.0.0.1", QWEN_SEEDS="none")  # a private network: nobody elsewhere is dialed
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
for _p in (ROOT / "upstream" / "bitcoin" / "test" / "functional", ROOT / "vendor"):
    if (_p / "test_framework").is_dir():
        sys.path.insert(0, str(_p))
        break
import net  # noqa: E402
from hashformer import work as W  # noqa: E402
from test_framework.blocktools import create_block, create_coinbase  # noqa: E402
from test_framework.messages import uint256_from_compact  # noqa: E402

BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GENESIS = "51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5"
GENESIS_WORK = "0083bfd7a2ae5796c139f180b33630684eb71751e4b2b3153ffe288da7739d1d"
LOG = open(ROOT / "logs" / "qwork-node.log", "w")
T0 = time.time()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def target_of(bits: int) -> int:
    return uint256_from_compact(bits)


ref, ctab = W.Table(net.QWORK), W.CTable(net.QWORK)


def work_of(header: bytes, exact=True) -> bytes:
    """A header's work value by the reference (numpy, exact integers) or the C kernel."""
    return W.pow_hash(ref, header) if exact else W.pow_hash(ctab, header, kernel=W.c_kernel)


def start_refused(table: Path, expect: str) -> str:
    d = NET / "refuse"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    r = subprocess.run([str(net.BIN / "qwencoind"), f"-datadir={d}", f"-qwork={table}", "-listen=0", "-server=0", "-connect=0"],
                       capture_output=True, text=True, timeout=300)
    msg = (r.stderr + r.stdout).strip().replace("\n", " ")
    assert r.returncode != 0, f"the node started with {table}"
    assert expect in msg, msg
    shutil.rmtree(d, ignore_errors=True)
    return msg


def wait_same_tip(nodes, timeout=180):
    for _ in range(timeout * 2):
        tips = {net.cli(i, "getbestblockhash") for i in nodes}
        if len(tips) == 1:
            return tips.pop()
        time.sleep(0.5)
    raise AssertionError(f"tips did not converge: {[(i, net.cli(i, 'getblockcount')) for i in nodes]}")


def header_of(i, h: str) -> bytes:
    return bytes.fromhex(net.cli(i, "getblockheader", h, "false"))


def template_block(i, extra: bytes):
    """A block on node i's tip paying an anyone-can-spend output, not yet mined."""
    tmpl = net.cli(i, "getblocktemplate", '{"rules":["segwit"]}')
    cb = create_coinbase(tmpl["height"], extra_output_script=b"\x6a" + bytes([len(extra)]) + extra, halving_period=210000)
    b = create_block(tmpl=tmpl, coinbase=cb, ntime=tmpl["curtime"])
    b.hashMerkleRoot = b.calc_merkle_root()
    return b, tmpl


def mine_qwork(b) -> int:
    """Try nonces under the chain's rule with the C kernel; returns the passes it took."""
    head, target = b.serialize()[:76], target_of(b.nBits)
    h0 = sha256d(head + bytes(4))
    n = 0
    while True:
        out = ctab.evaluate(h0, n)
        v = hashlib.sha256(bytes(x ^ y for x, y in zip(out, h0))).digest()
        if int.from_bytes(v, "little") <= target:
            b.nNonce = n
            return n + 1
        n += 1


def main():
    if NET.exists():
        subprocess.run(["pkill", "-f", "--", f"-datadir={NET}"])
        time.sleep(2)
        shutil.rmtree(NET)
    NET.mkdir(parents=True)
    say(f"table {net.QWORK}: fnv64 {ctab.fnv:#018x}")

    # ---- 1: the table is not optional
    say("1 a node without the right weights does not start")
    m = start_refused(NET / "absent.bin", "cannot be opened")
    say("    no file:", m[:150])
    junk = NET / "junk.bin"
    junk.write_bytes(os.urandom(1 << 16))
    say("    not a table:", start_refused(junk, "is not a work table")[:150])
    forged = NET / "forged.bin"
    shutil.copyfile(net.QWORK, forged)
    with open(forged, "r+b") as f:
        f.seek(24 + 2 * 123_456_789)
        w = f.read(2)
        f.seek(-2, 1)
        f.write(struct.pack("<h", struct.unpack("<h", w)[0] ^ 1))
    say("    one weight changed (of 264,241,152):", start_refused(forged, "has checksum")[:190])
    forged.unlink()
    junk.unlink()

    # ---- 2: genesis
    net.up(3)
    say(f"2 three nodes up ({time.time() - T0:.0f}s); genesis")
    g = net.cli(1, "getblockhash", 0)
    hdr = header_of(1, g)
    gh = net.cli(1, "getblockheader", g)
    assert g == GENESIS == sha256d(hdr)[::-1].hex(), g
    assert gh["powhash"] == GENESIS_WORK == work_of(hdr)[::-1].hex() == work_of(hdr, exact=False)[::-1].hex(), gh
    assert int(gh["powhash"], 16) <= target_of(int(gh["bits"], 16)) < int(g, 16)
    say(f"    hash {g}\n    work {gh['powhash']}  (node == numpy reference == C kernel == the constant; under the target {gh['target'][:12]}..; the hash itself is not)")

    # ---- 3: mining, two miners, three nodes
    say(f"3 mining {BLOCKS} blocks on two nodes")
    addr = {}
    for i in (1, 2):
        if "w" not in net.cli(i, "listwallets"):
            net.cli(i, "createwallet", "w")
        addr[i] = net.cli(i, "getnewaddress", wallet="w")
    t_mine, tries = time.time(), 0
    for k in range(BLOCKS):
        i = 1 + k % 2
        wait_same_tip((1, 2, 3))
        h = net.cli(i, "generatetoaddress", 1, addr[i], 1_000_000)[0]
        nonce = net.cli(i, "getblockheader", h)["nonce"]
        tries += nonce + 1
        say(f"    n{i} mined height {net.cli(i, 'getblockcount')} {h[:16]}.. at nonce {nonce}")
    dt = time.time() - t_mine
    tip = wait_same_tip((1, 2, 3))
    height = net.cli(3, "getblockcount")
    assert height == BLOCKS
    say(f"    {tries} passes in {dt:.0f}s: {1000 * dt / tries:.0f} ms per attempt in the node (the pass, one SHA-256 in the weights, the template); "
        f"all three nodes at height {height}, tip {tip[:16]}..")
    worst = 0
    for hgt in range(height + 1):
        h = net.cli(3, "getblockhash", hgt)
        hdr = header_of(3, h)
        info = [net.cli(i, "getblockheader", h) for i in (1, 2, 3)]
        v = work_of(hdr)
        assert sha256d(hdr)[::-1].hex() == h
        assert {x["powhash"] for x in info} == {v[::-1].hex()}, (hgt, info[0]["powhash"], v[::-1].hex())
        assert int.from_bytes(v, "little") <= target_of(int(info[0]["bits"], 16))
        worst = max(worst, int(h, 16).bit_length())
    say(f"    every header 0..{height}: work value by the reference == all three nodes', and under the target; "
        f"hash by hashlib == the node's. (The hashes themselves are ordinary numbers: the largest has {worst} bits.)")

    # ---- 4: native hashing does not count
    say("4 blocks ground the Bitcoin way")
    refused = 0
    for k in range(4):
        b, _ = template_block(1, b"native%d" % k)
        b.nNonce = 0
        b.solve()  # the test framework's miner: SHA-256d of the header under the target
        hdr = b.serialize()[:80]
        assert int.from_bytes(sha256d(hdr), "little") <= target_of(b.nBits)
        if int.from_bytes(work_of(hdr, exact=False), "little") <= target_of(b.nBits):
            say("    (one of them happens to be work as well: 1 chance in 256; skipped)")
            continue
        r1 = net.cli(1, "submitblock", b.serialize().hex())
        try:
            net.cli(2, "submitheader", hdr.hex())
            r2 = "accepted"
        except RuntimeError as e:
            r2 = str(e)
        assert r1 == "high-hash", r1
        assert "high-hash" in r2 or "proof of work failed" in r2, r2
        refused += 1
        if k == 0:
            say(f"    hash {sha256d(hdr)[::-1].hex()[:20]}.. is under the target; submitblock: {r1!r}; submitheader: {r2.splitlines()[-1][:80]!r}")
    assert refused >= 3 and wait_same_tip((1, 2, 3)) == tip
    say(f"    {refused} of {refused} refused, the tip did not move")

    # ---- 5, 6: a miner outside the node
    say("5 a block mined outside the node")
    b, tmpl = template_block(2, b"outside")
    n = mine_qwork(b)
    hdr = b.serialize()[:80]
    assert work_of(hdr) == work_of(hdr, exact=False)
    bad = create_block(tmpl=tmpl, coinbase=b.vtx[0], ntime=b.nTime)
    bad.hashMerkleRoot, bad.nNonce = b.hashMerkleRoot, b.nNonce + 1
    while int.from_bytes(work_of(bad.serialize()[:80], exact=False), "little") <= target_of(bad.nBits):
        bad.nNonce += 1
    r_bad = net.cli(2, "submitblock", bad.serialize().hex())
    assert r_bad == "high-hash", r_bad
    r = net.cli(2, "submitblock", b.serialize().hex())
    assert r in ("", None), r
    tip = wait_same_tip((1, 2, 3))
    assert tip == sha256d(hdr)[::-1].hex() and net.cli(1, "getblockcount") == BLOCKS + 1
    say(f"    {n} passes with the C kernel, winner confirmed by the reference; accepted by n2, relayed: all three at height {BLOCKS + 1}, tip {tip[:16]}..")
    say(f"6 the same block with nonce {bad.nNonce} instead of {b.nNonce}: {r_bad!r}")

    # ---- 7: restart, reindex
    say("7 restart and reindex")
    net.cli(3, "stop")
    time.sleep(6)
    t = time.time()
    net.up(3)
    assert net.cli(3, "getbestblockhash") == tip
    say(f"    n3 restarted: same tip ({time.time() - t:.0f}s)")
    net.cli(3, "stop")
    time.sleep(6)
    t = time.time()
    subprocess.run([str(net.BIN / "qwencoind"), f"-datadir={net.datadir(3)}", "-reindex", "-daemon"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=300)
    for _ in range(600):
        if net.cli(3, "getbestblockhash", check=False, timeout=20) == tip:
            break
        time.sleep(1)
    assert net.cli(3, "getbestblockhash") == tip
    say(f"    n3 restarted with -reindex: every block's work evaluated again, same tip ({time.time() - t:.0f}s)")

    # ---- 8: a new node
    say("8 a new node from genesis")
    t = time.time()
    net.up(4)
    assert wait_same_tip((1, 2, 3, 4), 300) == tip
    say(f"    n4 synchronised {BLOCKS + 1} blocks from its peers: same tip ({time.time() - t:.0f}s), peers {net.cli(4, 'getconnectioncount')}")
    net.down()
    say(f"OK ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        net.down()
        raise
