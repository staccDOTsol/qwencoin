"""Mine a genesis block with the model: every hash below is a run of the compiled weights.

    PYTHONPATH=. python3 tools/genesis.py "<coinbase text>" [--time T] [--bits 0x2000ffff] [--qwork dist/qwork.bin]

Prints the constants for kernel/chainparams.cpp. The coinbase pays 50 coins to the BIP341 NUMS
point (nobody knows its key; the genesis output is unspendable in Bitcoin anyway).

With --qwork the proof of work is the chain's: a pass through the weights of Qwen3.5-0.8B per
nonce (hashformer/work.py). The nonces are tried with the C kernel; the winner is evaluated
again with the exact numpy reference, and both SHA-256 implementations (the model's and
hashlib's) must give the same work value.
"""
import argparse, hashlib, struct, time
from hashformer import executor as X

NUMS = bytes.fromhex("50929b74c1a04954b78b4b6035e97a5e078a5a0f28ec96d547bfee9ace803ac0")


def push(data: bytes) -> bytes:
    if len(data) <= 75:
        return bytes([len(data)]) + data
    assert len(data) < 256
    return b"\x4c" + bytes([len(data)]) + data


def coinbase(text: bytes, reward: int) -> bytes:
    script_sig = push(struct.pack("<I", 486604799)) + push(b"\x04") + push(text)
    assert 2 <= len(script_sig) <= 100
    spk = push(b"\x02" + NUMS) + b"\xac"  # <pubkey> OP_CHECKSIG
    return (struct.pack("<i", 1) + b"\x01" + bytes(32) + b"\xff\xff\xff\xff"
            + bytes([len(script_sig)]) + script_sig + b"\xff\xff\xff\xff"
            + b"\x01" + struct.pack("<q", reward) + bytes([len(spk)]) + spk + struct.pack("<I", 0)), spk


def target_of(bits: int) -> int:
    return (bits & 0xFFFFFF) << 8 * ((bits >> 24) - 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--time", type=int, default=int(time.time()))
    ap.add_argument("--bits", type=lambda s: int(s, 0), default=0x2000FFFF)
    ap.add_argument("--qwork", help="the work table: mine under the chain's weights-bound proof of work")
    a = ap.parse_args()
    tx, spk = coinbase(a.text.encode(), 50 * 10**8)
    txid = X.sha256d(tx)
    assert txid == hashlib.sha256(hashlib.sha256(tx).digest()).digest()
    head = struct.pack("<i", 1) + bytes(32) + txid + struct.pack("<II", a.time, a.bits)
    t0, k0, nonce = time.time(), X.tokens(), 0
    if a.qwork:
        from hashformer import work as W
        ct, ref = W.CTable(a.qwork), W.Table(a.qwork)
        h0 = X.sha256d(head + bytes(4))  # the seed, in the weights
        assert h0 == hashlib.sha256(hashlib.sha256(head + bytes(4)).digest()).digest()
        while True:
            out = ct.evaluate(h0, nonce)
            v = X.sha256(bytes(x ^ y for x, y in zip(out, h0)))
            if int.from_bytes(v, "little") <= target_of(a.bits):
                break
            nonce += 1
        hdr = head + struct.pack("<I", nonce)
        assert out == W.evaluate(ref, h0, nonce), "the C kernel and the reference disagree"
        assert v == W.pow_hash(ref, hdr) == W.pow_hash(ref, hdr, sha256=X.sha256)
        h = X.sha256d(hdr)
        assert h == hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
        print(f"text      {a.text!r}")
        print(f"time      {a.time}\nbits      {a.bits:#x}\nnonce     {nonce}")
        print(f"merkle    {txid[::-1].hex()}\nhash      {h[::-1].hex()}\nwork      {v[::-1].hex()}")
        print(f"table     fnv64 {ct.fnv:#018x}")
        print(f"script    {spk.hex()}\nheader    {hdr.hex()}\nblock     {(hdr + bytes([1]) + tx).hex()}")
        print(f"mined in {time.time() - t0:.1f}s: {nonce + 1} passes through the weights, {X.tokens() - k0} model tokens of SHA-256")
        raise SystemExit
    while True:
        h = X.sha256d(head + struct.pack("<I", nonce))
        if int.from_bytes(h, "little") <= target_of(a.bits):
            break
        nonce += 1
    hdr = head + struct.pack("<I", nonce)
    assert h == hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
    print(f"text      {a.text!r}")
    print(f"time      {a.time}\nbits      {a.bits:#x}\nnonce     {nonce}")
    print(f"merkle    {txid[::-1].hex()}\nhash      {h[::-1].hex()}")
    print(f"script    {spk.hex()}\nheader    {hdr.hex()}")
    print(f"mined in {time.time() - t0:.1f}s: {nonce + 1} header hashes, {X.tokens() - k0} model tokens")
