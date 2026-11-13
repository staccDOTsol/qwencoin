
## 13. Left out of the test runs, and why

Every SHA-256 compression in these tests is 441 model tokens, about 4.4 ms in the executor, so a
megabyte of hashing is a minute. Bitcoin's 672 unit cases were not all run. Stopped or not run:

- `transaction_tests/test_big_witness_transaction`: 4,500 inputs signed and checked without the
  sighash cache, about 16 hours. Stopped after 35 minutes.
- `merkle_tests/merkle_test`: random blocks of up to 4,016 transactions with 16 branch
  recomputations each, about 4 hours. The five fixed merkle cases pass.
- `sighash_tests/sighash_test`, `sighash_anyprevout_legacy`, `sighash_anyprevout_taproot`: 50,000
  random transactions each, hours. `sighash_from_data` (Bitcoin's 500 reference vectors) passes.
- `ctvhash_tests`: 102 vectors, each with 200 mutation rounds over multi-kilobyte transactions,
  about 2.5 hours. An earlier pass ran it for 17 minutes with no failure. CTV itself is covered by
  `feature_checktemplateverify.py` and by the covenant test.
- `feature_taproot.py`: thousands of spends; one of its funding transactions (50 inputs, 641
  outputs) takes 16 minutes to sign and validate. Stopped after 40 minutes. Tapscript, CSFS, CAT
  and CTV are exercised by the three script-upgrade tests, `script_tests` and the covenant test.
- `mining_basic.py`: timed out at 25 minutes in the stage that fills 4 MB blocks; its earlier
  stages passed.
- Suites that do not hash, and suites that mine a 100-block chain per case (wallet, validation,
  mempool, miner): not selected.

Bound to Bitcoin's own chain rather than to the code:

- `bip324_tests` fails here, as it would on any chain with another network magic: BIP324 salts its
  key derivation with the magic and the vectors are Bitcoin's. With Bitcoin's magic put back in a
  scratch build of the test binary (nothing else changed) the suite passes, so the in-weights
  HKDF-SHA256 and the ellswift hashes are right. `p2p_v2_transport.py` and `p2p_v2_encrypted.py`
  then run the node's BIP324 against the framework's independent implementation.
- `miner_tests` and the other suites that replay blocks mined on Bitcoin's genesis: not adapted.

One upstream functional test was changed: `rpc_txoutproof.py` asks a node's transaction index for
a proof seconds after start. That index re-reads every block, each block hash is a run of the
model, and under load it was not ready (the first run failed with "Transaction not yet in block";
every proof assertion before that step had passed). The test now waits for the index it queries.

## 14. Defects found while verifying (fixed in this tree)

1. **Retarget overflow (consensus).** With the limit at 2^248, Bitcoin's `target * timespan`
   wraps at 256 bits. The first retarget after fast blocks would have set `0x1e0337ae`, 20,000
   times the work, instead of `0x1f3fffc0`, 4 times. `pow.cpp` now computes the same quotient
   without the product; `pow_tests` has the limit cases and Bitcoin's own four vectors.
2. **Precomputed SHA-256 states in libsecp256k1.** Upstream hard-codes eleven tagged-hash
   midstates (`BIP0340/challenge` and others): SHA-256 outputs that the model never computed. Each
   constant was checked against an independent computation of its tag, then removed; the model
   computes the midstate at first use. `tests/check_binary.py` scans for them.
3. **Holes in the graph's accounting.** Counting one stake per bond-shaped inscription let two
   messages share one output, let anyone replay a signed vouch onto other coins, let anyone pin an
   id on a key it did not hold, and let a bond name its own fee (a "burn" that pays the miner) or
   a one-block delay. The rules in `graph/index.py` and `protocol.bond_of` close each one; all are
   attacks in `tests/check_sybil.py`.
4. **Index trusted the mempool.** A stake being spent by an unconfirmed transaction was marked
   exited for good, so a bond that started unbonding could dodge its slash. The index now reads
   confirmed locations only.
5. **The demo's miner raced block relay.** Three nodes mining in turn without waiting for the
   previous block made 2 stale blocks in 6. All three nodes still converged on one tip, but the
   script's height arithmetic was off; miners now wait for a common tip.
6. **ord 0.29.1 against Core 29.** `importdescriptors` rejects a label on an internal descriptor;
   one line in ord's wallet (`patches/ord-0.29.1-qwencoin.patch`).
7. **Printed tokens tied with unused chip rows (the merge).** The chip's vocabulary block has
   246 control ids no token uses; their embedding rows carried codes, and those codes are the ones
   the bridge gives the text tokens it prints, so the head scored a printed digit and a dead chip
   row equally. The teacher-forced check reports the runner-up next to the winner, which showed it
   (1.000 of the winner instead of 0.889); the merge now writes those rows as zeros.

8. **The feed was in the order the index learned it.** ord lists inscriptions newest first, so an
   index built from scratch held the posts backwards and a node that had just started showed the
   oldest messages as the latest. Posts are now ordered by ord's inscription number, the chain's
   own order (`graph/index.py: feed`). Found by the first run of a node's model loop.
9. **A failed send counted as spent.** The tools counted a call's sats when it was approved, not
   when it went through, so a refused payment still used up a spending limit. Counted on success now.
10. **ord's reason lost behind its backtrace.** A failed `ord wallet` command reached the model as
   the last line of a Rust backtrace (`15: _start`). The node now passes on ord's `error:` line.
11. **Float32 tensors rounded by the merge.** The base stores a DeltaNet layer's `A_log` and head
   norm in float32; the merge wrote everything as bf16. Identical when the model runs in bf16 (the
   loader rounds them the same way), a small change in float32. They are copied bit for bit now
   (`tests/check_remerge.py` compares every tensor with the checkpoint the long checks ran on).
12. **ord's wallet on a fast chain with several users.** The first hour of the live chain had a
   block every four seconds, and a node's wallet is used by its agent, its model and whoever asks
   it something. `ord wallet` opens its database exclusively ("failed to open wallet database")
   and refuses to start while its index is a block behind the node, after waiting one second
   ("blocks behind `bitcoind`"); the server polled the node every five seconds. Two of the
   0.8B's twelve calls in its first live check were lost to this (the address and the payment).
   Wallet commands now take a per-node file lock, are started again when ord refuses for either
   reason (both are decided before the command does anything), and the server polls every half
   second. `tests/check_ord_busy.py` reproduces both refusals with bare commands and shows every
   command carried out through `tools/net.py`.
13. **The plain-chat probe asked another question than the base was asked.** Once a checkpoint's
   template declared the node's functions, `tests/check_merge.py` rendered its "no tool" prompt
   through that template (1,981 tokens instead of 30) and compared the answer with the base's at
   that offset: 0 of 24 by construction. The probe now passes an empty tool list, asserts that
   this renders the base's own prompt token for token, and prints separately what the model does
   when its own template speaks. The first 0.8B log has both lines; the corrected one reads 4 of 4.
14. **ord's index left on an abandoned chain.** ord undoes a reorganisation about ten blocks deep.
   A node that had mined by itself and then met the network left its whole chain, and its index
   stayed on the dead one for good (found by starting a public node for real: 29 of its own
   blocks, then 153 of the network's). `tools/net.py` now compares the block the index holds at
   its height with the node's; when they differ three times in a row the index is set aside and
   built again from the node's chain (15 seconds for 162 blocks).
15. **A fresh node mined before it met anyone.** What it mined from the first block was thrown
   away on contact, and caused defect 14. A node with no block yet that knows of nodes elsewhere
   (the file `SEEDS`), or is told to wait (`QWEN_MINE_WAIT`, for a node that is dialed), now
   mines only once a peer has said where it is and the chain under the peers' headers is fetched.
16. **Starting ord's server while the one before it was closing.** The new process found the
   index locked and gave up, and nothing started a third. `net.ord_ensure` starts it again until
   it answers.
17. **Processes keep the code they were started with.** Fixes copied into a running installation
   reached new processes only. `tools/chain.py` records which files the miners, agents, ord
   servers and node models were started from; a full run with other files stops them first
   (the nodes keep running) and `pipeline.sh refresh` does so on request.
18. **ord took in an orphan and its indexer died of it later.** ord checks that a block links to
   the one before it against its index as committed. Inside one batch of its indexer nothing is
   committed yet, so when a reorganisation falls between two blocks of a batch, both branches go
   in one after the other. The index then holds a block the node no longer has and lacks the
   outputs of the block that replaced it; at the first transaction that spends one of those the
   index thread dies (`assertion failed: !self.index.have_full_utxo_index()`), the server goes
   on answering, and every wallet command waits for an index that never moves again. On the live
   chain it happened to node 1 after 1,500 blocks at four seconds apart and stalled its agent,
   its model and the 2B's live check. `tests/check_ord_orphan.py` does it to a real ord on
   purpose: ord's process is stopped while it reads one branch and let go on the other. Two
   repairs. ord now drops a batch whose blocks do not link and reads the chain again
   (`patches/ord-0.29.1-qwencoin.patch`). And `tools/net.py` notices an index whose thread died,
   or that holds another block than the node under its tip, writes down what was wrong with it
   (the first input it could not serve and the orphan behind it), sets it aside and builds it
   again; that also covers an index made by the ord from before the patch.
19. **The commands of the cards on a fresh copy.** The model card said `pipeline.sh up 4` and
   `chat`; a fresh copy of the repository has no Python environment, no link to the work table
   and no model, so both stopped with an error, and the project page's command ran the whole
   build pipeline. `pipeline.sh join` is now what the cards give, behind one line that needs
   only curl (`install.sh`); `up` sets up what it needs, and the other commands say what to run
   first. Found by running the cards' commands on a machine that had nothing.
20. **`pipeline.sh ask "question"` never reached the model.** The script put the question after
   `--node`, where the runtime's argument parser no longer takes positionals, so the one command
   every run ends by recommending stopped with a usage error. The parser now accepts the order
   its own usage line shows, and the script no longer depends on it. Found by asking the model
   of a freshly installed node how many coins it has.
21. **One refused connection ended an installation.** The first run of the cards' install line
   from the live repository stopped at the model download: one connection of the many it opens
   was refused. Run again, it went on where it had stopped; but nobody should have to. The model
   download and the installer's own download now ask again (what has arrived is kept), up to
   seven and five times, and say so.

## 15. Not done

- **The bridge reads hex, up to 32 bytes.** A payload that is not hex digits (a `0x` prefix
  included), has an odd number of digits or is longer does not fire in the stream; the runtime
  then answers with the node's executor of the same circuit.
- **Only the hash is computed in the token stream.** The other functions a checkpoint declares are
  carried out by the node it runs on.
- **Choosing the function is the language model's skill** (section 11 has what the 0.8B did).
- **Text right after a trace.** Qwen's recurrent state passes a trace bit for bit and its
  attention ignores the chip positions, but its 4-tap convolution window does not: the first three
  text tokens after a trace see chip tokens there (section 9 has the numbers). In a bridged call
  those three tokens are the circuit's own `\n<tool_response>\n`.
- **The larger merges.** 0.8B is verified on this machine in full; 2B to 27B are merged and
  checked by the pipeline where there is memory for them, and published only if they pass there.
- **GPU kernels.** Exactness is verified on transformers' reference CPU kernels only.
- **The rest of ord's SHA-256.** ord's chain hashing and signing run in the weights (section 8).
  Its TLS stack (ring, aws-lc) and the `sha2` crate (BIP322 message signing, hashes of embedded
  web assets) keep native SHA-256; none of it touches chain data.
- **The work function ties validity to the weights, not to a program.** Any kernel computing the
  same integer sums mines; a GPU is faster than a CPU. It is a pass over the 0.8B's MLP matrices
  with sign activations, not the model's forward function.
- **A small network.** Since 2026-11-13 the chain runs as four nodes on one machine at home and
  one public node on a rented pod, whose address is in the file `SEEDS` of the project's
  repository (`tools/runpod.py`; a copy of the repository dials it). The home nodes reach it
  through their router and it fetched their chain. A fresh installation joining from a third
  network has been exercised on the build machine against a public node started there the same
  way (dialed, reorganised onto the longer chain, index rebuilt, its blocks paid to the address
  it was given), and not yet against the public one from outside. No DNS seeds, no explorer.
- **Sections 2 to 5 were produced on this project's first chain**, whose work was SHA-256d (the
  block hashes shown there have leading zeros for that reason). The script rules, ord and the
  graph code they exercise are the same on the current chain; the current chain's own runs are
  sections 10 and 11.
- **Fee bumping for slashes.** The covenant fee is fixed at 1,000 sat per transaction.
