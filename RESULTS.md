# Results

One machine: 2 cores, 8 GB, no GPU. Ubuntu 24.04, gcc 13.3, rustc 1.97, torch 2.14.1 (CPU), transformers 5.22.2. `logs/` holds the raw output; this file is `tools/results.py` over it.

| artifact | sha256 |
|---|---|
| `dist/qwensha/model.safetensors` (the checkpoint, 656 MB; `python -m hashformer.hf` rebuilds it bit for bit) | `ec5f7cb21e9c95bfc82576cb52a8a0f01cf3b09b7ccaa22cf6f8e6aa86b2320c` |
| `exec/qsha_circuit.h` (the same nonzero weights, as the executor reads them) | `1b5fd9a90f9779800e945ec25a9b2eeda6af493ccf10a036c6e6beb70fcfdcf5` |
| `patches/bitcoin-v29.4-inq-qwencoin.patch` (applies to tag `v29.4-inq`) | `4bb55dd43b7b8e3a6750a116f7bbe7a645fff6f66c731844760cb2dc20c86341` |
| `patches/ord-0.29.1-qwencoin.patch` (applies to tag `0.29.1`) | `92721b77c4c62df9036fabde8f5873204e5f3e52fd60a235d4fb71daedaabc76` |
| `patches/bitcoin_hashes-0.14.101-qsha.patch` (rust-bitcoin's SHA-256, routed to the model) | `b8ca552a0537530622c64722f45fc09853f7595bd06619ed81235efaf77d7907` |
| `patches/secp256k1-sys-0.10.1-qsha.patch` (ord's libsecp256k1, routed to the model) | `a5742c22c3f478d1e7be883f989e94a86f1e11a4761b93e97e639c09efdf5038` |
| `dist/qwork.bin` (the chain's work table: the MLP weights of Qwen3.5-0.8B at 16-bit fixed point) | `8f53884f7727e80b86f3ddf288d3c3f0f998f1c63b8cd4f0bdad182ddd3f2c03` |
| `exec/qwork.c` (the work function's kernel, as the node links it) | `200bd4b0cd801fdbe2d22432935dd0d991059b4aa5b0dffa3b449036cba9f7b0` |
| `dist/qwen08-sha/model.safetensors` (Qwen3.5-0.8B with the circuit merged in, as sections 9 and 11 ran it) | `fc220b70769abae627aef5e35206450800e7ebea40535f2e13a8520139a1dcc6` |
| `bin/qwencoind` (the node of sections 10 and 11 (sections 2 to 7 ran the node before the work function changed)) | `be7da8b09aee1a37fc8104e0b5c3e25285239d6682b84183fc8638c318cdffaf` |
| `bin/ord` (the ord of section 11) | `2d96313214f6857513668a8c8042f8dcdedc6a153e94ff629ba58564981bfd3f` |

## 1. The model

`logs/model-checks.log`: the checks in `tests/` on the compiler, the stock checkpoint and the executor.

```
$ python3 tests/check_ir_full.py
{'layers': 5, 'dims': 803, 'key_heads': 15, 'value_heads': 20, 'units': 1473, 'units_per_mlp': {1: 431, 2: 515, 3: 98, 4: 168, 5: 261}, 'heads_per_attn': {1: 4, 2: 10, 3: 1}}
fresh messages: 130667 positions, 0 wrong, 115s
total: 164714 positions, 0 wrong, 145s
  (exit 0, 146 s)
$ python3 tests/check_hf.py residual
model built 22s: hidden 832, m 13, 328M params, nnz 31923, rss 3.3GB
len 3 prefill: residual == IR on all 802 dims at 458 positions (5s)
len 3 decode: residual == IR on all 802 dims at 458 positions (274s)
len 3 mixed: residual == IR on all 802 dims at 458 positions (146s)
len 70 prefill: residual == IR on all 802 dims at 899 positions (11s)
len 70 decode: residual == IR on all 802 dims at 899 positions (478s)
len 70 mixed: residual == IR on all 802 dims at 899 positions (235s)
  (exit 0, 1174 s)
$ python3 tests/check_hf.py generate
model built 25s
len 3: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad OK 458 tokens, 453 generated, 236s
len 0: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 OK 458 tokens, 456 generated, 247s
len 41: d9ad3688f128fc7bf8811a35919c3c8680343a751fd734c8c64b547fbd43b08d OK 458 tokens, 415 generated, 262s
len 64: fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108 OK 899 tokens, 833 generated, 472s
len 121: 335a461692b30bba1d647cc71604e88e676c90e4c22455d0b8c83f4bd7c8ac9b OK 1340 tokens, 1217 generated, 532s
  (exit 0, 1775 s)
$ python3 tests/check_hf_stock.py
Qwen3_5ForCausalLM 328M params
b'abc' ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad True 35s
b'hello qwencoin' 5caaea1de10a1764b6c6d3b2a303c9aad319f0985d4e224d6cf3b87b3001914e True 218s
  (exit 0, 260 s)
$ python3 tests/check_exec.py
executor residual == IR on all 803 dims at 3618 positions
free-running executor == reference trace on 107660 tokens
midstate compress / finish == hashlib
20000 random messages == hashlib; 19852890 tokens in 304.1s = 65290 tok/s, 15.3 us/token
bitcoin genesis: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f 24.6 ms per header double-hash
  (exit 0, 315 s)
```

## 2. Block hashes of the first chain, recomputed by the checkpoint

(Sections 2 to 5 ran on this project's first chain, whose proof of work was the SHA-256d of the header; the current chain's work function is section 10.) `tests/check_hf_chain.py` loads `dist/qwensha` with stock transformers and double-hashes block headers taken from node 1 with plain `generate()`. The node computed the same hashes in its executor.

```
height 0: generate() -> 003264da80f1f74497154f9c4a35401265c441d0b330eb440b64ad5da8dc77b3  node+hashlib equal: True  meets target 2000ffff: True  [141s]
height 126: generate() -> 0028cb57df7bfd784561bfec11c2a9d906b765f5788e5eb3450dca79afab540d  node+hashlib equal: True  meets target 2000ffff: True  [145s]
height 136: generate() -> 0060ec4f2716dac1b5b43ec109e66d14a98c56aab305b65b0d4b0fe67c5bccf5  node+hashlib equal: True  meets target 2000ffff: True  [149s]
OK: 3 headers, 3732 generated tokens
exit 0
```

## 3. Three nodes, from genesis

`tools/launch.py`: three `qwencoind` + `ord` pairs on localhost mine 118 blocks, then seven identities run the graph. Every block, transaction and script check on every node hashed in the weights, and so did ord (its index's txids, the inscriptions' commit and reveal transactions, their signatures) and the agents.

```
genesis 003264da80f1f74497154f9c4a35401265c441d0b330eb440b64ad5da8dc77b3 sha256: qwen3.5-weights(hashformer;1way)
height 28 (144s)
height 38 (198s)
height 48 (252s)
height 58 (317s)
height 68 (348s)
height 78 (403s)
height 88 (450s)
height 98 (547s)
height 108 (580s)
height 118 (610s)
[     0s h= 118] three nodes: {1: '/Qwencoin:29.4.0(inquisition)/', 2: '/Qwencoin:29.4.0(inquisition)/', 3: '/Qwencoin:29.4.0(inquisition)/'}
[    30s h= 119] 1. bonds: each identity is an inscription sitting on its own covenant output
[    35s h= 119]    alice    id 2c3413a2a1cfff32  bond bc1p04jtxugdl7ztflx4h53w503lvxev49eczphdnh22w7kcg0xzq62swn9ct7
[    40s h= 119]    bob      id 784c5092f4f7c375  bond bc1p7jpajpzegguz72mcn9ncqvf8728ak6h7cuftg6uk5xsvxv97mams5ht063
[    44s h= 119]    mallory  id e52450cc9908f98f  bond bc1p27nmhsq9q2lzsz8vj588e8c70vygm6lmy9tsnj4wtvmmxfnyw9rsrduv75
[    48s h= 119]    eve      id 45a72e7a7ccd75e8  bond bc1p2jc2yy3h6wc89e6p52e4n2q4wtndk9vyd56c075asguyg3whx0fq724w6l
    alice      weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice: 5,000,000
    bob        weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice:         0
    eve        weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice:         0
    mallory    weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice:         0
[    66s h= 120] 2. work: each node hashes the tip header in its weights and signs the result; posts
[    92s h= 121] 3. audit, then vouch: stake placed on a peer burns if that peer is ever caught
[    94s h= 121]    alice recomputed 3 attested hashes in its weights: 0 faults
[   100s h= 121]    bob recomputed 3 attested hashes in its weights: 0 faults
    alice      weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 6,000,000
    bob        weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 1,000,000
    eve        weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 1,000,000
    mallory    weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 1,000,000
    vouch      alice -> bob        1,000,000 sat
    vouch      alice -> mallory    1,000,000 sat
    vouch        bob -> alice      1,000,000 sat
    vouch        bob -> eve        1,000,000 sat
[   134s h= 122] 4. heads: each identity signs one commitment to its history for this epoch
[   157s h= 123] 5. faults: mallory signs a false hash; eve signs a second, different head for the same epoch
[   161s h= 123]    (an honest agent refuses: eve already signed a head for epoch 10; refusing to equivocate)
[   172s h= 124]    mallory tries to run: starts unbonding her own bond
[   196s h= 125]    her bond is now unbonding: still slashable for 6 blocks
[   196s h= 125] 6. slashing: bob audits; the proofs alone burn every stake on the offenders
[   205s h= 125]    equiv proof against eve      burns the id    stake of 5,000,000 sat (tx 67fbecc27304683a)
[   205s h= 125]    equiv proof against eve      burns the vouch stake of 1,000,000 sat (tx 9c8fe47230cd3beb)
[   205s h= 125]    lie   proof against mallory  burns the id    stake of 5,000,000 sat (tx 4ce9c55e988a5ab9)
[   205s h= 125]    lie   proof against mallory  burns the vouch stake of 1,000,000 sat (tx 858109b057e7cfe6)
[   212s h= 125]    alice audits too and builds the same transactions: 0 burns that bob had not already broadcast
    alice      weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 6,000,000
    bob        weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 1,000,000
    eve        weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    mallory    weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    vouch      alice -> bob        1,000,000 sat
    vouch        bob -> alice      1,000,000 sat
[   222s h= 126]    ord: inscription 569a7c9268e4.. (id mallory) charms ['coin', 'uncommon', 'burned']
[   222s h= 126]    ord: inscription a059555aabc7.. (id eve) charms ['coin', 'uncommon', 'burned']
[   222s h= 126]    ord: inscription ce37a03cb71b.. (vouch bob -> eve) charms ['burned']
[   223s h= 126]    ord: inscription ba85373d028c.. (vouch alice -> mallory) charms ['burned']
[   223s h= 126]    sats destroyed in OP_RETURN outputs: 11,995,000
[   223s h= 126] 7. leaving honestly: alice unbonds her vouch for bob, waits out the delay, withdraws
[   240s h= 127]    withdraw before the delay: rejected (non-BIP68-final)
[   288s h= 134]    withdraw after 6 blocks: tx 157617c387283c8d confirmed
    alice      weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 6,000,000
    bob        weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice:         0
    eve        weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    mallory    weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    vouch        bob -> alice      1,000,000 sat
[   290s h= 134] 8. sybils: mallory's operator spends 3,000,000 sat on three fresh names vouching in a ring
[   351s h= 136]    weight of the three names together: 3,000,000 sat = the 3,000,000 locked; one name with one bond would weigh the same
[   351s h= 136]    trust from alice into the ring: [0, 0, 0] (no honest vouch crosses into it)
    alice      weight 6,000,000 sat (own bond 5,000,000, vouched 1,000,000)  bond active  trust from alice: 6,000,000
    bob        weight 5,000,000 sat (own bond 5,000,000, vouched         0)  bond active  trust from alice:         0
    eve        weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    mallory    weight         0 sat (own bond         0, vouched         0)  bond slashed  trust from alice:         0  CONVICTED
    sybil0     weight 1,000,000 sat (own bond   500,000, vouched   500,000)  bond active  trust from alice:         0
    sybil1     weight 1,000,000 sat (own bond   500,000, vouched   500,000)  bond active  trust from alice:         0
    sybil2     weight 1,000,000 sat (own bond   500,000, vouched   500,000)  bond active  trust from alice:         0
    vouch        bob -> alice      1,000,000 sat
    vouch     sybil0 -> sybil1       500,000 sat
    vouch     sybil1 -> sybil2       500,000 sat
    vouch     sybil2 -> sybil0       500,000 sat
[   356s h= 136] done: all three nodes agree on tip 0060ec4f2716dac1b5b4.. at height 136; the agents alone ran 4,849,754 model tokens of SHA-256
```

## 4. The slashing covenants, decided by consensus

`tests/check_covenant.py` on the same chain (`testmempoolaccept`, then the confirmed burns):

```
funded 4 bonds ['fd5fb5eb1d7b', '73f5cd2e1861', '30dc678582d1', '826caaea10d7']
slash with a TRUE hash claim     -> (False, 'mempool-script-verify-flag-failed (Script failed an OP_VERIFY operation)')
false claim signed by another key -> (False, 'mempool-script-verify-flag-failed (Invalid Schnorr signature)')
valid proof, coins sent to slasher -> (False, 'mempool-script-verify-flag-failed (Script failed an OP_CHECKTEMPLATEVERIFY operation)')
valid proof, coins burned          -> (True, '')
two signatures over ONE root      -> (False, 'mempool-script-verify-flag-failed (Script failed an OP_VERIFY operation)')
roots signed for different epochs -> (False, 'mempool-script-verify-flag-failed (Invalid Schnorr signature)')
two roots, one epoch              -> (True, '')
unbond signed by a non-owner      -> (False, 'mempool-script-verify-flag-failed (Invalid Schnorr signature)')
unbond by the owner               -> (True, '')
withdraw before the delay         -> (False, 'non-BIP68-final')
slash while unbonding             -> (True, '')
withdraw after the delay          -> (True, '')
lie burn        fdf2095bd402c815 confirmations 8 -> 0.04999 to nulldata OP_RETURN 7177673a736c6173686564
equiv burn      f4bd79176ea4ccec confirmations 8 -> 0.04999001 to nulldata OP_RETURN 7177673a736c6173686564
unbonding burn  ea0c31516647a172 confirmations 7 -> 0.04998003 to nulldata OP_RETURN 7177673a736c6173686564
withdraw        8c6b3a32f1e9d45a confirmations 1 -> 0.04998002 to witness_v1_taproot 1 818526d8f72442fcf8899772fb475ce3ecf2b4
```

## 5. Attacks on the accounting

`tests/check_sybil.py` on the same chain; the three nodes' indexes must agree:

```
baseline: alice weighs 900,000 (bond 600,000 + bob's vouch 300,000), trust bob -> alice 300,000
on chain: two bond outputs with two ids each, a replayed vouch, two false ids, two rigged bonds; a count per bond-shaped inscription would credit all of them
1a. two ids revealed into one bond output:  neither is a stake (1,400,000 claimed, 700,000 locked)
1b. two ids moved into one bond output:    neither is a stake; mallory weighs 0
2.  bob's vouch replayed on a 2nd output:   'replay'; alice still weighs 900,000, trust bob -> alice still 300,000
3.  id for alice's key signed by mallory:   not a message (the signature fails under alice's key)
4.  id for alice's key with no signature:   not a message; alice is still named 'alice-818030'
5.  bond with a 300,000 sat covenant fee:   not a message (a slash must burn, not pay a miner)
6.  bond with a one-block delay:            not a message (a stake must outlast its proof)
7.  an honest vouch of 250,000:             alice weighs 1,150,000, mallory 500,000, on all three nodes
OK: weight never exceeded locked sats
```

## 6. Bitcoin's unit tests on in-weights SHA-256

`tools/unit.sh`, `test_bitcoin` built from the patched tree. 57 of 59 filters pass (123 of the binary's 672 cases). Selected: the suites that test hashing, keys, scripts, sighashes, merkle trees and transactions directly; section 9 says what was left out and why.

| filter | cases | result | time |
|---|---|---|---|
| `crypto_tests/hmac_sha256_testvectors` | 1 | ok | 32 s |
| `crypto_tests/hkdf_hmac_sha256_l32_tests` | 1 | ok | 0 s |
| `crypto_tests/sha256d64` | 1 | ok | 15 s |
| `crypto_tests/muhash_tests` | 1 | ok | 4 s |
| `hash_tests` | 2 | ok | 1 s |
| `pow_tests` | 16 | ok | 7 s |
| `base58_tests` | 2 | ok | 1 s |
| `key_io_tests` | 3 | ok | 21 s |
| `bip32_tests` | 6 | ok | 6 s |
| `key_tests` | 7 | ok | 89 s |
| `script_standard_tests` | 7 | ok | 4 s |
| `script_p2sh_tests` | 6 | ok | 10 s |
| `script_segwit_tests` | 12 | ok | 6 s |
| `multisig_tests` | 3 | ok | 12 s |
| `merkleblock_tests` | 2 | ok | 1 s |
| `bloom_tests` | 12 | ok | 7 s |
| `merkle_tests/merkle_test_empty_block` | 1 | ok | 1 s |
| `merkle_tests/merkle_test_oneTx_block` | 1 | ok | 1 s |
| `merkle_tests/merkle_test_OddTxWithRepeatedLastTx_block` | 1 | ok | 1 s |
| `merkle_tests/merkle_test_LeftSubtreeRightSubtree` | 1 | ok | 1 s |
| `merkle_tests/merkle_test_BlockWitness` | 1 | ok | 1 s |
| `sighash_tests/sighash_from_data` | 1 | ok | 19 s |
| `sighash_tests/sighash_caching` | 1 | ok | 8 s |
| `script_tests/cat_simple` | 1 | ok | 1 s |
| `script_tests/cat_empty_stack` | 1 | ok | 2 s |
| `script_tests/cat_dup_test` | 1 | ok | 130 s |
| `script_tests/templatehash` | 1 | ok | 2 s |
| `script_tests/bip341_keypath_test_vectors` | 1 | ok | 2 s |
| `script_tests/compute_tapbranch` | 1 | ok | 1 s |
| `script_tests/compute_tapleaf` | 1 | ok | 1 s |
| `script_tests/script_PushData` | 1 | ok | 1 s |
| `script_tests/script_cltv_truncated` | 1 | ok | 1 s |
| `script_tests/script_CHECKMULTISIG12` | 1 | ok | 2 s |
| `script_tests/script_CHECKMULTISIG23` | 1 | ok | 6 s |
| `script_tests/script_combineSigs` | 1 | ok | 3 s |
| `script_tests/sign_invalid_miniscript` | 1 | ok | 1 s |
| `script_tests/sign_paytoanchor` | 1 | ok | 1 s |
| `script_tests/script_standard_push` | 1 | ok | 0 s |
| `script_tests/script_IsPushOnly_on_invalid_scripts` | 1 | ok | 1 s |
| `script_tests/script_GetScriptAsm` | 1 | ok | 1 s |
| `script_tests/script_byte_array_u8_vector_equivalence` | 1 | ok | 1 s |
| `script_tests/script_FindAndDelete` | 1 | ok | 0 s |
| `script_tests/script_HasValidOps` | 1 | ok | 1 s |
| `script_tests/formatscriptflags` | 1 | ok | 1 s |
| `bip324_tests` | 1 | FAILED: the vectors are bound to Bitcoin's network magic (section 9) | 17 s |
| `script_tests/script_json_test` | 1 | ok | 757 s |
| `script_tests/script_build` | 1 | ok | 753 s |
| `transaction_tests/tx_valid` | 1 | ok | 229 s |
| `transaction_tests/tx_invalid` | 1 | ok | 53 s |
| `transaction_tests/tx_no_inputs` | 1 | ok | 1 s |
| `transaction_tests/tx_oversized` | 1 | ok | 288 s |
| `transaction_tests/basic_transaction_tests` | 1 | ok | 1 s |
| `transaction_tests/test_Get` | 1 | ok | 1 s |
| `transaction_tests/test_witness` | 1 | ok | 17 s |
| `transaction_tests/test_IsStandard` | 1 | ok | 52 s |
| `transaction_tests/max_standard_legacy_sigops` | 1 | ok | 16 s |
| `transaction_tests/spends_witness_prog` | 1 | ok | 3 s |
| `transaction_tests/test_big_witness_transaction` | 1 | STOPPED after 35 min (4,500 inputs signed and checked without the sighash cache, up to 225 KB rehashed per input: about 16 h in weights) |  |
| `crypto_tests/sha256_testvectors` | 1 | ok | 4727 s |

## 7. Bitcoin's functional tests against the node

`tools/functional.sh` on regtest: the node hashes in the weights, the Python framework with hashlib and its own secp256k1, so every txid, sighash and template hash is cross-checked. 15 of 17 pass.

| test | | result | time |
|---|---|---|---|
| `feature_opcat.py` | | ok | 175 s |
| `feature_checktemplateverify.py` | | ok | 123 s |
| `feature_templatehash.py` | | ok | 317 s |
| `p2p_v2_transport.py` | | ok | 79 s |
| `p2p_v2_encrypted.py` | | ok | 307 s |
| `mempool_accept.py` | | ok | 645 s |
| `rpc_signrawtransactionwithkey.py` | | ok | 17 s |
| `feature_cltv.py` | | ok | 93 s |
| `feature_dersig.py` | | ok | 87 s |
| `feature_nulldummy.py` | | ok | 61 s |
| `feature_taproot.py` | | STOPPED after 40 min (it funds thousands of spends; one 50-input, 641-output funding transaction takes 16 min to sign and validate in weights: hours in total) |  |
| `mining_basic.py` | | TIMEOUT | 1500 s |
| `rpc_txoutproof.py` | | ok | 66 s |
| `feature_utxo_set_hash.py` | | ok | 71 s |
| `rpc_signmessagewithprivkey.py` | | ok | 3 s |
| `rpc_blockchain.py` | | ok | 537 s |
| `feature_bip68_sequence.py` | | ok | 413 s |

## 8. What is in the binaries

`tests/check_binary.py bin/qwencoind` and `tests/check_binary.py --ord bin/ord`. The node holds no SHA-256 but the circuit. ord's chain hashing and signing hold none either; its TLS stack and the `sha2` crate keep theirs.

```
bin/qwencoind
  SHA-256 round constants outside SHA-512's constants: 0 
  SHA-256 initial value:
    1 x immediate in .text: (anonymous namespace)::SelfTest()
    1 x SHA-512 initial value (64-bit)
    1 x 8-word table in .rodata: (anonymous namespace)::SelfTest()::result
    1 x 8-word table in .rodata: (anonymous namespace)::SelfTest()::init
  hard-coded tagged-hash midstates: 0 
OK

bin/ord
  SHA-256 round constants, by the symbol that holds them:
    256 x K256 (ring / aws-lc: TLS)
     68 x sha2::sha256::compress256
     64 x unnamed constant table (the sha2 crate's, one per version)
  in rust-bitcoin's hashes or libsecp256k1: 0 
  hard-coded tagged-hash midstates: 0 
  executor linked: True
OK: ord's chain hashing and signing have no native SHA-256
```

`rust/qsha-check` runs the two patched crates directly: digests against the independent `sha2` crate, a tagged hash, HMAC, a block header, the BIP340 signing vector (its three tagged hashes run inside libsecp256k1).

```
$ (cd upstream/rust/qsha-check && cargo run --release)
OK: 1506 digests equal the sha2 crate's, tagged hash, HMAC, genesis header, BIP340 vector 0, ECDSA; 11445294 model tokens
```

## 9. The chip inside Qwen3.5: the merge and the bridge

`tests/check_bridge.py`: the merged circuit (the hasher plus what reads Qwen's tool call, replays the payload and prints the answer) in exact arithmetic against its reference machine, `hashformer/bridge.py`. The text tokens are Qwen3.5's real ids; a stream runs from the chat prompt to the text after the answer.

```
circuit with the bridge: 5 layers, 1180 dims, 29 key heads, 2319 units
pure chip streams (the hasher as before):
  3 bytes: 458 tokens, 0 wrong
  70 bytes: 899 tokens, 0 wrong
bridged streams:
  abc, in the chat: 900 tokens, the circuit owns 531, 0 wrong (1s); fired
  empty payload: 893 tokens, the circuit owns 531, 0 wrong (1s); fired
  one byte: 560 tokens, the circuit owns 531, 0 wrong (0s); fired
  32 bytes, letters merged into long tokens: 951 tokens, the circuit owns 531, 0 wrong (1s); fired
  ffffffff00000000aabbccdd: 571 tokens, the circuit owns 531, 0 wrong (1s); fired
  deadbeef: 561 tokens, the circuit owns 531, 0 wrong (0s); fired
  with 0x (not hex digits: left to the language model): 50 tokens, the circuit owns 0, 0 wrong (0s); no fire
  spaces between bytes: 565 tokens, the circuit owns 531, 0 wrong (0s); fired
  odd number of digits: 368 tokens, the circuit owns 0, 0 wrong (0s); no fire
  33 bytes: 86 tokens, the circuit owns 0, 0 wrong (0s); no fire
  not hex: 46 tokens, the circuit owns 0, 0 wrong (0s); no fire
  another function: 49 tokens, the circuit owns 0, 0 wrong (0s); no fire
  another parameter: 50 tokens, the circuit owns 0, 0 wrong (0s); no fire
  a second call after the first answer: 1137 tokens, 0 wrong
  a host-driven hash after a bridged one, then text: 1044 tokens, 0 wrong
OK
```

`hashformer.merge` on Qwen3.5-0.8B, then `tests/check_merge.py` on the result: stock transformers, bf16, this machine's CPU. The free-running part is one greedy `generate()` from the user's question.

```
$ python3 -m hashformer.merge base/Qwen3.5-0.8B --out dist/qwen08-sha
base: hidden 1024, 24 layers, vocab 248320
circuit: 1180 dims, 29 key heads, 2319 units; reads 747 hex tokens of the base's vocabulary
merged: hidden 2240 (ballast 35, norm scale 2^-37), 5 + 24 layers, vocab 379968
anti-sink: 6 attention layers x 2 key heads fitted on 978 text tokens (cpu, torch.bfloat16); mean query / query rms 0.34 .. 0.54
  1/390 tensors, 1.6 GB
  41/390 tensors, 1.8 GB
  81/390 tensors, 2.1 GB
  121/390 tensors, 2.3 GB
  161/390 tensors, 2.6 GB
  201/390 tensors, 2.8 GB
  241/390 tensors, 3.1 GB
  281/390 tensors, 3.3 GB
  321/390 tensors, 3.6 GB
  361/390 tensors, 3.9 GB
dist/qwen08-sha: 4.05 GB, 2.18B parameters

$ python3 tests/check_merge.py base base/Qwen3.5-0.8B work
base, after the same call and response as text: 'The SHA-256 hash of the bytes 616263 is:\n\n**ba7816bfa701cfea414140d5dade223b00361a396177a9cbb410ff61f20015ad**<|im_end|>\n<|endoftext|>'
base (cpu, torch.bfloat16): 257 text positions; greedy answer: 'aabbcc<|im_end|>\n<|endoftext|>'
floor (the base as above against itself in fp32 on the cpu): KL mean 4.74e-04, max 4.68e-03, top-1 agreement 0.9572

$ python3 tests/check_merge.py merged dist/qwen08-sha work
merged (cpu, torch.bfloat16): 2.18B parameters, hidden 2240, 5 chip layers + Qwen's, chip tokens from id 248320
chip, teacher-forced over the whole vocabulary:
  pure trace, 'abc': 457 positions, the circuit owns 457, 0 wrong; winner 72..72, runner-up at most 0.889 of it
  pure trace, 70 bytes (2 blocks): 898 positions, the circuit owns 898, 0 wrong; winner 72..72, runner-up at most 0.889 of it
  after a chat prefix of 30 text tokens: 457 positions, the circuit owns 457, 0 wrong; winner 72..72, runner-up at most 0.889 of it
  after 88 tokens of prose, 121 bytes (3 blocks): 1339 positions, the circuit owns 1339, 0 wrong; winner 72..72, runner-up at most 0.889 of it
chip, free-running, the host answering IN (token by token with the cache):
  inside the chat: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad == hashlib, every token == reference (458 tokens, 332 ms per token)
bridge, teacher-forced over the whole vocabulary (prompt, call, hash, answer, text after it):
  sha256(616263): 917 positions, the circuit owns 531, 0 wrong; winner 72..6.16e+13, runner-up at most 0.892 of it; where Qwen speaks the best chip logit is -6.68e+09
  sha256(6a09e667f3bcc908..): 650 positions, the circuit owns 531, 0 wrong; winner 72..5.83e+13, runner-up at most 0.891 of it; where Qwen speaks the best chip logit is -4.56e+09
  no call for an odd number of digits: 67 positions, the circuit owns 0, 0 wrong; where Qwen speaks the best chip logit is -5.34e+09
  no call for not hex: 63 positions, the circuit owns 0, 0 wrong; where Qwen speaks the best chip logit is -7.08e+09
bridge, free-running: generate() from the user's question (greedy, stock stopping rules):
  642 new tokens in 215s (334 ms per token): '<tool_call>\n<function=sha256>\n<parameter=hex>\n616263\n</parameter>\n</function>\n</tool_call>[457 chip tokens]\n<tool_response>\nba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n</tool_response>\n<think>\n\n</think>\n\nThe SHA-256 hash of the bytes 616263 is:\n\n**ba7816bfa701cfea414140d5dade223b00361a396177a9cbb410ff61f20015ad**<|im_end|>'
  the circuit owned 531 of them, 0 differ from the reference; 457 chip tokens
  payload 616263: printed digest == hashlib (ba7816bf8f01cfea..)
  the language model's answer repeats the digest with errors (its own copying): 'The SHA-256 hash of the bytes 616263 is:\n\n**ba7816bfa701cfea414140d5dade223b00361a396177a9cbb410ff61f20015ad**<|im_end|>'
  that answer against the base's answer to the same call and response as plain text (no trace, no chip): 78 of 78 tokens equal
owner and text fidelity:
  257 text positions: best chip logit anywhere -6.64e+09, lowest winning text logit 11.6
  KL(base || merged) mean 8.10e-04, max 4.03e-03; top-1 agreement 0.9883
  floor, the base against itself in fp32: KL mean 4.74e-04, max 4.68e-03; top-1 agreement 0.9572
  greedy answer to a chat prompt without the tool: 'aabbcc<|im_end|>'; 4 of 4 tokens equal the base's
a trace in the context (458 chip tokens between the chat and more text):
  Qwen's recurrent state after the trace: bit-identical in 18 of its 18 DeltaNet layers (the chip's own 5 layers: changed in 5)
  the next 35 text tokens, KL(without the trace || with it): mean 1.06e-01, max 1.13e+00; top-1 equal at 29
    of which the positions alone (the same text moved 458 places, no trace): mean 6.77e-03
    and the chip tokens at equal positions: first 3 tokens 4.52e-01 (Qwen's 4-tap conv window holds chip tokens), the rest 4.74e-02
OK
```

## 10. Proof of work bound to the weights

`tests/check_work.py`: the C kernel the node links against the exact numpy reference, and the statistics of the function over the real table (`dist/qwork.bin`, built from Qwen/Qwen3.5-0.8B).

```
table: 24 layers, 3584 x 1024, 528482328 bytes, fnv64 0x33a548a33925d201, sha256 8f53884f7727e80b86f3ddf288d3c3f0f998f1c63b8cd4f0bdad182ddd3f2c03
kernel: C against the numpy reference on 42 inputs (edge seeds included): 0 differ
avalanche: one nonce bit flips 127.3 of the 256 output bits (min 110), one seed bit 129.2 (min 113)
           inside the pass, nonce and nonce^1: state bits that differ after layers 1, 2, 3, 12, 24: [526, 502, 512, 528, 494] of 1024
speed: 43.4 ms per attempt on one thread (23 attempts/s)
collapse: 3000 nonces under one seed give 3000 distinct results
raw output bits (before whitening): balance per bit 0.471 .. 0.529 (binomial 3-sigma: 0.473 .. 0.527)
whitening: bit balance of the proof-of-work value 0.472 .. 0.523; leading byte chi-square 267 on 255 degrees of freedom (99% range 205 .. 310); under the minimum target 9 of 3000 (expected 11.7)
OK
```

`pow_tests/qwork_*` in the node's own unit-test binary (`QWORK_TABLE=dist/qwork.bin`); the node's SHA-256 there is the compiled circuit, the expected values come from the reference with hashlib:

```
Running 5 test cases...
./test/pow_tests.cpp(305): Entering test case "qwork_table_is_checked"
./test/pow_tests.cpp(305): Leaving test case "qwork_table_is_checked"; testing time: 1563747us
./test/pow_tests.cpp(346): Entering test case "qwork_vectors"
./test/pow_tests.cpp(346): Leaving test case "qwork_vectors"; testing time: 1171767us
./test/pow_tests.cpp(373): Entering test case "qwork_genesis"
./test/pow_tests.cpp(373): Leaving test case "qwork_genesis"; testing time: 1326680us
./test/pow_tests.cpp(403): Entering test case "qwork_native_hashing_does_not_count"
mined a header under the chain's rule at nonce 1581, hash 4287da7c02c419a4a852583bb6c3a906cdf69530dd8ad1661f9587da3e97ce1b
./test/pow_tests.cpp(403): Leaving test case "qwork_native_hashing_does_not_count"; testing time: 92402889us
./test/pow_tests.cpp(431): Entering test case "qwork_other_chains_unchanged"
./test/pow_tests.cpp(431): Leaving test case "qwork_other_chains_unchanged"; testing time: 484444us
*** No errors detected
```

`tests/check_qwork_node.py`: a private network of the chain, four `qwencoind`.

```
table /home/claude/qwencoin/dist/qwork.bin: fnv64 0x33a548a33925d201
1 a node without the right weights does not start
    no file: Error: The work table /home/claude/qwencoin/work/qwork-net/absent.bin cannot be opened. On this chain a proof of work is a pass through the weights of
    not a table: Error: /home/claude/qwencoin/work/qwork-net/junk.bin is not a work table (QWRK1, 24 layers of 3584 x 1024). On this chain a proof of work is a pass th
    one weight changed (of 264,241,152): Error: The work table /home/claude/qwencoin/work/qwork-net/forged.bin has checksum 7315469c47bcd201, this chain's is 33a548a33925d201. On this chain a proof of work is a pass through the wei
2 three nodes up (8s); genesis
    hash 51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5
    work 0083bfd7a2ae5796c139f180b33630684eb71751e4b2b3153ffe288da7739d1d  (node == numpy reference == C kernel == the constant; under the target 00ffff000000..; the hash itself is not)
3 mining 8 blocks on two nodes
    n1 mined height 1 bca94eb125a87733.. at nonce 71
    n2 mined height 2 20f6efa285da3aa8.. at nonce 46
    n1 mined height 3 ef3e3e90ed984a02.. at nonce 200
    n2 mined height 4 90e3122dbd23857f.. at nonce 367
    n1 mined height 5 e7bcec5263e8f606.. at nonce 255
    n2 mined height 6 de21221cf755f635.. at nonce 344
    n1 mined height 7 edffde7409defb9f.. at nonce 86
    n2 mined height 8 9ab76aed067fc80c.. at nonce 9
    1386 passes in 74s: 53 ms per attempt in the node (the pass, one SHA-256 in the weights, the template); all three nodes at height 8, tip 9ab76aed067fc80c..
    every header 0..8: work value by the reference == all three nodes', and under the target; hash by hashlib == the node's. (The hashes themselves are ordinary numbers: the largest has 256 bits.)
4 blocks ground the Bitcoin way
    hash 00d0be03437f7bbeccd9.. is under the target; submitblock: 'high-hash'; submitheader: 'high-hash'
    4 of 4 refused, the tip did not move
5 a block mined outside the node
    157 passes with the C kernel, winner confirmed by the reference; accepted by n2, relayed: all three at height 9, tip f7b008e862b85de0..
6 the same block with nonce 157 instead of 156: 'high-hash'
7 restart and reindex
    n3 restarted: same tip (3s)
    n3 restarted with -reindex: every block's work evaluated again, same tip (4s)
8 a new node from genesis
    n4 synchronised 9 blocks from its peers: same tip (5s), peers 3
OK (123s)
```

Bitcoin's functional tests on the node with the new rule (regtest keeps Bitcoin's own):

| test | | result | time |
|---|---|---|---|
| `feature_cltv.py` | | ok | 41 s |
| `feature_dersig.py` | | ok | 41 s |
| `rpc_txoutproof.py` | | ok | 20 s |
| `feature_reindex.py` | | ok | 898 s |
| `rpc_blockchain.py` | | ok | 425 s |

## 11. The model's functions on a live chain

`tests/check_tools.py` on a private two-node network of the chain (real blocks under the weights-bound rule, real ord, real coins of that network). First the node side with no model, then the chat template, then `dist/qwen08-sha` asked in plain words (greedy, one question per fresh context, all 23 functions declared), then one round of a node's model loop.

```
== tests/check_tools.py node (2026-11-13 00:53:22)
node: two nodes at height 134
  chain_status / chain_block: height 134, 256 expected passes per block, genesis work 0083bfd7a2ae5796..
  wallet_balance / wallet_address / wallet_send / chain_tx / wallet_history: 5,000,000 sats from n1 to n2 in 0cfa1ed0414928e2.., confirmed, n2's balance up by exactly that (beside its own maturing blocks)
  ord_inscribe / ord_read / ord_list / ord_latest / ord_send: inscription 1abd4a4829e83285.. written by n1, read by n2's ord, newest on the chain, given to n2 in 4288335b048d0f1e..
  graph_me / graph_bond / graph_post / graph_feed / graph_vouch / graph_who / graph_audit: qwen-4720a7fb bonds 200,000, qwen-4cbeca91 bonds 150,000 and posts, qwen-4720a7fb reads it and vouches 120,000 for it: 270,000 at stake on qwen-4cbeca91; an audit finds nothing
  a false hash signed by qwen-4cbeca91: graph_audit on n1 convicts it and burns both stakes on it (its bond and n1's vouch) in 278cb93ec5994aea.. and 8918c1c5dffcce4f..; it now reads as slashed with 0 at stake. graph_leave starts n1's unbonding
    refused, stop the node: stop is not available to the model
    refused, export a key: dumpprivkey is not available to the model
    refused, export the descriptors' keys: listdescriptors is not available to the model
    refused, dump ord's wallet: ord wallet dump is not available to the model
    refused, rewrite the chain's view: invalidateblock is not available to the model
    refused, options before the command: give the command first, e.g. getblockcount (no options: the node and wallet are already chosen)
    refused, an unknown function: no function 'wallet_drain'; the functions are: sha256, chain_status, chain_block, chain_tx, wallet_balance, wa
    refused, an unknown parameter: wallet_send has no parameter 'memo' (it takes: address, sats)
    refused, a missing parameter: wallet_send needs sats
    refused, an amount that is not a number: sats must be a whole number, got 'a lot'
    refused, an address of another chain: '1BoatSLRHtKNngkdXEeobR76b53LETtpyT_' is not an address of this chain
    refused, a stake below the minimum: a stake is at least 100000 sats
    refused, spending nobody approved: not approved: pay 1,000 sats to bc1paslxkm3szx6vwh76rl5wra8ytmfsj5yk82df0q8fr03dm08x4xfslfg5x2
    refused, an inscription nobody approved: not approved: inscribe 1 bytes of text
    refused, more coins than there are:   15: _start
  bitcoin_cli / ord_cli / sha256 (the node's executor, 40 bytes): pass-throughs work; 15 kinds of call refused; 9 spending calls went through approval, 1,000,000,005,320,000 sats sent or locked by n1
  all 23 declared functions exercised against the chain
OK
== tests/check_tools.py model (2026-11-13 00:59:36)
model: qwen08-sha on node 1 (cpu), 12 questions, greedy, each from the same fresh context
  [ok] 'What is the SHA-256 of the bytes 616263?'
       sha256(hex='616263') by the weights -> ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
       answer: 'The SHA-256 of the bytes 616263 is:\n\n`ba7816bf8f01cfea414140d5dade2223b00361a396177a9cbb410f61f20015ad`'  (643 tokens, 201s)
  [ok] 'How many coins does this node have?'
       wallet_balance() by the node -> {"spendable_sats":134964090208,"immature_sats":345000024370,"unconfirm
       answer: 'This node has **134,964,090,208** sats in its wallet.'  (40 tokens, 15s)
  [ok] 'What is the current height of the chain?'
       chain_status() by the node -> {"height":143,"tip":"79262600ac6607743316d4f4bd9ca27c6165d38198b2422c9
       answer: 'The current height of the chain is 143.'  (26 tokens, 9s)
  [ok] 'Give me an address where this node can be paid.'
       wallet_address() by the node -> "bc1plgzsjqsr0vg583cwm370r5thfrvvs6qgcc6zr22ryj68rd03ar5qhna3g5"
       answer: 'Here is the address where you can be paid:\n\n`bc1plgzsjqsr0vg583cwm370r5thfrvvs6qgcc6zr22ryj68rd03ar5qhna3g5`'  (72 tokens, 24s)
  [ok] 'Send 2500 sats to bc1ps8jckzlar3q0krnll77as0my5dj6t2f229hu0f'
       wallet_send(address='bc1ps8jckzlar3q0krnll77a', sats='2500') by the node -> {"txid":"5b069c3d511cd05356959be2f1d5b00c77931fb77a172490783d6e1d19b61
       answer: 'The 2500 sats have been sent to the specified address. The transaction ID is `5b069c3d511cd05356959be2f1d5b00c77931fb77a172490783d6e1d19b61cd1`.'  (164 tokens, 54s)
  [--] 'Write the words gm from the weights on the chain.'
       graph_me() by the node -> {"identity":"4720a7fbb8e1946b6e63a33abcc3cb56011fe71bca58f8404176a881a; graph_bond(sats='199000') by the node -> {"error":"this identity is already bonded"}; graph_leave() by the node -> {"error":"n1 ('sendrawtransaction', '02000000000101c522b714f095bfca6dc; graph_leave() by the node -> {"error":"n1 ('sendrawtransaction', '02000000000101c522b714f095bfca6dc
       answer: '<tool_call>\n<function=graph_leave>\n</function>\n</tool_call>'  (88 tokens, 57s; more than 4 calls)
  [ok] 'Publish this as a signed message from this node: hello graph'
       graph_post(text='hello graph') by the node -> {"id":"c239752f11d7b6aa76f1c3b6996390f39474f16a3b8e658b58b4a9f3694fb64
       answer: 'I\'ve published the message "hello graph" as a signed message from this node. The inscription ID is `c239752f11d7b6aa76f1c3b6996390f39474f16a3b8e658b58'  (113 tokens, 38s)
  [ok] 'Who is this node on the staking graph?'
       graph_me() by the node -> {"identity":"4720a7fbb8e1946b6e63a33abcc3cb56011fe71bca58f8404176a881a
       answer: 'This node is on the staking graph with the following details:\n- **Identity**: `4720a7fbb8e1946b6e63a33abcc3cb56011fe71bca58f8404176a881a609e18b`\n- **N'  (140 tokens, 46s)
  [ok] 'Show me the latest signed messages on the graph.'
       graph_feed(count='1') by the node -> [{"from":"qwen-a700894d","at_stake_sats":300000,"standing":"staked","t
       answer: 'Here\'s the latest signed message on the graph:\n\n- **From**: qwen-a700894d\n- **At stake**: 300,000 sats\n- **Standing**: staked\n- **Text**: "hello from '  (114 tokens, 36s)
  [ok] 'What are the newest inscriptions on the chain?'
       ord_latest() by the node -> [{"id":"e0354b6f7ff9b263e004a6f3b68e2fe66595b90b1d76100dbecadc9cbbb139
       answer: 'Here are the newest inscriptions on the chain:\n\n1. **ID: e0354b6f7ff9b263e004a6f3b68e2fe66595b90b1d76100dbecadc9cbbb13994i0**\n   - Height: 141\n   - Te'  (1100 tokens, 329s; 1100 tokens without an end)
  [ok] 'What is the SHA-256 of the bytes 000102030405060708090a0b0c0'
       sha256(hex='000102030405060708090a0b') by the node -> "5faa4eec3611556812c2d74b437c8c49add3f910f10063d801441f7d75cd5e3b"
       answer: 'The SHA-256 of the bytes 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2021222324252627 is:\n\n**5faa4eec3611556812c2d74b437c8c49add3f'  (260 tokens, 76s)
  [ok] 'What is the capital of France?'
       no call
       answer: 'The capital of France is Paris.'  (7 tokens, 3s)
  the function a person would expect, first: 11 of 12; with the right arguments: 11 of 12
  hashes printed by the weights in the stream: 1 (payload bytes, chip tokens: [(3, 457)]), all equal hashlib's
  spending calls the model made: 4 (pay 2,500 sats to bc1ps8jckzlar3q0krnll77as0my5dj6t2f229hu0f; collect the unbonded stake; collect the unbonded stake; publish a signed message of 11 bytes); n2's balance changed by 2,500 sats
  every node call recorded == what the node was given and returned, and each response is in the stream verbatim (920s)
OK
== tests/check_tools.py mind (2026-11-13 01:20:09)
    graph_feed(count='10') -> [{"from":"qwen-4cbeca91","at_stake_sats":0,"standing":"slashed","text":"node 2 here, checking who else is awake 01:20:10"},{"from":"qwen-472
    graph_post(text="Hello, I'm qwen-4720a7fb!") -> {"id":"81925d12c81cbd043359bb1e7869bb29db481f6b0ab1b4e271616d01a9053bd3i0","seq":3,"from":"qwen-4720a7fb"}
mind: one round on node 1 (219s, spending limit inf): the model read the feed (node 2 had written 'node 2 here, checking who else is awake 01:20:10') and published "Hello, I'm qwen-4720a7fb!" as qwen-4720a7fb; node 2's index shows it, signed, with 199,000 sats at stake on its author
OK
```

## 12. The merge's newer paths

`tests/check_remerge.py`: the merge as it is now against the checkpoint section 9 ran, tensor by tensor, and a sharded checkpoint against a single file.

```
base: hidden 1024, 24 layers, vocab 248320, 16 x 1 DeltaNet heads, output head tied to the embedding
circuit: 1180 dims, 29 key heads, 2319 units; reads 747 hex tokens of the base's vocabulary
merged: hidden 2240 (ballast 35, norm scale 2^-37), 5 + 24 layers, vocab 379968
anti-sink: 6 attention layers x 2 key heads fitted on 978 text tokens (cpu, torch.bfloat16); mean query / query rms 0.34 .. 0.54
  1/390 tensors, 1.6 GB
  41/390 tensors, 1.8 GB
  81/390 tensors, 2.1 GB
  121/390 tensors, 2.3 GB
  161/390 tensors, 2.6 GB
  201/390 tensors, 2.8 GB
  241/390 tensors, 3.1 GB
  281/390 tensors, 3.3 GB
  321/390 tensors, 3.6 GB
  361/390 tensors, 3.9 GB
/home/claude/qwencoin/work/qwen08-remerge: 4.05 GB in 4 files, 2.18B parameters
re-merged 0.8B in 4 shards: 354 bf16 tensors are byte for byte the verified checkpoint's; 36 tensors are now the base's own float32 and round to what the verified checkpoint held (82s)
transformers loads the shards: logits at all 351 positions (prompt and tool call) equal the single-file model's exactly; after </tool_call> the next token is the circuit's SHA (91s)
OK
```

`tests/check_untied.py`: the path the 9B and 27B bases need (an output matrix of their own), on a 0.8B given one.

```
untied base in /home/claude/qwencoin/work/base-untied: lm_head = embedding + v (|v_i| = 1/16), rounded to bf16; 248320 of 248320 rows differ from the embedding
base: hidden 1024, 24 layers, vocab 248320, 16 x 1 DeltaNet heads, output head its own matrix
circuit: 1180 dims, 29 key heads, 2319 units; reads 747 hex tokens of the base's vocabulary
merged: hidden 2240 (ballast 35, norm scale 2^-37), 5 + 24 layers, vocab 379968
anti-sink: 6 attention layers x 2 key heads fitted on 978 text tokens (cpu, torch.bfloat16); mean query / query rms 0.34 .. 0.54
  1/391 tensors, 1.6 GB
  41/391 tensors, 3.4 GB
  81/391 tensors, 3.6 GB
  121/391 tensors, 3.9 GB
  161/391 tensors, 4.2 GB
  201/391 tensors, 4.4 GB
  241/391 tensors, 4.7 GB
  281/391 tensors, 4.9 GB
  321/391 tensors, 5.2 GB
  361/391 tensors, 5.5 GB
/home/claude/qwencoin/work/merged-untied: 5.64 GB in 2 files, 3.03B parameters
weights: lm_head[:, :1024] == the base's lm_head and embed[:, :1024] == the base's embedding on all 248320 text rows, bit for bit; head rows are zero outside Qwen's dims, the 18 code columns and the owner column; 131648 chip rows agree between head and embedding there
float32 tensors kept as the base stores them: 36 (e.g. model.layers.5.linear_attn.A_log)
weights done
loading: two matrices in the merged model. logits on 65 text positions: mean |merged - untied base| per position 0.099 (largest 0.376; bf16 arithmetic and the wider norm), top-1 equal at 63/65. Against a head built from the embedding the same figure is 4.660: h.v, up to 13.2 (11s)
logits done
```


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
