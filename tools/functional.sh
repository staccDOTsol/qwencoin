#!/usr/bin/env bash
# Run Bitcoin's functional tests against qwencoind (regtest). The node hashes in the weights; the
# Python test framework hashes with hashlib and signs with its own secp256k1. Every txid, block
# hash, sighash and template hash the two sides exchange is therefore cross-checked between the
# model and an independent implementation.
#
#   tools/functional.sh <log> <test.py> [test.py ...]     e.g. tools/functional.sh logs/f.log feature_opcat.py
#
# Needs upstream/bitcoin configured (tools/build.sh node) for build/test/config.ini.
cd "$(dirname "$0")/.."
ROOT=$PWD; log=$1; shift
export BITCOIND=$ROOT/bin/qwencoind BITCOINCLI=$ROOT/bin/qwencoin-cli
for t in "$@"; do
  t0=$(date +%s)
  tmp=$(mktemp -u -d /tmp/qwfunc.XXXXXX)
  out=$(cd upstream/bitcoin && timeout "${LIMIT:-3000}" python3 build/test/functional/$t --timeout-factor=40 --tmpdir="$tmp" 2>&1)
  rc=$?
  if grep -q 'Tests successful' <<<"$out"; then res="ok"; rm -rf "$tmp"
  elif grep -q 'Test skipped' <<<"$out"; then res="skipped: $(grep -o 'Test skipped.*' <<<"$out" | head -1)"
  elif [ "$rc" = 124 ]; then res="TIMEOUT"
  else res="FAILED: $(grep -E 'Error|error|Assert' <<<"$out" | tail -3 | tr '\n' ' ' | cut -c1-400) (log dir $tmp)"; fi
  echo "$t: $res [$(( $(date +%s) - t0 ))s]" | tee -a "$log"
done
