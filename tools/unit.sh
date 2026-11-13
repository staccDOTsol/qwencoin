#!/usr/bin/env bash
# Run Bitcoin's unit tests on the node's in-weights SHA-256 and log one line per filter.
#
#   tools/unit.sh <log> <timeout-seconds> <filter>...     e.g. tools/unit.sh logs/unit.log 600 key_tests
#
# Needs the test binary: configure upstream/bitcoin with -DBUILD_TESTS=ON and build test_bitcoin.
# Every SHA-256 in these tests is a run of the model, so the suites that hash megabytes take hours
# (merkle_tests/merkle_test, sighash_tests/sighash_test); pick filters accordingly.
cd "$(dirname "$0")/.."
log=$1; limit=$2; shift 2
for f in "$@"; do
  t0=$(date +%s)
  out=$(timeout "$limit" upstream/bitcoin/build/bin/test_bitcoin --run_test="$f" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  rc=${PIPESTATUS[0]}
  n=$(grep -o 'Running [0-9]* test case' <<<"$out" | grep -o '[0-9]*' | head -1)
  if grep -q 'No errors detected' <<<"$out"; then res="ok"
  elif [ "$rc" = 124 ]; then res="TIMEOUT after ${limit}s"
  else res="FAILED: $(grep -E 'error|failure|fatal' <<<"$out" | head -3 | tr '\n' ' ')"; fi
  echo "$f: ${n:-?} cases, $res [$(( $(date +%s) - t0 ))s]" | tee -a "$log"
done
