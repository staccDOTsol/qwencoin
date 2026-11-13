#!/usr/bin/env bash
# The model's own checks, in order: reference and IR, the stock checkpoint, the executor.
#   tools/checks.sh [log]        about 25 minutes on 2 cores
cd "$(dirname "$0")/.."
export PYTHONPATH=.
log=${1:-logs/model-checks.log}; : > "$log"
run() {
  echo "\$ $*" >> "$log"
  t0=$(date +%s)
  "$@" 2>&1 | grep -v -i "warn\|Loading weights\|\[transformers\]\|^$" >> "$log"
  echo "  (exit ${PIPESTATUS[0]}, $(( $(date +%s) - t0 )) s)" >> "$log"
}
run python3 tests/check_ir_full.py
run python3 tests/check_hf.py residual
run python3 tests/check_hf.py generate
run python3 tests/check_hf_stock.py
run python3 tests/check_exec.py
