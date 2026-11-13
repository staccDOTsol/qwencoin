#!/usr/bin/env bash
# What a newcomer does, on a runner: the one command of the cards, and then the node has to be
# on the public chain, level with the public node it dialed.
#
#   tools/ci-join.sh published   the installer as it is on Hugging Face now (curl ... | bash)
#   tools/ci-join.sh head        this checkout's installer, with this checkout's files laid over
#                                the download: a change is tried before it is published.
#                                QWENCOIN_NODE_URL=file:///.../qwencoin-node-PLATFORM.tar.gz: with that
#                                package's programs (one just built); TORCH=none: the node alone, no model;
#                                SYNC_BLOCKS=n: enough when the node has taken n blocks of the public chain
#                                from the public node and checked them (the whole chain takes longer each day)
#
# The node does not mine here (MINERS=0): coins mined on a runner would be lost with it.
set -uo pipefail
HOW=${1:-head}
SRC="$(cd "$(dirname "$0")/.." && pwd)"
ACCOUNT=$(head -1 "$SRC/ORIGIN" | tr -d ' \r\n'); GITHUB=$(head -1 "$SRC/GITHUB" | tr -d ' \r\n')
W="${RUNNER_TEMP:-/tmp}/qwencoin-join"
mkdir -p "$W"; cd "$W" || exit 1
export QWENCOIN_DIR="$W/qwencoin" MINERS=0
D="$QWENCOIN_DIR"
say() { echo "ci-join: $*"; }
cli() { "$D/bin/qwencoin-cli" -datadir="$D/run/n1" "$@"; }
py() { python3 -c "$@"; }

say "$HOW on $(uname -srm), $( (nproc 2>/dev/null || sysctl -n hw.ncpu) ) cpus"
PLAT="$(uname -s | tr 'A-Z' 'a-z' | sed 's/darwin/macos/')-$(uname -m)"
if [ "$PLAT" != linux-x86_64 ] && [ -z "${QWENCOIN_NODE_URL:-}" ] && ! curl -fsL -r 0-0 -o /dev/null "https://github.com/$GITHUB/releases/latest/download/qwencoin-node-$PLAT.tar.gz" 2>/dev/null; then
  say "SKIPPED: https://github.com/$GITHUB/releases has no node for $PLAT yet"
  exit 0
fi
if [ "$HOW" = published ] && [ "$PLAT" != linux-x86_64 ] && ! curl -fsSL "https://huggingface.co/$ACCOUNT/qwencoin/resolve/main/install.sh" | grep -q "releases/latest/download"; then
  say "SKIPPED: the installer on Hugging Face is the one from before the other platforms' builds"
  exit 0
fi
if [ "$HOW" = published ]; then
  say "curl -fsSL https://huggingface.co/$ACCOUNT/qwencoin/resolve/main/install.sh | bash"
  ( curl -fsSL "https://huggingface.co/$ACCOUNT/qwencoin/resolve/main/install.sh" | bash ) > install.log 2>&1 &
else
  ( sed "s/@ACCOUNT@/$ACCOUNT/g; s#@GITHUB@#$GITHUB#g" "$SRC/tools/install.sh" | QWENCOIN_OVERLAY="$SRC" bash ) > install.log 2>&1 &
fi
PID=$!
tail -n +1 -f install.log 2>/dev/null &
TAIL=$!
end() {  # end MESSAGE CODE: stop the node, keep what it wrote
  say "$1"
  [ -f "$D/tools/pipeline.sh" ] && (cd "$D" && bash tools/pipeline.sh down 2>&1 | tail -2)
  kill "$PID" "$TAIL" 2>/dev/null
  mkdir -p "$SRC/ci/join"
  cp install.log "$SRC/ci/join/install.log" 2>/dev/null
  for f in "$D"/logs/*.txt "$D"/logs/*.log "$D"/run/n1/debug.log "$D"/run/*.log "$D"/run/ord1/server.log; do
    [ -f "$f" ] && tail -c 200000 "$f" > "$SRC/ci/join/$(basename "$(dirname "$f")")-$(basename "$f")"
  done
  exit "$2"
}

# the installer ends by following the chain; it has done its part when logs/DONE is there
t0=$(date +%s)
while kill -0 "$PID" 2>/dev/null && [ ! -f "$D/logs/DONE" ]; do
  [ $(( $(date +%s) - t0 )) -gt "${JOIN_PATIENCE:-3600}" ] && end "FAILED: the installer did not finish within ${JOIN_PATIENCE:-3600}s" 1
  sleep 5
done
if [ ! -f "$D/logs/DONE" ]; then wait "$PID"; end "FAILED: the installer ended without joining (exit $?)" 1; fi
say "the installer has joined after $(( $(date +%s) - t0 ))s"

# on the public chain: connected to the node it dialed, and level with it
t0=$(date +%s); OK=""
while [ $(( $(date +%s) - t0 )) -lt "${SYNC_PATIENCE:-2400}" ]; do
  INFO=$(cli getblockchaininfo 2>/dev/null); PEERS=$(cli getpeerinfo 2>/dev/null)
  LINE=$(py '
import json, sys
info, peers = json.loads(sys.argv[1] or "{}"), json.loads(sys.argv[2] or "[]")
best = max([max(p.get("synced_headers", -1), p.get("startingheight", -1)) for p in peers] or [-1])
want = min(best - 2, int(sys.argv[3])) if int(sys.argv[3]) > 0 else best - 2
print(info.get("blocks", -1), best, len(peers), int(bool(peers) and best > 0 and info.get("blocks", -1) >= want), info.get("bestblockhash", ""))' "$INFO" "$PEERS" "${SYNC_BLOCKS:-0}" 2>/dev/null)
  set -- $LINE
  say "height ${1:-?}, the best a peer has ${2:-?}, peers ${3:-?} ($(( $(date +%s) - t0 ))s)"
  if [ "${4:-0}" = 1 ]; then OK=1; break; fi
  sleep 20
done
[ -n "$OK" ] || end "FAILED: the node did not reach the public node's chain within ${SYNC_PATIENCE:-2400}s" 1
[ "$(cli getblockhash 0)" = 51ebb1f392844e411a0c52ba535773b0cb4a04cac6aede0d822249d5c6ce3cf5 ] || end "FAILED: another genesis" 1
say "on the public chain: height $1 (peers have $2), $3 peer(s), tip $5, genesis 51ebb1f3..${SYNC_BLOCKS:+ (asked for: $SYNC_BLOCKS blocks taken from the public node and checked)}"
(cd "$D" && bash tools/pipeline.sh status 2>&1 | cut -c1-300)

# the node's model, where it was started: one question that it has to answer through the node
if ls "$D"/.done/joined-* >/dev/null 2>&1; then
  # One copy of the model at a time: the node's own is stopped before another is loaded to be asked. (Join run 7 lost
  # both of its linux-aarch64 runners at this step, "the hosted runner lost communication", the way a machine out of
  # memory goes: two copies of the model can be more than a runner's 16 GB.)
  say "the model was started: $(grep -h -m1 "loaded" "$D"/run/mind1.log 2>/dev/null | cut -c1-120 || true)"
  (cd "$D" && .venv/bin/python tools/chain.py pause-minds 2>&1 | tail -1)
  say "asking it for the node's coins"
  (cd "$D" && bash tools/pipeline.sh ask "How many coins does this node have?" 2>&1 | cut -c1-600 | tee "$W/ask.log")
  grep -q "wallet_balance" "$W/ask.log" || end "FAILED: the model did not ask the node for its balance" 1
  cp "$W/ask.log" "$D/logs/ask.log"
else
  say "no model on this machine (the installer says why above); the node runs without it"
fi
end "OK" 0
