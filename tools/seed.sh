#!/bin/bash
# One public node of the qwencoin chain, for a machine with a public address (a small VPS):
# the meeting point that nodes behind routers dial out to.
#
#   bash tools/seed.sh            node + ord + miner + agent, listening on 0.0.0.0:8733, and the chain's
#                                 public pages on 0.0.0.0:8080: ord's explorer and the graph, read-only
#                                 (tools/web.py; QWEN_WEB_PORT=0 for none, or another port)
#   bash tools/seed.sh status | down
#
# Needs python3 with numpy and, once, the chain's work table: qwork.bin from the project's
# Hugging Face repository (it is in a clone of it), or built here from Qwen3.5-0.8B.
# Open TCP port 8733. Others join with a file SEEDS holding THIS_HOST:8733, or QWEN_SEEDS.
# QWEN_EXTERNAL=host:port if others reach it under another address than its own (a mapped port):
# the node then tells its peers that one. AGENTS=0: relay and mine only, no identity on the graph.
set -eu
cd "$(dirname "$0")/.."
case "${1:-up}" in status|down|watch) exec python3 tools/chain.py "$1" ;; esac
python3 -c "import numpy" 2>/dev/null || python3 -m pip install --user --quiet numpy || python3 -m pip install --user --quiet --break-system-packages numpy
mkdir -p dist
[ -f dist/qwork.bin ] || { [ -f qwork.bin ] && ln -fL qwork.bin dist/qwork.bin 2>/dev/null || { [ -f qwork.bin ] && cp -L qwork.bin dist/qwork.bin; } || true; }
if [ ! -f dist/qwork.bin ]; then
  echo "building the work table from Qwen3.5-0.8B (a 1.7 GB download, once)"
  python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user --quiet huggingface_hub || python3 -m pip install --user --quiet --break-system-packages huggingface_hub
  python3 tools/fetch.py Qwen/Qwen3.5-0.8B base/Qwen3.5-0.8B
  PYTHONPATH=. python3 -m hashformer.work build base/Qwen3.5-0.8B dist/qwork.bin
fi
PYTHONPATH=. python3 -m hashformer.work info dist/qwork.bin | grep -q "(the chain's table)" || { echo "dist/qwork.bin is not the chain's table"; exit 1; }
chmod +x bin/* libexec/* lib/ld-linux-*.so.* 2>/dev/null || true
export QWEN_WEB_PORT="${QWEN_WEB_PORT-8080}" QWEN_WEB_BIND="${QWEN_WEB_BIND:-0.0.0.0}"
[ -f /status/log ] && export QWEN_WEB_STATUS="${QWEN_WEB_STATUS:-/status}"  # a rented pod's own status files go on being served
QWEN_BIND=0.0.0.0 MINDS=0 PYTHONPATH=. python3 tools/chain.py up 1
echo "this node listens on port 8733. Others reach it with a file named SEEDS holding  $(hostname -I 2>/dev/null | awk '{print $1}'):8733  (use the public address)"
