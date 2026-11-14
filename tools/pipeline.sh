#!/bin/bash
# qwencoin on a Linux machine (native, or WSL2 on Windows): everything, in order, resumable.
#
#   bash tools/pipeline.sh                  all stages; a finished stage is skipped on the next run. The nodes
#                                           come up first and keep mining while the models are merged, checked
#                                           and published one size after the other; it ends following the chain
#   bash tools/pipeline.sh join             the quick way onto the network: one node (NODES=n for more) with its ord,
#                                           miner and agent, and the PUBLISHED merged 0.8B as its model; nothing is
#                                           merged or published here. It dials the nodes in the file SEEDS.
#   bash tools/pipeline.sh up [N] | status | watch | down      the network alone
#   bash tools/pipeline.sh ask "question"   one question to the largest merged model, driving node 1
#   bash tools/pipeline.sh chat             a conversation with it
#   bash tools/pipeline.sh publish          push what has passed its checks and is not on Hugging Face yet
#   bash tools/pipeline.sh seed [status|logs|down]   a public node of the chain on a rented RunPod pod (tools/runpod.py)
#   bash tools/pipeline.sh diag             what runs here and what it last said, into logs/diag.txt
#   bash tools/pipeline.sh said             everything said on the chain as plain text: every signed post, every text inscription,
#                                           the stakes, vouches and burns (also into logs/said.txt)
#   bash tools/pipeline.sh cards            every card on Hugging Face as things are now (no weights are sent)
#   bash tools/pipeline.sh refresh          restart the miners, agents, ord servers and node models from the files as they are now
#   bash tools/pipeline.sh minds [SIZE]     which merged model the nodes here run as their own (minds 2B: the merged 2B for each
#                                           node, if that many fit in this machine's memory); without SIZE: what they run now
#   bash tools/pipeline.sh recheck SIZE     a merged model's functions on the running chain once more; its card and logs again
#
#   NODES=4   MODELS="0.8B 2B 4B 9B 27B"   TORCH=auto|cpu|cuda   BACKEND="cpu float32"   QUICK=1   FORCE=1
#   LOGS=dir   NO_NET=1   NO_PUBLISH=1   NO_TOOLS=1   HF_TOKEN=... (or a token file, see tools/publish.py)
#
# It makes its own Python (uv) and keeps everything, caches included, inside this folder.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
[ -f "$ROOT/local.env" ] && { set -a; . "$ROOT/local.env"; set +a; }  # this folder's own settings (QWEN_PORT=..., for a second node on one machine)
LOGS="${LOGS:-$ROOT/logs}"
MODELS="${MODELS:-$(cat MODELS 2>/dev/null || echo "0.8B 2B 4B 9B 27B")}"  # Qwen3.5 sizes to merge, smallest first
mkdir -p "$LOGS" work dist base .done
export UV_CACHE_DIR="$ROOT/.uv/cache" UV_PYTHON_INSTALL_DIR="$ROOT/.uv/python" UV_NO_CONFIG=1
export HF_HOME="$ROOT/.hf" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPYCACHEPREFIX="$ROOT/.pycache" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
UV="$ROOT/.uv/bin/uv"
HASHER_SHA=ec5f7cb21e9c95bfc82576cb52a8a0f01cf3b09b7ccaa22cf6f8e6aa86b2320c
filter() { grep --line-buffered -v "^\[transformers\]\|^Loading weights\|^Loading checkpoint"; }
PLAT="$(uname -s | tr 'A-Z' 'a-z' | sed 's/darwin/macos/')-$(uname -m)"  # linux-x86_64, linux-aarch64, macos-arm64, macos-x86_64
wsl() { grep -qi microsoft /proc/version 2>/dev/null; }
ram_gb() { if [ -r /proc/meminfo ]; then awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo; else echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )); fi; }
free_gb() { if df -BG --output=avail . >/dev/null 2>&1; then df -BG --output=avail . | tail -1 | tr -dc 0-9; else df -g . | tail -1 | awk '{print $4}'; fi; }
guarded() {  # guarded CMD...: a step that loads whole models. Out of memory, a machine does not fail the step: it stops answering,
  # nodes included (the 9B's merge did that to a 62 GB machine). The step is ended while there is still some left.
  "$@" &
  local pid=$! low=0 av sf st tot
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    read -r av sf st tot < <(awk '/^MemAvailable/ {a=$2} /^SwapFree/ {s=$2} /^SwapTotal/ {t=$2} /^MemTotal/ {m=$2} END {printf "%d %d %d %d\n", a/1024, s/1024, t/1024, m/1024}' /proc/meminfo 2>/dev/null) || continue
    [ -n "${tot:-}" ] && [ "$tot" -gt 0 ] || continue
    if [ "$st" -gt 0 ]; then  # with swap: little left and a quarter of the swap already eaten
      if [ "$av" -lt $(( tot / 20 > 3072 ? tot / 20 : 3072 )) ] && [ "$sf" -lt $(( st * 3 / 4 )) ]; then low=$((low + 1)); else low=0; fi
    else                      # without: next to nothing left
      if [ "$av" -lt $(( tot / 40 > 512 ? tot / 40 : 512 )) ]; then low=$((low + 1)); else low=0; fi
    fi
    if [ "$low" -ge 2 ]; then
      echo "STOPPED: this step was running the machine out of memory ($av MB left of $tot, swap $sf of $st MB free). It is ended before the nodes and everything else stop answering."
      kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      return 137
    fi
  done
  wait "$pid"
}
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
vram_gb() { "$PY" -c "import json; print(json.load(open('$LOGS/backend.json')).get('cuda', {}).get('gb', 0))" 2>/dev/null || echo 0; }
dir_of() { case $1 in 0.8B) echo dist/qwen08-sha ;; 2B) echo dist/qwen2b-sha ;; 4B) echo dist/qwen4b-sha ;; 9B) echo dist/qwen9b-sha ;; 27B) echo dist/qwen27b-sha ;; esac; }
gb_of() { "$PY" -c "import sys, pathlib; print(round(sum(f.stat().st_size for f in pathlib.Path(sys.argv[1]).glob('*.safetensors')) / 2**30, 1))" "$1"; }
largest() { for M in 27B 9B 4B 2B 0.8B; do { [ -f ".done/merged-$M" ] || [ -f ".done/joined-$M" ]; } && { dir_of $M; return; }; done; }

done_p() { [ -f ".done/$1" ] && [ -z "${FORCE:-}" ]; }
mark() { date '+%Y-%m-%d %H:%M:%S' > ".done/$1"; }
fail() { echo "FAILED: $*"; [ -n "${STAGES:-}" ] && echo "$*" > "$LOGS/FAILED"; exit 1; }
env_stage() {  # Python and packages, inside this folder
  if ! done_p env || [ ! -x "$PY" ]; then
    echo "== env: Python, torch, transformers (inside this folder)"
    if [ ! -x "$UV" ]; then
      (curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null || wget -qO- https://astral.sh/uv/install.sh) \
        | env UV_UNMANAGED_INSTALL="$ROOT/.uv/bin" sh || fail "could not install uv (no curl or wget, or no network)"
    fi
    [ -x "$PY" ] || "$UV" venv --python 3.12 "$ROOT/.venv" || fail "uv venv"
    T="${TORCH:-auto}"
    if [ "$T" = auto ]; then T=cpu; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && T=cuda; fi
    IDX=https://download.pytorch.org/whl/cpu; [ "$T" = cuda ] && IDX=https://download.pytorch.org/whl/cu126
    # A node (qwencoind, ord, miner, agent) needs Python and numpy. torch is for the model: where there is no build
    # of it (Intel Macs) or it does not install, a node that joins runs without a model; merging needs it.
    rm -f .done/no-torch
    if [ "$PLAT" = macos-x86_64 ] || [ "$T" = none ]; then
      echo "   torch: none for $PLAT"; : > .done/no-torch
    elif [ "$(uname -s)" = Darwin ]; then
      echo "   torch build: macOS"
      "$UV" pip install --python "$PY" torch==2.14.1 || : > .done/no-torch
    else
      echo "   torch build: $T"
      "$UV" pip install --python "$PY" torch==2.14.1 --index-url "$IDX" \
        || "$UV" pip install --python "$PY" torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu || : > .done/no-torch
    fi
    if [ -f .done/no-torch ]; then
      [ -n "${NODE_ONLY_OK:-}" ] || fail "torch could not be installed here, and merging the models needs it (a node alone does not:  bash tools/pipeline.sh join)"
      echo "   no torch on this machine: the node runs without a model"
      "$UV" pip install --python "$PY" numpy huggingface_hub || fail "packages"
    else
      "$UV" pip install --python "$PY" transformers==5.22.2 numpy safetensors huggingface_hub || fail "packages"
    fi
    mark env
  fi
}
table_here() {  # a copy of the project's repository holds the chain's work table itself, and the node programs of several platforms
  if [ ! -f dist/qwork.bin ] && [ -f qwork.bin ]; then ln -fL qwork.bin dist/qwork.bin 2>/dev/null || cp -L qwork.bin dist/qwork.bin; fi
  # node/PLATFORM/ holds this machine's programs: they are put in place as bin/ libexec/ lib/ (once, and again when they change).
  # Without that folder, bin/ is what it is: Linux x86-64 in the repository.
  if [ -d "node/$PLAT/bin" ] && { [ "$(cat bin/PLATFORM 2>/dev/null)" != "$PLAT" ] || [ "node/$PLAT/libexec/qwencoind" -nt bin/PLATFORM ]; }; then
    rm -rf bin libexec lib
    cp -R "node/$PLAT/bin" "node/$PLAT/libexec" .
    [ -d "node/$PLAT/lib" ] && cp -R "node/$PLAT/lib" .
    # the circuit's executor and the work kernel as libraries for this platform (what Python loads; the ones
    # in the repository's exec/ are Linux x86-64)
    if [ -d "node/$PLAT/exec" ]; then for f in "node/$PLAT/exec/"*; do cp "$f" "exec/$(basename "$f")"; done; fi
    echo "$PLAT" > bin/PLATFORM
  fi
  chmod +x bin/* libexec/* lib/ld-linux-*.so.* 2>/dev/null  # a download does not keep the mode
  return 0
}
NOTHING="nothing has been set up in this folder yet:  bash tools/pipeline.sh join  (one node on the public chain, with the published model)"

JOIN=""
case "${1:-}" in join|up) NODE_ONLY_OK=1 ;; esac
if [ "${1:-}" = join ]; then
  shift; JOIN=1; export NO_PUBLISH=1 NODES="${NODES:-1}"
  # A joined node's model posts and inscribes on its own, and pays or vouches up to this many sats in total
  # (MIND_LIMIT=inf: no limit, as on the nodes of the machine that publishes). Kept in local.env for later starts.
  grep -q "^MIND_LIMIT=" local.env 2>/dev/null || echo "MIND_LIMIT=${MIND_LIMIT:-100000000}" >> local.env
  export MIND_LIMIT="$(grep "^MIND_LIMIT=" local.env | tail -1 | cut -d= -f2)"
  # a kernel chosen by hand (BACKEND='cpu bfloat16') stays chosen for this folder
  [ -n "${BACKEND:-}" ] && ! grep -q "^BACKEND=" local.env && echo "BACKEND='$BACKEND'" >> local.env
fi
case "${1:-}" in
  up)   env_stage; table_here; exec "$PY" tools/chain.py "$@" ;;  # also on a fresh copy of the repository
  status|watch|down|mine|agent) [ -x "$PY" ] || { echo "$NOTHING"; exit 1; }; exec "$PY" tools/chain.py "$@" ;;
  ask|chat)
        CMD=$1; shift
        [ -x "$PY" ] || { echo "$NOTHING"; exit 1; }
        MODEL=${MODEL:-$(largest)}
        [ -n "$MODEL" ] || [ ! -f .done/fetched-0.8B ] || { echo "the published model is in this folder but was not started on this machine (memory):  BACKEND='cpu bfloat16' bash tools/pipeline.sh join  tries it all the same"; exit 1; }
        [ -n "$MODEL" ] || { echo "there is no merged model in this folder yet:  bash tools/pipeline.sh join  fetches the published one and starts this node with it"; exit 1; }
        PICK=${BACKEND:-$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$(gb_of "$MODEL")")} || exit 1
        echo "$MODEL on ${PICK% *} as ${PICK#* }, driving node ${NODE:-1} (a call that moves coins asks first)"
        export DEVICE=${PICK% *} DTYPE=${PICK#* }
        # (ask: under the memory guard. A second copy of the model beside the node's own can be more than the machine has.)
        if [ "$CMD" = ask ]; then guarded "$PY" -m rails.runtime --node "${NODE:-1}" "$MODEL" "${1:-What is the SHA-256 of the bytes 616263?}"; exit $?
        else exec "$PY" -m rails.runtime --node "${NODE:-1}" "$MODEL"; fi ;;
  seed) shift  # a public node on a rented RunPod pod: seed | seed status | seed logs | seed down
        { echo "== seed ${1:-seed} $(date '+%Y-%m-%d %H:%M:%S')"; "$PY" tools/runpod.py "${1:-seed}" "$LOGS" 2>&1; echo "(exit $?)"; } | tee -a "$LOGS/seed.log"
        exit "$(tail -1 "$LOGS/seed.log" | tr -dc 0-9)" ;;
  diag)  # a snapshot of everything that runs here, into logs/diag.txt (on Windows: readable from outside WSL)
        { date '+%Y-%m-%d %H:%M:%S'
          echo "--- processes (seconds running, cpu, MB)"
          { ps -eo pid,etimes,pcpu,rss,args --sort=-etimes 2>/dev/null || ps -axo pid,etime,pcpu,rss,args; } | awk '{ $4 = int($4 / 1024); print }' | grep -E "qwencoin|/ord |chain.py|check_|rails.runtime|publish.py|runpod.py|hashformer|fetch.py" | grep -v grep | cut -c1-240
          echo "--- file locks held"
          command -v lslocks >/dev/null 2>&1 && lslocks -o COMMAND,PID,TYPE,MODE,PATH 2>/dev/null | grep -E "COMMAND|\.lock"
          echo "--- status"
          $(command -v timeout >/dev/null 2>&1 && echo "timeout 180") "$PY" tools/chain.py status 2>&1 | cut -c1-400
          for f in run/agent*.log run/mind*.log run/miner*.log run/sweeper*.log; do [ -s "$f" ] && { echo "--- $f"; tail -14 "$f" | cut -c1-420; }; done
          for d in run/ord*; do [ -d "$d" ] || continue; echo "--- $d"; ls -la --time-style=+%H:%M:%S "$d" | grep -E "index|lock|pid" | awk '{print $6, $5, $7}'; [ -s "$d/server.log" ] && tail -4 "$d/server.log" | cut -c1-300; done
          for f in run/ord*/index-set-aside-*.txt; do [ -s "$f" ] && { echo "--- $f"; cut -c1-700 "$f"; }; done
          for d in run/n*; do [ -f "$d/debug.log" ] && { echo "--- $d/debug.log"; grep -v "\[ord\]" "$d/debug.log" | tail -12 | cut -c1-240; }; done
        } > "$LOGS/diag.txt" 2>&1
        echo "written to $LOGS/diag.txt ($(wc -l < "$LOGS/diag.txt") lines)"; exit 0 ;;
  minds) shift  # minds [SIZE]: which merged model the nodes of this folder run as their own (0.8B, 2B, 4B ..); kept in local.env
        [ -x "$PY" ] || { echo "$NOTHING"; exit 1; }
        M=${1:-}
        if [ -z "$M" ]; then
          NOW=$(grep "^MIND_MODEL=" local.env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "'")
          echo "the nodes' models here: ${NOW:-$ROOT/dist/qwen08-sha (the default)}"
          echo "merged models in this folder: $(for X in 0.8B 2B 4B 9B 27B; do [ -f "$(dir_of $X)/hashformer.json" ] && printf '%s ' "$X"; done)"
          echo "bash tools/pipeline.sh minds SIZE   changes it and starts the nodes' models again"; exit 0
        fi
        OUT=$(dir_of "$M"); [ -n "$OUT" ] || { echo "unknown size $M (0.8B, 2B, 4B, 9B, 27B)"; exit 1; }
        if [ ! -f "$OUT/hashformer.json" ] || { [ ! -f ".done/merged-$M" ] && [ ! -f ".done/joined-$M" ]; }; then
          # not merged in this folder: the published checkpoint of that size, checked here before a node is given it (as `join` does)
          REPO=${QWEN_MODEL_REPO:-$(head -1 ORIGIN 2>/dev/null | tr -d ' \r\n')/Qwen3.5-$M-qwencoin}
          case "$REPO" in /*) echo "there is no merged Qwen3.5-$M in this folder, and which account published the checkpoints is not known here (no file ORIGIN)"; exit 1 ;; esac
          echo "== the published checkpoint https://huggingface.co/$REPO (nothing is merged here)"
          "$PY" tools/fetch.py "$REPO" "$OUT" || { echo "the download of $REPO did not arrive; the nodes' models stay as they are"; exit 1; }
          mark "fetched-$M"
          PICK=${BACKEND:-$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$(gb_of "$OUT")" 2>/dev/null)} || PICK=""
          [ -n "$PICK" ] || { echo "this machine has $(ram_gb) GB of memory: not enough for the merged $M in a kernel that computes the hash exactly here. The nodes' models stay as they are."; exit 1; }
          echo "== that checkpoint on this machine (${PICK% *} as ${PICK#* }): a hash asked for in plain words, computed by its weights, against hashlib"
          DEVICE=${PICK% *} DTYPE=${PICK#* } "$PY" tests/check_published.py "$OUT" 2>&1 | filter | tee "$LOGS/published-$M.log"
          [ "${PIPESTATUS[0]}" = 0 ] || { echo "the published $M did not compute the hash here (see $LOGS/published-$M.log). The nodes' models stay as they are."; exit 1; }
          mark "joined-$M"
          { grep -v -E "^(DEVICE|DTYPE)=" local.env 2>/dev/null; echo "DEVICE=${PICK% *}"; echo "DTYPE=${PICK#* }"; } > local.env.new && mv local.env.new local.env
          export DEVICE=${PICK% *} DTYPE=${PICK#* }
        fi
        N=${NODES:-$(ls -d run/n[0-9]* 2>/dev/null | wc -l | tr -d ' ')}; [ "${N:-0}" -gt 0 ] || N=1
        # each copy in the kernel the nodes' models run in here (bfloat16 unless this folder says float32), and room for the nodes
        NEED=$(awk "BEGIN {print int($N * ($(gb_of "$OUT") * $([ "${DTYPE:-bfloat16}" = float32 ] && echo 2 || echo 1) * 1.08 + 1.5) + 4.99)}")
        [ "$NEED" -le "$(ram_gb)" ] || [ -n "${FORCE:-}" ] || { echo "$N models of that size need about $NEED GB of memory and this machine has $(ram_gb) GB (FORCE=1 tries it all the same)"; exit 1; }
        { grep -v "^MIND_MODEL=" local.env 2>/dev/null; echo "MIND_MODEL='$ROOT/$OUT'"; } > local.env.new && mv local.env.new local.env
        export MIND_MODEL="$ROOT/$OUT"
        echo "the nodes' models: Qwen3.5-$M ($OUT), $N of them, about $NEED GB"
        "$PY" tools/chain.py pause-minds && MINDS=1 "$PY" tools/chain.py up "$N"; exit $? ;;
  said)  # the chain's posts, text inscriptions, stakes, vouches and burns as plain text, from node 1's own ord (NODE=n for another)
        [ -x "$PY" ] || { echo "$NOTHING"; exit 1; }
        "$PY" tools/said.py "${NODE:-1}" 2> >(tee "$LOGS/said.err" >&2) | tee "$LOGS/said.txt"
        RC=${PIPESTATUS[0]}; [ "$RC" = 0 ] && echo "(also in $LOGS/said.txt)"; exit "$RC" ;;
  minds-log)  # what this machine's nodes' models did lately, into the project's repository (logs/minds.txt)
        exec "$PY" tools/publish.py minds "$LOGS" ;;
  cards)  # every card on Hugging Face as things are now (the public node's address, the latest logs); no weights are sent
        "$PY" tools/publish.py cards "$LOGS" 2>&1 | grep -v "No files have been modified" | tee -a "$LOGS/publish.log"; exit 0 ;;
  refresh)  # the miners, agents, ord servers and node models again, from the files as they are now (the nodes keep running)
        { echo "== refresh $(date '+%Y-%m-%d %H:%M:%S')"
          "$PY" tools/chain.py stop-helpers && MINDS=$([ -n "$(largest)" ] && echo "${MINDS:-1}" || echo 0) "$PY" tools/chain.py up "${NODES:-$(ls -d run/n[0-9]* 2>/dev/null | wc -l)}"
          echo "(exit $?)"; } 2>&1 | tee -a "$LOGS/refresh.log"
        exit "$(tail -1 "$LOGS/refresh.log" | tr -dc 0-9)" ;;
  recheck) shift  # recheck SIZE: its functions on the running chain once more, then its card and logs again
        M=${1:?recheck SIZE, e.g. recheck 0.8B}; OUT=$(dir_of "$M")
        [ -n "$OUT" ] && [ -f ".done/merged-$M" ] || { echo "Qwen3.5-$M has not passed its checks here"; exit 1; }
        PICK=${BACKEND:-$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$(gb_of "$OUT")")} || exit 1
        echo "== Qwen3.5-$M: its functions on the running chain again, as node 1 (the earlier run is kept beside it)"
        [ -f "$LOGS/tools-$M.log" ] && cp "$LOGS/tools-$M.log" "$LOGS/tools-$M-run-$(date +%H%M%S).log"
        DEVICE=${PICK% *} DTYPE=${PICK#* } "$PY" tests/check_tools.py live "$OUT" 2>&1 | filter | tee "$LOGS/tools-$M.log"
        [ "${PIPESTATUS[0]}" = 0 ] || { echo "FAILED: see $LOGS/tools-$M.log"; exit 1; }
        date '+%Y-%m-%d %H:%M:%S' > ".done/tools-$M"
        # its card and logs again; a size that had not passed before (and is not on Hugging Face yet) is published now
        { [ -f ".done/published-$M" ] || { [ -f .done/minds ] && [ -z "${NO_PUBLISH:-}" ]; }; } && "$PY" tools/publish.py model "$M" "$OUT" "$LOGS" 2>&1 | tee -a "$LOGS/publish.log"
        exit 0 ;;
  publish) shift  # the same gate as the full run: every node's model up and heard on the graph
        [ -f .done/minds ] || [ -n "${NO_NET:-}" ] || { "$PY" tools/chain.py minds "${NODES:-4}" && date '+%Y-%m-%d %H:%M:%S' > .done/minds; } || exit 1
        exec "$PY" tools/publish.py pending "$LOGS" "$@" ;;
esac

rm -f "$LOGS/FAILED" "$LOGS/DONE"
exec > >(tee -a "$LOGS/pipeline.log") 2>&1
echo
echo "=== qwencoin pipeline $(date '+%Y-%m-%d %H:%M:%S') in $ROOT"
STAGES=1

# ------------------------------------------------------------------ the machine
{
  if [ "$(uname -s)" = Darwin ]; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(sysctl -n hw.ncpu 2>/dev/null) threads"
    echo "memory: $(ram_gb) GB"
    echo "disk free here: $(free_gb)G"
    echo "os: macOS $(sw_vers -productVersion 2>/dev/null), $(uname -m), kernel $(uname -r)"
  else
    echo "cpu: $(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ //'), $(nproc) threads"
    echo "cpu flags: $(grep -o -w -E 'avx2|avx512f|avx512_bf16|amx_bf16' /proc/cpuinfo | sort -u | tr '\n' ' ')"
    echo "memory: $(ram_gb) GB$(wsl && echo ' (what WSL gives this distro; raise it with memory= under [wsl2] in %UserProfile%\.wslconfig, then wsl --shutdown)')"
    echo "disk free here: $(free_gb)G"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
    echo "os: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}"), glibc $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}'), kernel $(uname -r)"
    wsl && echo "wsl: $(wslinfo --wsl-version 2>/dev/null || echo '?'), networking $(wslinfo --networking-mode 2>/dev/null || echo '?'), swap $(awk '/SwapTotal/ {printf "%d GB", $2 / 1048576}' /proc/meminfo)"
  fi
} | tee "$LOGS/machine.txt"
diag() {  # what the nodes themselves said, where the Windows side can read it
  { [ -f run/n1/debug.log ] && { echo "--- run/n1/debug.log, from the start"; head -60 run/n1/debug.log; }
    for d in run/n*; do [ -f "$d/debug.log" ] && { echo "--- $d/debug.log"; tail -40 "$d/debug.log"; }; [ -s "$d/start.log" ] && { echo "--- $d/start.log"; tail -5 "$d/start.log"; }; done
    for f in run/*.log run/ord*/server.log; do [ -s "$f" ] && { echo "--- $f"; tail -8 "$f"; }; done; } > "$LOGS/diag.txt" 2>&1
}

# ------------------------------------------------------------------ env: Python and packages
env_stage
if [ -f .done/no-torch ]; then
  [ -n "$JOIN" ] || fail "torch is not installed here, and merging the models needs it"
  "$PY" -c "import sys, numpy; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| no torch: the node runs without a model')" || fail "imports"
else
  "$PY" -c "import sys, torch, transformers; print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" || fail "imports"
fi

fetch() {  # fetch SIZE: the stock checkpoint from Hugging Face into base/
  done_p "model-$1" && return 0
  echo "== download Qwen/Qwen3.5-$1"
  "$PY" tools/fetch.py "Qwen/Qwen3.5-$1" "base/Qwen3.5-$1" &
  local FP=$!
  while kill -0 $FP 2>/dev/null; do sleep 30; kill -0 $FP 2>/dev/null && echo "   $(du -sh "base/Qwen3.5-$1" 2>/dev/null | cut -f1) so far"; done
  wait $FP || return 1
  mark "model-$1"
}

# ------------------------------------------------------------------ the chain's work table: the weights every proof of work passes through
table_here
if [ -f dist/qwork.bin ] && ! done_p table && "$PY" -m hashformer.work info dist/qwork.bin 2>/dev/null | grep -q "(the chain's table)"; then
  echo "== the work table: the one in this folder is the chain's"
  mark table
fi
if [ ! -f dist/qwork.bin ] || ! done_p table; then
  fetch 0.8B || fail "download of Qwen3.5-0.8B (the chain's proof of work is a pass through its weights)"
  echo "== the work table: the 72 MLP matrices of Qwen3.5-0.8B at 16-bit fixed point"
  "$PY" -m hashformer.work build base/Qwen3.5-0.8B dist/qwork.bin || fail "the table built here is not the chain's (see above)"
  mark table
fi

"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOGS/machine.txt"
import os, time
from hashformer import executor as X, work as W
t = W.CTable("dist/qwork.bin"); h0 = bytes(range(32))
t0 = time.time(); [t.evaluate(h0, n) for n in range(8)]; a = (time.time() - t0) / 8
t0 = time.time(); [X.sha256(bytes([n]) * 32) for n in range(200)]; b = (time.time() - t0) / 200
t0 = time.time()
for n in range(20):
    with open("work/.fsync-probe", "wb") as f:
        f.write(b"x" * 4096); f.flush(); os.fsync(f.fileno())
c = (time.time() - t0) / 20
os.remove("work/.fsync-probe")
print(f"speed: one pass through the work table {1000 * a:.0f} ms; one SHA-256 in the circuit's executor {1000 * b:.1f} ms; one fsync {1000 * c:.1f} ms")
PYEOF

# ------------------------------------------------------------------ the network: up first, it runs while the rest goes on
chain() { "$PY" tools/chain.py "$@" 2>&1 | tee -a "$LOGS/chain.log"; return "${PIPESTATUS[0]}"; }
if [ -z "${NO_NET:-}" ]; then
  echo "== the network: ${NODES:-4} nodes (qwencoind + ord + miner + agent each), from the chain in run/ or from genesis"
  chmod +x bin/* libexec/* lib/ld-linux-*.so.* 2>/dev/null
  table_here
  V=$(bin/qwencoind -version 2>&1 | head -1); case "$V" in *ersion*) echo "   $V" ;; *) fail "the node does not run on this machine ($PLAT): $V. This copy has node programs for: linux-x86_64 $(ls node 2>/dev/null | tr '\n' ' ')" ;; esac
  # New tooling in this folder (the project downloaded again, the kit updated): what still runs from the old files stops
  # here and is started again below from the new ones. The nodes themselves keep running.
  if "$PY" tools/chain.py stale-helpers >/dev/null 2>&1; then
    echo "   new node tooling: the miners, agents, ord servers and node models start again from it (the nodes keep running)"
    "$PY" tools/chain.py stop-helpers 2>&1 | tail -1
  fi
  # a node's model is started only from a checkpoint that has passed its checks here
  MINDS=$([ -n "$(largest)" ] && echo "${MINDS:-1}" || echo 0) chain up "${NODES:-4}" || { diag; fail "the nodes did not start (see $LOGS/chain.log and $LOGS/diag.txt)"; }
  chain status
  diag
fi
minds_gate() {  # before anything is published: every node has its own model loaded, and each has spoken on the graph
  [ -n "${NO_NET:-}" ] && return 0
  done_p minds && return 0
  [ -f .done/merged-0.8B ] || { echo "   the nodes' models need the merged 0.8B, which has not passed here"; return 1; }
  echo "== the nodes' models: one per node, each taking its turns in the graph's conversations (the gate: every one has published)"
  chain up "${NODES:-4}" || return 1
  chain minds "${NODES:-4}" || { echo "   not every node's model has spoken yet (run/mind*.log has what each did)"; return 1; }
  mark minds
}

# ------------------------------------------------------------------ the hasher alone, and which kernels run it exactly
if [ -f .done/no-torch ]; then
  echo "== the node's model is NOT started: there is no torch on this machine ($PLAT). The node runs without it."
  date '+%Y-%m-%d %H:%M:%S' > "$LOGS/DONE"
  chain status
  echo "=== joined $(cat "$LOGS/DONE"), without a model.  bash tools/pipeline.sh status | down"
  echo "== following the chain (Ctrl-C stops following, not the node)"
  exec "$PY" tools/chain.py watch
fi
if ! done_p hasher; then
  echo "== the stand-alone hasher: compiled here, compared with the published build"
  "$PY" -m hashformer.hf --out dist/qwensha 2>&1 | filter; [ "${PIPESTATUS[0]}" = 0 ] || fail "hasher build"
  GOT=$(sha256_of dist/qwensha/model.safetensors)
  if [ "$GOT" = "$HASHER_SHA" ]; then echo "   model.safetensors sha256 $GOT: bit for bit the build the node's executor was generated from"
  else echo "   model.safetensors sha256 $GOT differs from $HASHER_SHA (other library versions?); the checks below still decide"; fi
  mark hasher
fi
if ! done_p probe; then
  echo "== probe: which kernels on this machine run the chip exactly"
  "$PY" tools/probe_backend.py "$LOGS/backend.json" 2>&1 | filter; [ "${PIPESTATUS[0]}" = 0 ] || fail "backend probe"
  mark probe
fi
if [ -z "$JOIN" ] && ! done_p bridge-ir; then
  echo "== the bridge circuit against its reference, in exact arithmetic (no model weights)"
  "$PY" tests/check_bridge.py base/Qwen3.5-0.8B 2>&1 | filter | tee "$LOGS/bridge-ir.log"; [ "${PIPESTATUS[0]}" = 0 ] || fail "bridge circuit"
  mark bridge-ir
fi

# ------------------------------------------------------------------ join: the published checkpoint as this node's model
if [ -n "$JOIN" ]; then
  M=${JOIN_MODEL:-0.8B}; OUT=$(dir_of "$M")
  REPO=${QWEN_MODEL_REPO:-$(head -1 ORIGIN 2>/dev/null | tr -d ' \r\n')/Qwen3.5-$M-qwencoin}
  case "$REPO" in /*) fail "which account published the checkpoints is not known here (no file ORIGIN): QWEN_MODEL_REPO=ACCOUNT/Qwen3.5-$M-qwencoin" ;; esac
  if [ ! -f "$OUT/hashformer.json" ] || ! done_p "fetched-$M"; then
    echo "== the published checkpoint https://huggingface.co/$REPO (nothing is merged here)"
    "$PY" tools/fetch.py "$REPO" "$OUT" || fail "download of $REPO"
    mark "fetched-$M"
  fi
  PICK=${BACKEND:-$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$(gb_of "$OUT")" 2>/dev/null)} || PICK=""
  if [ -z "$PICK" ]; then  # a node without a model is a node all the same
    echo "== the node's model is NOT started: this machine has $(ram_gb) GB of memory, and the merged $M needs about $(awk "BEGIN {print int($(gb_of "$OUT") * 1.08 + 4.99)}") GB to run in a kernel that computes the hash exactly. The node runs without it."
    wsl && echo "   (that is what WSL gives this distro: raise it with memory= under [wsl2] in %UserProfile%\.wslconfig, then wsl --shutdown, then join again)"
    echo "   (BACKEND='cpu bfloat16' bash tools/pipeline.sh join   tries it all the same)"
  else
    if ! done_p "joined-$M"; then
      echo "== that checkpoint on this machine (${PICK% *} as ${PICK#* }): a hash asked for in plain words, computed by its weights, against hashlib"
      DEVICE=${PICK% *} DTYPE=${PICK#* } "$PY" tests/check_published.py "$OUT" 2>&1 | filter | tee "$LOGS/published-$M.log"
      [ "${PIPESTATUS[0]}" = 0 ] || fail "the published checkpoint did not compute the hash here (see $LOGS/published-$M.log); the node runs on without a model"
      mark "joined-$M"
    fi
    if grep -q "^MIND_MODEL=" local.env 2>/dev/null && [ -f "${MIND_MODEL:-/nonexistent}/hashformer.json" ]; then
      :  # this folder's node was given another size since (`minds SIZE`): that one stays, in the kernel chosen for it
    else
      # the size and the kernel picked for this machine are this folder's from now on: a later start (up, refresh) runs the same
      export DEVICE=${PICK% *} DTYPE=${PICK#* } MIND_MODEL="$ROOT/$OUT"
      { grep -v -E "^(DEVICE|DTYPE|MIND_MODEL)=" local.env 2>/dev/null; echo "DEVICE=$DEVICE"; echo "DTYPE=$DTYPE"; echo "MIND_MODEL='$MIND_MODEL'"; } > local.env.new && mv local.env.new local.env
    fi
    echo "== the node's model: ${MIND_MODEL#"$ROOT"/} (${DEVICE:-cpu} as ${DTYPE:-bfloat16}), talking with the other nodes' models on the graph"
    chain up "${NODES:-1}" || fail "the node's model did not start"
  fi
  date '+%Y-%m-%d %H:%M:%S' > "$LOGS/DONE"
  chain status
  if [ -n "$PICK" ]; then echo "=== joined $(cat "$LOGS/DONE"). Talk to the model:  bash tools/pipeline.sh ask \"How many coins does this node have?\""
  else echo "=== joined $(cat "$LOGS/DONE"), without a model.  bash tools/pipeline.sh status | down"; fi
  echo "== following the chain (Ctrl-C stops following, not the node)"
  exec "$PY" tools/chain.py watch
fi

publish() {  # publish WHAT...: never a reason to stop; without a token it says so and goes on
  [ -n "${NO_PUBLISH:-}" ] && return 0
  "$PY" tools/publish.py "$@" 2>&1 | tee -a "$LOGS/publish.log"
  return 0
}

# ------------------------------------------------------------------ per size, smallest first: merge, check, tools, publish
resync() {  # files replaced in the kit's source folder since this run began (never this script: bash is reading it)
  [ -n "${KIT_SRC:-}" ] && [ -d "$KIT_SRC" ] || return 0
  [ -f .done/synced ] || touch -d '2020-01-01' .done/synced
  ( cd "$KIT_SRC" && find . -type f ! -path ./tools/pipeline.sh ! -path './libexec/*' ! -path './lib/*' ! -path './bin/*' -newer "$ROOT/.done/synced" -print0 \
      | while IFS= read -r -d '' f; do mkdir -p "$ROOT/$(dirname "$f")" && cp "$f" "$ROOT/$f" && echo "   updated from the kit folder: $f"; done )
  touch .done/synced
}
touch .done/synced
BAD=""; PAUSED=""
for M in $MODELS; do
  resync
  OUT=$(dir_of "$M"); [ -n "$OUT" ] || { echo "unknown model size $M"; BAD="$BAD $M"; continue; }
  if ! done_p "merged-$M"; then
    case $M in  # merged size in bf16 (GB) and cost per token relative to the stand-alone circuit
      0.8B) SIZE=4.1; SCALE=7 ;; 2B) SIZE=8.5; SCALE=13 ;; 4B) SIZE=15.3; SCALE=23 ;; 9B) SIZE=33; SCALE=50 ;; 27B) SIZE=83; SCALE=125 ;;
    esac
    if [ -n "${BACKEND:-}" ]; then PICK="$BACKEND"; else
      PICK=$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$SIZE") || {
        echo "== Qwen3.5-$M: SKIPPED, its merged checkpoint is $SIZE GB and this machine gives $(ram_gb) GB (WSL: raise memory= in .wslconfig)"; BAD="$BAD $M"; continue; }
    fi
    FREE=$(free_gb)
    [ "$FREE" -gt "$(awk "BEGIN {print int($SIZE * 1.9 + 10)}")" ] || { echo "== Qwen3.5-$M: SKIPPED, needs about $(awk "BEGIN {print int($SIZE * 1.9 + 10)}") GB of disk, $FREE GB free"; BAD="$BAD $M"; continue; }
    fetch "$M" || { echo "== Qwen3.5-$M: the download failed"; BAD="$BAD $M"; continue; }
    export DEVICE=${PICK% *} DTYPE=${PICK#* }
    if [ -z "${NO_NET:-}" ] && [ "$(awk "BEGIN {print int($SIZE * 1.08 + 4 + 6 * ${NODES:-4})}")" -gt "$(ram_gb)" ]; then
      echo "== Qwen3.5-$M needs the memory the nodes' models hold: they pause until this size is done (nodes, miners and agents go on)"
      chain pause-minds; PAUSED=1
    fi
    # the probe timed the 328M-parameter circuit; a merged model costs SCALE times that per token
    SLOW=$("$PY" -c "import json; r=[x for x in json.load(open('$LOGS/backend.json'))['runs'] if x['backend']=='$DEVICE/$DTYPE']; print(int(bool(r) and r[0]['free_running']['ms_per_token']*$SCALE > 4000))" 2>/dev/null || echo 0)
    Q="${QUICK:-}"; [ "$SLOW" = 1 ] && [ -z "${FULL:-}" ] && Q=1
    {
      echo "== Qwen3.5-$M: merge, then the checks on $DEVICE as $DTYPE${Q:+ (the short list)}"
      if [ ! -f "$OUT/hashformer.json" ] || [ -n "${REMERGE:-}" ]; then
        echo "\$ python -m hashformer.merge base/Qwen3.5-$M --out $OUT --device auto"
        guarded "$PY" -m hashformer.merge "base/Qwen3.5-$M" --out "$OUT" --device auto 2>&1 | filter; [ "${PIPESTATUS[0]}" = 0 ] || exit 3
      fi
      echo; echo "\$ python tests/check_merge.py base base/Qwen3.5-$M work/$M"
      QUICK=$Q guarded "$PY" tests/check_merge.py base "base/Qwen3.5-$M" "work/$M" 2>&1 | filter; [ "${PIPESTATUS[0]}" = 0 ] || exit 4
      echo; echo "\$ python tests/check_merge.py merged $OUT work/$M"
      QUICK=$Q guarded "$PY" tests/check_merge.py merged "$OUT" "work/$M" 2>&1 | filter; [ "${PIPESTATUS[0]}" = 0 ] || exit 5
    } 2>&1 | tee "$LOGS/merge-$M.log"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "FAILED: Qwen3.5-$M did not pass (see $LOGS/merge-$M.log); it is not published; going on with the next size"; BAD="$BAD $M"
                                    [ -n "$PAUSED" ] && { chain up "${NODES:-4}"; PAUSED=""; }; continue; }
    mark "merged-$M"
  fi
  if [ -z "${NO_TOOLS:-}" ] && [ -z "${NO_NET:-}" ] && ! done_p "tools-$M"; then
    # the model driving node 1 of the running chain: what it does there is real
    echo "== Qwen3.5-$M: its functions on the running chain, as node 1 (its payment, its inscription and its post are real)"
    until [ "$("$PY" tools/net.py cli 1 getblockcount 2>/dev/null | tr -dc 0-9)" -gt 101 ] 2>/dev/null; do echo "   waiting for the chain to pass height 101 (a node's first coins mature then)"; sleep 30; done
    PICK=${BACKEND:-$("$PY" tools/pick.py "$LOGS/backend.json" "$(ram_gb)" "$(vram_gb)" "$(gb_of "$OUT")")}
    DEVICE=${PICK% *} DTYPE=${PICK#* } guarded "$PY" tests/check_tools.py live "$OUT" 2>&1 | filter | tee "$LOGS/tools-$M.log"
    if [ "${PIPESTATUS[0]}" = 0 ]; then mark "tools-$M"
    else echo "FAILED: Qwen3.5-$M with its functions (see $LOGS/tools-$M.log); it is not published; going on"; BAD="$BAD $M"; [ -n "$PAUSED" ] && chain up "${NODES:-4}"; continue; fi
  fi
  [ -n "$PAUSED" ] && { chain up "${NODES:-4}"; PAUSED=""; }  # the nodes' models again
  resync
  if minds_gate; then
    done_p hub || { publish hub "$LOGS" && mark hub; }  # the project page: code, node binaries, the table, the stand-alone hasher
    done_p "published-$M" || publish model "$M" "$OUT" "$LOGS"
  else
    echo "== Qwen3.5-$M passed and is NOT published yet: the nodes' models are not all up and talking (bash tools/pipeline.sh publish, later)"
  fi
done

done_p minds && publish hub "$LOGS"  # again: its page lists what is published now
date '+%Y-%m-%d %H:%M:%S' > "$LOGS/DONE"
echo "=== all stages done $(cat "$LOGS/DONE")${BAD:+; not done:$BAD}"
for M in $MODELS; do grep -h -E "printed digest|answer repeats|^OK" "$LOGS/merge-$M.log" 2>/dev/null | sed "s/^/    Qwen3.5-$M: /"; done
[ -s "$LOGS/published.txt" ] && { echo "    on Hugging Face:"; sed 's/^/      /' "$LOGS/published.txt"; }
echo "    talk to it:  bash tools/pipeline.sh ask \"How many coins does this node have?\""
[ -n "$BAD" ] && echo "not done:$BAD" > "$LOGS/FAILED"
if [ -z "${NO_NET:-}" ]; then
  chain status
  echo "== following the chain (Ctrl-C stops following, not the nodes)"
  exec "$PY" tools/chain.py watch
fi
