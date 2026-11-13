#!/bin/bash
# qwencoin: one node on the public chain, with the published model as its mind.
#
#   curl -fsSL https://huggingface.co/@ACCOUNT@/qwencoin/resolve/main/install.sh | bash
#
# Linux x86-64 and arm64, macOS on Apple silicon and Intel (Windows through WSL2).
# Makes the folder ./qwencoin (or $QWENCOIN_DIR) and keeps everything inside it: its own Python
# (uv), the project, the chain, the wallet, the model. No sudo, nothing system-wide. Run again, it
# goes on where it stopped. QWEN_PORT / QWEN_ORD_PORT (default 8733 / 18080) move the node's ports,
# for a second node on one machine; they are remembered in the folder (local.env).
main() {
  set -euo pipefail
  ACCOUNT="@ACCOUNT@"
  DIR="${QWENCOIN_DIR:-$PWD/qwencoin}"
  command -v curl >/dev/null 2>&1 || { echo "qwencoin: curl is needed"; exit 1; }
  # The node programs are built per platform. Linux x86-64 is in the repository's bin/; every other platform's
  # package is built and checked by the project's workflow on GitHub and taken from its latest release.
  GITHUB="@GITHUB@"
  PLAT="$(uname -s | tr 'A-Z' 'a-z' | sed 's/darwin/macos/')-$(uname -m)"
  [ "$PLAT" = linux-arm64 ] && PLAT=linux-aarch64
  NODE_URL="https://github.com/$GITHUB/releases/latest/download/qwencoin-node-$PLAT.tar.gz"
  if [ "$PLAT" != linux-x86_64 ] && ! curl -fsL -r 0-0 -o /dev/null "$NODE_URL" 2>/dev/null; then
    echo "qwencoin: there is no node build for $PLAT (https://github.com/$GITHUB/releases has them for linux-aarch64, macos-arm64, macos-x86_64; linux-x86_64 is in the project). The project: https://huggingface.co/$ACCOUNT/qwencoin"
    exit 1
  fi
  if grep -qi microsoft /proc/version 2>/dev/null; then
    case "$DIR" in /mnt/[a-z]/*) echo "qwencoin: $DIR is on the Windows disk, where a node crawls: run this from the Linux home (cd ~)"; exit 1 ;; esac
  fi
  mkdir -p "$DIR"
  cd "$DIR"
  DIR="$(pwd)"
  if [ -d run ] && [ ! -f ORIGIN ]; then
    echo "qwencoin: $DIR holds a node that was not installed from https://huggingface.co/$ACCOUNT/qwencoin; QWENCOIN_DIR=... picks another folder"
    exit 1
  fi
  FREE=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc 0-9 || true); [ -n "$FREE" ] || FREE=$(df -g . 2>/dev/null | tail -1 | awk '{print $4}' || true)
  [ "${FREE:-99}" -ge 9 ] || echo "qwencoin: $FREE GB free here; the node, the model and their Python take about 8 GB"
  for v in QWEN_PORT QWEN_ORD_PORT; do  # ports other than the chain's own: kept for every later command in this folder
    eval "x=\${$v:-}"
    if [ -n "$x" ]; then { grep -v "^$v=" local.env 2>/dev/null || true; echo "$v=$x"; } > local.env.new && mv local.env.new local.env; fi
  done
  export UV_CACHE_DIR="$DIR/.uv/cache" UV_PYTHON_INSTALL_DIR="$DIR/.uv/python" UV_TOOL_DIR="$DIR/.uv/tools" UV_NO_CONFIG=1
  export HF_HOME="$DIR/.hf" HF_HUB_DISABLE_TELEMETRY=1
  if [ ! -x .uv/bin/uv ]; then
    echo "== qwencoin: uv (it installs Python and packages), into $DIR/.uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$DIR/.uv/bin" sh
    [ -x .uv/bin/uv ] || { echo "qwencoin: uv could not be installed (https://astral.sh/uv/install.sh)"; exit 1; }
  fi
  echo "== qwencoin: the project from https://huggingface.co/$ACCOUNT/qwencoin into $DIR (about 600 MB)"
  for k in 1 2 3 4 5; do  # a connection that drops: what has arrived is kept, the rest is asked for again
    OTHERS=(); [ "$PLAT" = linux-x86_64 ] || OTHERS=(--exclude "bin/*" --exclude "libexec/*" --exclude "lib/*" --exclude "exec/*.so")  # Linux x86-64 programs
    .uv/bin/uv tool run --python 3.12 --from huggingface_hub hf download "$ACCOUNT/qwencoin" --local-dir . --exclude "qwensha/*" --exclude "node/*" --exclude "logs/*" ${OTHERS[@]+"${OTHERS[@]}"} </dev/null >/dev/null && break
    [ "$k" = 5 ] && { echo "qwencoin: the download did not arrive; run the same command again, it goes on where it stopped"; exit 1; }
    echo "qwencoin: the download was interrupted; again in $((k * 10))s"; sleep $((k * 10))
  done
  [ -f tools/pipeline.sh ] || { echo "qwencoin: the download did not arrive"; exit 1; }
  if [ "$PLAT" != linux-x86_64 ] && [ ! -f "node/$PLAT/PLATFORM" ]; then
    echo "== qwencoin: the node programs for $PLAT from https://github.com/$GITHUB/releases"
    mkdir -p node
    curl -fsSL "$NODE_URL" -o node/package.tar.gz && tar -xzf node/package.tar.gz -C node && rm -f node/package.tar.gz \
      && [ -f "node/$PLAT/PLATFORM" ] || { echo "qwencoin: the node programs for $PLAT did not arrive ($NODE_URL); run the same command again"; exit 1; }
  fi
  exec bash tools/pipeline.sh join </dev/null
}
main "$@"
