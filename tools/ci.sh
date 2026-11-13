#!/usr/bin/env bash
# The node for one platform, from upstream sources and this project's patches: what the build
# workflow runs on every runner, and the same on a developer's machine (Linux or macOS).
#
#   tools/ci.sh deps      the packages the build needs (apt, or Homebrew on a Mac)
#   tools/ci.sh build     qwencoind, qwencoin-cli and ord into bin/ (tools/build.sh)
#   tools/ci.sh pack      dist/node/PLATFORM/ and dist/qwencoin-node-PLATFORM.tar.gz
#   tools/ci.sh check     the packed programs with the chain's work table: tests/check_qwork_node.py
#                         (four nodes; every work value and hash against the exact reference)
#   tools/ci.sh all
#
# PLATFORM is linux-x86_64, linux-aarch64, macos-arm64 or macos-x86_64 (uname's own words).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PLAT="$(uname -s | tr 'A-Z' 'a-z' | sed 's/darwin/macos/')-$(uname -m)"
TABLE_URL="${QWORK_URL:-https://huggingface.co/$(head -1 ORIGIN 2>/dev/null || echo staccs)/qwencoin/resolve/main/qwork.bin}"

deps() {
  if [ "$(uname -s)" = Darwin ]; then
    for f in cmake ninja pkgconf; do brew list "$f" >/dev/null 2>&1 || brew install "$f"; done
    python3 -c "import numpy" 2>/dev/null || python3 -m pip install --user --break-system-packages numpy 2>/dev/null || python3 -m pip install --user numpy
    xcrun --show-sdk-path; clang --version | head -1
  else
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build pkg-config git curl ca-certificates \
      libboost-dev libevent-dev libsqlite3-dev python3 python3-numpy
  fi
  command -v cargo >/dev/null 2>&1 || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; . "$HOME/.cargo/env"; }
  command -v rustup >/dev/null 2>&1 && { rustup update stable >/dev/null 2>&1 || true; }
  cargo --version
  echo "$PLAT: $(uname -srm), $( (nproc 2>/dev/null || sysctl -n hw.ncpu) ) cpus, python $(python3 -V 2>&1 | cut -d' ' -f2), numpy $(python3 -c 'import numpy; print(numpy.__version__)')"
}

build() {
  bash tools/build.sh node
  bash tools/build.sh ord
  bin/qwencoind -version | head -1
  bin/ord --version
}

pack() {
  rm -rf "dist/node/$PLAT"
  mkdir -p "dist/node/$PLAT"
  python3 tools/pack.py "dist/node/$PLAT"
  echo "$PLAT" > "dist/node/$PLAT/PLATFORM"
  tar -C dist/node -czf "dist/qwencoin-node-$PLAT.tar.gz" "$PLAT"
  ls -la "dist/qwencoin-node-$PLAT.tar.gz"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "dist/qwencoin-node-$PLAT.tar.gz"; else shasum -a 256 "dist/qwencoin-node-$PLAT.tar.gz"; fi
}

check() {
  mkdir -p dist logs
  if [ ! -f dist/qwork.bin ]; then
    echo "the chain's work table (505 MB) from $TABLE_URL"
    curl -fsSL -o dist/qwork.bin.part "$TABLE_URL" && mv dist/qwork.bin.part dist/qwork.bin
  fi
  # the packed programs and libraries, not the ones in the build tree
  cp "dist/node/$PLAT/exec/"*.so exec/
  PYTHONPATH=. python3 -c "
import hashlib, os
from graph.modelhash import sha256
for n in (0, 3, 55, 56, 64, 119, 120, 1000):
    x = os.urandom(n)
    assert sha256(x) == hashlib.sha256(x).digest(), n
print('the packed executor (exec/libqsha.so): SHA-256 of 8 random strings of 0 to 1000 bytes equals hashlib')"
  QWEN_BIN="$ROOT/dist/node/$PLAT/bin" python3 tests/check_qwork_node.py "${BLOCKS:-8}"
}

case "${1:-all}" in
  deps) deps ;;
  build) build ;;
  pack) pack ;;
  check) check ;;
  all) deps; build; pack; check ;;
  *) echo "usage: $0 [deps|build|pack|check|all]"; exit 1 ;;
esac
