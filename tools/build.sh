#!/usr/bin/env bash
# Rebuild everything from upstream sources: the weights, the executor, the node, ord.
#
#   tools/build.sh            all of it (about 30 min on 2 cores)
#   tools/build.sh weights    circuit header + executor library + stock Qwen3.5 checkpoint
#   tools/build.sh node       qwencoind / qwencoin-cli   (Bitcoin Inquisition v29.4 + patch)
#   tools/build.sh tests      the node plus its unit-test binary (for tools/unit.sh)
#   tools/build.sh ord        ord                        (ord 0.29.1 + patch)
#
# Needs: python3 with numpy, torch and transformers>=5.19 (checkpoint only), gcc, cmake, ninja,
# libboost-dev, libevent-dev, libsqlite3-dev, a Rust toolchain >= 1.95. On a Mac: the Xcode command
# line tools, cmake, ninja and pkgconf (Homebrew), Rust; the node's libraries are built by its depends/.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
what=${1:-all}
mkdir -p bin upstream

weights() {
  PYTHONPATH=. python3 -c "from hashformer import executor as X; print(X.build_lib(force=True))"
  PYTHONPATH=. python3 -m hashformer.hf --out dist/qwensha
}

node() {
  [ -d upstream/bitcoin ] || git clone --depth 1 --branch v29.4-inq \
      https://github.com/bitcoin-inquisition/bitcoin upstream/bitcoin
  cd upstream/bitcoin
  git apply --check "$ROOT/patches/bitcoin-v29.4-inq-qwencoin.patch" 2>/dev/null \
      && git apply "$ROOT/patches/bitcoin-v29.4-inq-qwencoin.patch"
  # the executor and the circuit in the node are the ones in exec/ (regenerate with: weights),
  # and so is the kernel of the chain's work function
  cp "$ROOT/exec/qsha.c" src/crypto/qsha.cpp
  cp "$ROOT/exec/qsha.h" "$ROOT/exec/qsha_circuit.h" src/crypto/
  cp "$ROOT/exec/qwork.c" src/crypto/qwork.cpp
  cp "$ROOT/exec/qwork.h" src/crypto/
  TOOLCHAIN=()
  if [ "$(uname -s)" = Darwin ]; then
    # On a Mac the libraries the node needs (libevent, SQLite, Boost) are built here by Bitcoin's own depends/ and
    # linked statically, for macOS 13 and later: the programs then need nothing but the system, on any Mac.
    make -C depends -j"$(sysctl -n hw.ncpu)" NO_QT=1 NO_QR=1 NO_ZMQ=1 NO_BDB=1 NO_USDT=1
    TOOLCHAIN=(--toolchain "$(ls depends/*/toolchain.cmake | head -1)")
  fi
  cmake -B build -G Ninja ${TOOLCHAIN[@]+"${TOOLCHAIN[@]}"} -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_TESTS=${TESTS:-OFF} \
      -DBUILD_BENCH=OFF -DBUILD_FUZZ_BINARY=OFF -DWITH_ZMQ=OFF -DENABLE_WALLET=ON -DBUILD_TX=OFF \
      -DBUILD_UTIL=OFF -DBUILD_WALLET_TOOL=OFF
  cmake --build build --target bitcoind bitcoin-cli $([ "${TESTS:-OFF}" = ON ] && echo test_bitcoin)
  cp build/bin/qwencoind build/bin/qwencoin-cli "$ROOT/bin/"
  cd "$ROOT"
}

ord() {
  [ -d upstream/ord ] || git clone --depth 1 --branch 0.29.1 https://github.com/ordinals/ord upstream/ord
  # rust-bitcoin's hashes and the libsecp256k1 that rust-secp256k1 bundles, with their SHA-256
  # routed to the model, plus qsha-sys (the executor). ord's manifest patches them in.
  mkdir -p upstream/rust
  for c in bitcoin_hashes-0.14.101 secp256k1-sys-0.10.1; do
    n=${c%-*}
    rm -rf "upstream/rust/$n" "upstream/rust/$c"
    curl -sSL "https://static.crates.io/crates/$n/$c.crate" | tar -xz -C upstream/rust
    mv "upstream/rust/$c" "upstream/rust/$n"
    (cd "upstream/rust/$n" && patch -s -p1 < "$ROOT/patches/$c-qsha.patch")
  done
  rm -rf upstream/rust/qsha-sys && cp -r rust/qsha-sys upstream/rust/qsha-sys && mkdir -p upstream/rust/qsha-sys/c
  cp exec/qsha.c exec/qsha.h exec/qsha_circuit.h upstream/rust/qsha-sys/c/
  # the patched crates against an independent SHA-256 and the BIP340 vector (3 minutes)
  rm -rf upstream/rust/qsha-check && cp -r rust/qsha-check upstream/rust/qsha-check
  (cd upstream/rust/qsha-check && cargo run --release)
  cd upstream/ord
  git apply --check "$ROOT/patches/ord-0.29.1-qwencoin.patch" 2>/dev/null \
      && git apply "$ROOT/patches/ord-0.29.1-qwencoin.patch"
  [ "$(uname -s)" = Darwin ] && export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"  # as the node: macOS 13 and later
  cargo build --release
  cp target/release/ord "$ROOT/bin/"
  cd "$ROOT"
}

case $what in
  weights) weights ;;
  node) node ;;
  tests) TESTS=ON node ;;
  ord) ord ;;
  all) weights; node; ord ;;
  *) echo "usage: $0 [all|weights|node|tests|ord]"; exit 1 ;;
esac
