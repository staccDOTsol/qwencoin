#!/bin/bash
# Runs inside WSL, started by go.cmd from the Windows folder that holds the kit. The work happens
# on WSL's own disk (~/qwencoin: 15 GB files on /mnt/c would crawl); the logs are written back to
# the Windows folder.
set -u
WIN="$(pwd)"
[ -d "$WIN/kit" ] || { echo "run go.cmd from the folder that holds kit/"; exit 1; }
# go.cmd update [COMMAND]: the kit's code again from the project's repository on GitHub, then COMMAND (refresh if none:
# the miners, agents, ord servers and node models start again from it). The node programs, MODELS and SEEDS stay as
# they are here. (One block, read whole before it runs: this file is among what it replaces.)
if [ "${1:-}" = update ]; then
  REPO="$(head -1 "$WIN/kit/GITHUB" 2>/dev/null | tr -d ' \r\n')"; REPO="${REPO:-staccDOTsol/qwencoin}"
  echo "the kit's code from https://github.com/$REPO"
  T="$(mktemp -d)"
  curl -fsSL "https://github.com/$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$T" --strip-components=1 || { echo "the download did not arrive; the kit is as it was"; exit 1; }
  [ -f "$T/tools/pipeline.sh" ] && [ -f "$T/win/go.sh" ] || { echo "what arrived is not the project; the kit is as it was"; exit 1; }
  for d in tools graph rails hashformer tests exec patches rust vendor win; do
    [ -d "$T/$d" ] && mkdir -p "$WIN/kit/$d" && cp -r "$T/$d/." "$WIN/kit/$d/"
  done
  cp "$T/GITHUB" "$WIN/kit/GITHUB"; cp "$T/BUILD.md" "$WIN/kit/README.md"; cp "$T/RESULTS.md" "$WIN/kit/RESULTS.md"
  cp "$T/win/go.sh" "$WIN/kit/go.sh.new" && mv "$WIN/kit/go.sh.new" "$WIN/kit/go.sh"
  echo "kit updated ($(cd "$T" && find tools graph rails hashformer tests exec -type f | wc -l) files of code)"
  rm -rf "$T"
  shift
  [ $# -gt 0 ] || set -- refresh
  exec bash "$WIN/kit/go.sh" "$@"
fi
# go.cmd follow: stay here and follow the project's repository. Whenever its main branch has a commit this folder has
# not deployed, the kit's code is taken again, the project is published and the public node told (seed), and what runs
# here from files that changed starts again (refresh; the nodes keep running). Unattended: whatever is pushed to that
# repository runs on this machine from then on, until this window is closed or Ctrl-C is pressed. Nothing is rented
# or terminated unattended: `seed` then only goes on with a pod the account already has.
if [ "${1:-}" = follow ]; then
  REPO="$(head -1 "$WIN/kit/GITHUB" 2>/dev/null | tr -d ' \r\n')"; REPO="${REPO:-staccDOTsol/qwencoin}"
  DEST="${QWENCOIN_HOME:-$HOME/qwencoin}"
  echo "following https://github.com/$REPO (main): every new commit is deployed here, unattended. Ctrl-C ends it."
  while true; do
    sha="$(curl -fsSL -m 30 -H 'Accept: application/vnd.github.sha' "https://api.github.com/repos/$REPO/commits/main" 2>/dev/null | tr -dc '0-9a-f' | head -c 40)"
    if [ "${#sha}" = 40 ] && [ "$sha" != "$(cat "$WIN/kit/.followed" 2>/dev/null)" ]; then
      echo "== $(date '+%Y-%m-%d %H:%M:%S') commit ${sha:0:7}: taking the kit's code, publishing it, telling the public node"
      # SEED_RENT=0: goes on with the pod there is, rents none; SEED_KEEP=1: a pod that does not answer is left as it is
      if SEED_RENT=0 SEED_KEEP=1 bash "$WIN/kit/go.sh" update seed; then
        if ( cd "$DEST" && python3 tools/chain.py stale-helpers ) >/dev/null 2>&1; then
          echo "== what runs here was started from other files: the miners, agents, ord servers and node models start again"
          bash "$WIN/kit/go.sh" refresh
        else
          echo "== nothing that runs here changed: the nodes' helpers go on as they are"
        fi
        echo "$sha" > "$WIN/kit/.followed"
        echo "== $(date '+%H:%M:%S') commit ${sha:0:7} is deployed; looking again every two minutes"
        exec bash "$WIN/kit/go.sh" follow   # this file is among what was replaced: go on as the new one
      fi
      echo "== that did not go through; again in two minutes"
    fi
    # every quarter of an hour: what the nodes' models here did lately, into the project's repository (logs/minds.txt:
    # their answers, what they were offered and declined, what went wrong). FOLLOW_LOGS=0: not.
    now=$(date +%s)
    if [ "${FOLLOW_LOGS:-900}" != 0 ] && [ $((now - ${logged:-0})) -ge "${FOLLOW_LOGS:-900}" ]; then
      logged=$now
      ( cd "$DEST" && LOGS="$WIN/logs" bash tools/pipeline.sh minds-log 2>&1 | grep -v "No files have been modified" | tail -1 )
    fi
    sleep 120
  done
fi
DEST="${QWENCOIN_HOME:-$HOME/qwencoin}"
mkdir -p "$DEST" "$WIN/logs" || exit 1
# New node binaries in the kit: stop what runs from the old ones first (the chain stays on disk;
# the pipeline starts everything again). A program that is running cannot be overwritten in place,
# so every file is replaced, not rewritten.
changed=""; neword=""
for b in qwencoind qwencoin-cli; do
  [ -f "$DEST/libexec/$b" ] && ! cmp -s "$WIN/kit/libexec/$b" "$DEST/libexec/$b" && changed=1
done
[ -f "$DEST/libexec/ord" ] && ! cmp -s "$WIN/kit/libexec/ord" "$DEST/libexec/ord" && neword=1
if [ -n "$changed" ]; then
  echo "new node binaries: stopping the running nodes (their chain and wallets stay in $DEST/run)"
  ( cd "$DEST" && [ -f tools/chain.py ] && python3 tools/chain.py down >/dev/null 2>&1 )
  pkill -f "$DEST/libexec/" 2>/dev/null
  sleep 3
elif [ -n "$neword" ]; then  # ord alone: its servers and what uses them stop and start again; the nodes keep running
  echo "new ord: its servers, the miners, agents and node models stop and start again from it (the nodes keep running)"
  ( cd "$DEST" && [ -f tools/chain.py ] && python3 tools/chain.py stop-helpers 2>&1 | tail -1 )
fi
cp -r --remove-destination "$WIN/kit/." "$DEST/" || exit 1
# New tooling (the python that runs the miners, agents, ord servers and node models): what is running keeps the
# files it was started from, so on a full run it is stopped here and the pipeline starts it again. The nodes go on.
if [ $# -eq 0 ] && [ -z "$changed" ] && ( cd "$DEST" && python3 tools/chain.py stale-helpers ) >/dev/null 2>&1; then
  echo "new node tooling: the miners, agents, ord servers and node models start again from it (the nodes keep running)"
  ( cd "$DEST" && python3 tools/chain.py stop-helpers 2>&1 | tail -1 )
fi
[ -f "$WIN/SEEDS" ] && cp "$WIN/SEEDS" "$DEST/SEEDS"  # nodes elsewhere to dial, one host:port per line
[ -f "$WIN/WEB" ] && cp "$WIN/WEB" "$DEST/WEB"  # where the public node serves the chain's pages
[ -f "$WIN/HF_TOKEN_FILE" ] && cp "$WIN/HF_TOKEN_FILE" "$DEST/HF_TOKEN_FILE"  # where the Hugging Face token file is (its path, not the token)
chmod +x "$DEST"/bin/* "$DEST"/libexec/* "$DEST"/lib/ld-linux-*.so.* 2>/dev/null
cd "$DEST" || exit 1
echo "kit in $DEST, logs in $WIN/logs"
if [ -n "$neword" ] && [ -z "$changed" ] && [ $# -gt 0 ] && [ "$1" != refresh ] && [ "$1" != down ]; then  # a full run and `refresh` start them themselves
  LOGS="$WIN/logs" bash tools/pipeline.sh refresh
fi
# KIT_SRC: the pipeline looks here again before each model size, so a file replaced in the
# Windows folder while it runs is used from the next size on
LOGS="$WIN/logs" KIT_SRC="$WIN/kit" exec bash tools/pipeline.sh "$@"
