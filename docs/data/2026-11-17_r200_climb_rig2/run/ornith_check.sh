#!/bin/bash
# R200: the profile checked on every card's whole memory. Ornith-1.5-122B Pascal N2P GGUF on all nine cards with its v4
# release settings (release/ornith-v4/settings.env, the 49,152-token context: about 5.5 GB on each card), and the kit's
# probability-hashed greedy check (greedy_check.py: two prompts, 256 tokens each, temperature 0, every token's top-10
# probabilities hashed) against the server, ROUNDS times.
#
#   TAG=stock   ROUNDS=2  bash ornith_check.sh    # on a fresh boot, every offset 0: the reference (round 2 = round 1?)
#   TAG=profile ROUNDS=10 bash ornith_check.sh    # after `climb.py apply` passed, the watchdog running
#
# Run as root in the kit folder with env.sh sourced (BIN). MODEL is the Ornith file; SETTINGS the release settings
# file; OUT the folder for the hashes (the same for both tags, on a disk that survives the reboot). TAG=profile
# compares every round with stock's round 1, writes the server's pid into each card's pids file (so a one-card trip
# stops it), and stops at the first round that differs. The server is stopped at the end; the offsets are left alone.
set -u
: "${BIN:?BIN=<engine bin folder>}" "${MODEL:?MODEL=<the Ornith GGUF>}" "${SETTINGS:?SETTINGS=<ornith-v4 settings.env>}"
TAG=${TAG:?TAG=stock|profile} ROUNDS=${ROUNDS:-2} OUT=${OUT:-/var/tmp/r200-ornith} PORT=${PORT:-8300}
WORK=${WORK:-/var/tmp/climb3} N=${N:-256}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/ornith.log"; }
. "$SETTINGS"
if [ "$TAG" = profile ] && [ ! -f "$OUT/greedy-stock1.sha" ]; then log "no stock reference in $OUT"; exit 2; fi
log "$TAG: loading $(basename "$MODEL") with: $PRESET_ENV | $PRESET_ARGS"
t0=$(date +%s)
env $PRESET_ENV CUDA_DEVICE_ORDER=PCI_BUS_ID setsid "$BIN/llama-server" -m "$MODEL" $PRESET_ARGS --host 127.0.0.1 \
  --port "$PORT" > "$OUT/server-$TAG.log" 2>&1 < /dev/null &
PID=$!
if [ "$TAG" = profile ] && [ -d "$WORK/pids" ]; then for g in 0 1 2 3 4 5 6 7 8; do echo "$PID" > "$WORK/pids/$g.pids"; done; fi
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then log "$TAG: the server exited while loading (server-$TAG.log)"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1500 ]; then log "$TAG: not ready after 1,500 s"; kill "$PID"; exit 1; fi
done
log "$TAG: ready in $(( $(date +%s) - t0 )) s; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB used per card; $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB host memory available"
rc=0
for r in $(seq 1 "$ROUNDS"); do
  if [ "$TAG" = stock ] && [ "$r" = 1 ]; then ref=""; else ref="--ref stock1"; fi
  if [ "$TAG" = profile ] && [ -s "$WORK/TRIPPED" ]; then log "$TAG: the watchdog tripped; stop"; rc=1; break; fi
  out=$(timeout 900 python3 "$HERE/greedy_check.py" --url "http://127.0.0.1:$PORT" --out "$OUT" --tag "$TAG$r" --n "$N" $ref)
  c=$?
  log "$TAG round $r: $out (exit $c)"
  if [ "$c" != 0 ]; then rc=1; break; fi
done
kill "$PID" 2>/dev/null; sleep 5; kill -9 "$PID" 2>/dev/null
if [ "$TAG" = profile ] && [ -d "$WORK/pids" ]; then for g in 0 1 2 3 4 5 6 7 8; do : > "$WORK/pids/$g.pids"; done; fi
log "$TAG: done, exit $rc"
echo "exit $rc" > "$OUT/ORNITH_${TAG}_DONE"
exit $rc
