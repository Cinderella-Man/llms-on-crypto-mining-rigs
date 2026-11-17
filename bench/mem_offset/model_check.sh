#!/bin/bash
# model_check.sh: a clock-offset profile checked on every card's whole memory. The climb's check model covers about
# 2 GB of a 6 GB card; this serves the model the rig really runs, on all cards with its release settings, and repeats
# the kit's greedy check (greedy_check.py: two prompts, 256 tokens each, temperature 0, every token's top-10
# probabilities hashed) against it. The output is the same token for token at stock and with the offsets, so each
# round's generation tok/s also shows what the offsets buy on this model.
#
#   TAG=stock   ROUNDS=2   bash model_check.sh    # on a fresh boot, every offset 0: the reference (round 2 = round 1?)
#   TAG=profile MINUTES=30 bash model_check.sh    # after `climb.py apply` and a soak passed, the watchdog running
#
# Run as root in the kit folder. BIN is the engine's bin folder, MODEL the GGUF, SETTINGS a release settings file
# (PRESET_ENV and PRESET_ARGS, e.g. release/ornith-v4/settings.env), OUT the folder for the hashes (the same for both
# tags, on a disk that survives a reboot), WORK the soak's work folder. Rounds run until ROUNDS are done and MINUTES
# have passed. TAG=profile compares every round with stock's round 1, writes the server's pid into each card's pids
# file (so a one-card trip of the watchdog stops it), and stops at the first round that differs. The server is
# stopped at the end; the offsets are left alone. Exit 0: every round the same; 1: a difference or a failure; 2: no
# stock reference.
set -u
: "${BIN:?BIN=<engine bin folder>}" "${MODEL:?MODEL=<the GGUF>}" "${SETTINGS:?SETTINGS=<release settings.env>}"
TAG=${TAG:?TAG=stock|profile} ROUNDS=${ROUNDS:-2} MINUTES=${MINUTES:-0} OUT=${OUT:-/var/tmp/model-check}
PORT=${PORT:-8300} WORK=${WORK:-/var/tmp/climb} N=${N:-256} CARDS=${CARDS:-0 1 2 3 4 5 6 7 8}
LOAD_LIMIT=${LOAD_LIMIT:-1500} GREEDY=${GREEDY:-$(cd "$(dirname "$0")" && pwd)/greedy_check.py}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/model.log"; }
med() { sort -n | awk '{v[NR]=$1} END{if (NR) print (NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2)}'; }
. "$SETTINGS"
if [ "$TAG" = profile ] && [ ! -f "$OUT/greedy-stock1.sha" ]; then log "no stock reference in $OUT"; exit 2; fi
rm -f "$OUT/MODEL_${TAG}_DONE" "$OUT/gen-$TAG.txt"
log "$TAG: loading $(basename "$MODEL") with: $PRESET_ENV | $PRESET_ARGS"
t0=$(date +%s)
env $PRESET_ENV CUDA_DEVICE_ORDER=PCI_BUS_ID setsid "$BIN/llama-server" -m "$MODEL" $PRESET_ARGS --host 127.0.0.1 \
  --port "$PORT" > "$OUT/server-$TAG.log" 2>&1 < /dev/null &
PID=$!
if [ "$TAG" = profile ] && [ -d "$WORK/pids" ]; then for g in $CARDS; do echo "$PID" > "$WORK/pids/$g.pids"; done; fi
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then log "$TAG: the server exited while loading (server-$TAG.log)"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt "$LOAD_LIMIT" ]; then log "$TAG: not ready after $LOAD_LIMIT s"; kill "$PID"; exit 1; fi
done
log "$TAG: ready in $(( $(date +%s) - t0 )) s; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB used per card; $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB host memory available"
rc=0 r=0 t1=$(date +%s)
while [ "$r" -lt "$ROUNDS" ] || [ $(( $(date +%s) - t1 )) -lt "$(awk -v m="$MINUTES" 'BEGIN{print int(m * 60)}')" ]; do
  r=$((r + 1))
  if [ "$TAG" = stock ] && [ "$r" = 1 ]; then ref=""; else ref="--ref stock1"; fi
  if [ "$TAG" = profile ] && [ -s "$WORK/TRIPPED" ]; then log "$TAG: the watchdog tripped; stop"; rc=1; break; fi
  out=$(timeout 900 python3 "$GREEDY" --url "http://127.0.0.1:$PORT" --out "$OUT" --tag "$TAG$r" --n "$N" $ref)
  c=$?
  log "$TAG round $r: $out (exit $c)"
  if [ "$c" != 0 ]; then rc=1; break; fi
  echo "$out" | sed -n 's/.*generation tok\/s \([0-9.]*\) \([0-9.]*\).*/\1 \2/p' >> "$OUT/gen-$TAG.txt"
  if [ "$r" -gt 1 ]; then rm -f "$OUT/greedy-$TAG$r.json"; fi   # a round that equals the reference is its hash
done
kill "$PID" 2>/dev/null; sleep 5; kill -9 "$PID" 2>/dev/null
if [ "$TAG" = profile ] && [ -d "$WORK/pids" ]; then for g in $CARDS; do : > "$WORK/pids/$g.pids"; done; fi
if [ -s "$OUT/gen-$TAG.txt" ]; then
  log "$TAG: generation tok/s, median of $(wc -l < "$OUT/gen-$TAG.txt") rounds: prompt 1 $(cut -d' ' -f1 "$OUT/gen-$TAG.txt" | med), prompt 2 $(cut -d' ' -f2 "$OUT/gen-$TAG.txt" | med)"
fi
log "$TAG: done, $r rounds in $(( $(date +%s) - t1 )) s, exit $rc"
echo "exit $rc" > "$OUT/MODEL_${TAG}_DONE"
exit $rc
