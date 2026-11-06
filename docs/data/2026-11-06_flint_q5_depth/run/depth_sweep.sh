#!/usr/bin/env bash
# depth_sweep.sh: decode speed at fixed context depths on llama-server, one build against another, loads alternated.
# Per load: one llama-server on the rig, started and stopped by bench/agent_check/rig_server.sh (the agent check's own
# start, with its RAM watchdog and monitors) on the line the agent check runs, then run/depth_client.py on the rig's
# loopback (one warm-up, then per depth one chat turn of exactly that many token ids, read fresh, N_PREDICT greedy
# tokens out, twice), the server's own timings to OUT/<load>.jsonl, readable lines to OUT/<load>.txt; the server's log,
# with each request's pipelined guesses, comes back in OUT/rig/server-<load>.log.
# Run from a machine that reaches the rig over ssh, with nothing else on the rig:
#   RIG=rig MODEL=<gguf on the rig> V2_BIN=<llama-server A on the rig> V3_BIN=<llama-server B> OUT=<local folder> \
#     bash run/depth_sweep.sh
# Env: LOADS (default "v2a:V2 v3a:V3 v2b:V2 v3b:V3"), DEPTHS (default "0 8192 16384 24576 32768"), N_PREDICT (256),
# LINE (the agent check's release line for the 5-bit files, below), PORT (8091), LOGDIR (on the rig,
# /tmp/depth-sweep; copied to OUT/rig/ at the end and removed), STDLIB (/usr/lib/python3.12, the prompt text),
# RIG_SERVER (rig_server.sh, default the repository's). Prints "LOAD <name> done" per load, then DEPTH_DONE, or
# DEPTH_FAILED with the reason.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RIG=${RIG:-rig}
: "${MODEL:?}" "${V2_BIN:?}" "${V3_BIN:?}" "${OUT:?}"
LOADS=${LOADS:-v2a:V2 v3a:V3 v2b:V2 v3b:V3}
DEPTHS=${DEPTHS:-0 8192 16384 24576 32768}
N_PREDICT=${N_PREDICT:-256}
LINE=${LINE:--ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui -b 512 -ub 128 --pipe-depth 4}
PORT=${PORT:-8091}
LOGDIR=${LOGDIR:-/tmp/depth-sweep}
STDLIB=${STDLIB:-/usr/lib/python3.12}
RS=${RIG_SERVER:-$HERE/../../../../bench/agent_check/rig_server.sh}
mkdir -p "$OUT"
[ -r "$RS" ] || { echo "DEPTH_FAILED: no rig_server.sh at $RS"; exit 1; }

quiet() {  # quiet_<name>.txt: llama processes, per-card memory, temperature and clock, host memory
  { date -Is; ssh "$RIG" 'ps -eo comm | grep -x -E "llama-server|llama-bench|llama-perplexity|llama-cli" || echo "no llama process"
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader | tr "\n" " "; echo
    grep -E "MemAvailable|SwapFree" /proc/meminfo'; } > "$OUT/quiet_$1.txt" 2>&1
}
rs() {  # rs <action> [env ...]: rig_server.sh on the rig
  local a=$1; shift
  ssh "$RIG" "ACTION=$a PORT=$PORT LOGDIR=$LOGDIR $* bash -s" < "$RS"
}

echo "=== $(date -Is) depth sweep: $(basename "$MODEL"); loads $LOADS; depths $DEPTHS; $N_PREDICT tokens; line: $LINE"
quiet start
if ! grep -q "no llama process" "$OUT/quiet_start.txt"; then echo "DEPTH_FAILED: the rig is not idle"; cat "$OUT/quiet_start.txt"; exit 1; fi
fail=""
for spec in $LOADS; do
  name=${spec%%:*}
  if [ "${spec#*:}" = V2 ]; then bin=$V2_BIN; else bin=$V3_BIN; fi
  [ "$name" = "${LOADS%% *}" ] || quiet "$name"
  echo "=== $(date -Is) LOAD $name: $bin"
  if ! rs start "ARM=$name BIN=$bin MODEL=$MODEL SERVER_ARGS='$LINE'"; then
    fail="$name: the server did not start"; rs stop; break
  fi
  ssh "$RIG" "python3 -I - $PORT $N_PREDICT $STDLIB ${DEPTHS// /,} $name" < "$HERE/depth_client.py" 2>&1 > "$OUT/$name.jsonl" \
    | tee "$OUT/$name.txt"
  rc=${PIPESTATUS[0]}
  rs stop
  [ "$rc" = 0 ] || { fail="$name: the client exited with $rc"; break; }
  echo "LOAD $name done $(date -Is)"
done
rs end
mkdir -p "$OUT/rig"
ssh "$RIG" "tar -C $LOGDIR -cf - ." | tar -C "$OUT/rig" -xf - && ssh "$RIG" "rm -rf $LOGDIR"
quiet end
cat "$OUT/quiet_end.txt"
if [ -n "$fail" ]; then echo "DEPTH_FAILED: $fail $(date -Is)"; exit 1; fi
echo "DEPTH_DONE $(date -Is)"
