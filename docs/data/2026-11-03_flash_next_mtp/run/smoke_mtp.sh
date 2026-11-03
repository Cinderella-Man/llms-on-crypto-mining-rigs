#!/usr/bin/env bash
# smoke_mtp.sh OUT: Q226's smoke test of the draft line before its timed agent check. llama-server starts on the rig with
# release_check.sh's COMMON settings and ARGS_mtp (through bench/agent_check/rig_server.sh, so with the same monitors
# and RAM watchdog), records each card's memory once loaded, answers one short coding request (REQUEST, a chat
# completion body; the reply's timings carry draft_n and draft_n_accepted), then stops; the rig's logs come back to OUT
# and are removed on the rig. Runs only ssh here. Environment: RIG, BENCH (bench/agent_check), REL_BIN, MODEL,
# RIG_PORT, ARGS_mtp, SMOKE_LOGDIR (a folder on the rig), REQUEST (a file here), optional CTX (65536), MIN_AVAIL_MB
# (400).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: smoke_mtp.sh OUT}")
: "${RIG:?}" "${BENCH:?}" "${REL_BIN:?}" "${MODEL:?}" "${RIG_PORT:?}" "${ARGS_mtp:?}" "${SMOKE_LOGDIR:?}" "${REQUEST:?}"
CTX=${CTX:-65536}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/smoke.log"; }
rs() { ssh "$RIG" "LOGDIR=$SMOKE_LOGDIR MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} PORT=$RIG_PORT $1 bash -s" < "$BENCH/rig_server.sh"; }

say "starting: $REL_BIN $COMMON $ARGS_mtp -m $MODEL"
rs "ACTION=start ARM=mtp-smoke BIN=$REL_BIN MODEL=$MODEL SERVER_ENV='' SERVER_ARGS='$COMMON $ARGS_mtp'" 2>&1 \
  | tee -a "$OUT/smoke.log"
up=${PIPESTATUS[0]}
if [ "$up" = 0 ]; then
  ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$OUT/vram_loaded.csv"
  ssh "$RIG" "awk '/^MemAvailable/ {print int(\$2 / 1024)}' /proc/meminfo" > "$OUT/memavail_loaded_mib.txt"
  # host memory as a whole and the server's own share (resident, swapped), once loaded and after the request
  host() { ssh "$RIG" "cat /proc/meminfo; cat /proc/\$(cat $SMOKE_LOGDIR/server.pid)/status" > "$OUT/host_$1.txt"; }
  host loaded
  say "request"
  ssh "$RIG" "curl -s -m 900 127.0.0.1:$RIG_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
    < "$REQUEST" > "$OUT/reply.json"
  say "request done ($(wc -c < "$OUT/reply.json") bytes)"
  host after_request
  ssh "$RIG" "curl -s 127.0.0.1:$RIG_PORT/metrics" > "$OUT/metrics.txt"
else
  say "the server did not come up"
fi
rs "ACTION=end" 2>&1 | tee -a "$OUT/smoke.log"
ssh "$RIG" "tar -C $SMOKE_LOGDIR -cf - ." | tar -C "$OUT" -xf - && ssh "$RIG" "rm -rf $SMOKE_LOGDIR" \
  && say "rig logs in $OUT, removed on the rig"
[ "$up" = 0 ]
