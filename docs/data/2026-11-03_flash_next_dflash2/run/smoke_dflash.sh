#!/usr/bin/env bash
# smoke_dflash.sh OUT: Q561's smoke test of a draft line before its timed agent check (smoke_mtp.sh from Q226, with
# several requests and S15's memory guard). The guard (mem_guard.sh, GUARD) starts on the rig first, at real-time
# priority; then llama-server starts with release_check.sh's COMMON settings and ARGS (through
# bench/agent_check/rig_server.sh, so with the same monitors and RAM watchdog), each card's memory and the host's
# (/proc/meminfo, the server's /proc status) are recorded once loaded, each request in REQUESTS (chat completion bodies,
# files here, sent in order; the replies' timings carry draft_n and draft_n_accepted) is answered, and the host is
# recorded again after each. Then the server and the guard stop; the rig's logs come back to OUT and are removed on the
# rig. Runs only ssh here. Environment: RIG, BENCH (bench/agent_check), REL_BIN, MODEL, RIG_PORT, ARGS,
# SMOKE_LOGDIR (a folder on the rig), REQUESTS, GUARD (mem_guard.sh here), optional CTX (65536), MIN_AVAIL_MB (400).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: smoke_dflash.sh OUT}")
: "${RIG:?}" "${BENCH:?}" "${REL_BIN:?}" "${MODEL:?}" "${RIG_PORT:?}" "${ARGS:?}" "${SMOKE_LOGDIR:?}" "${REQUESTS:?}" \
  "${GUARD:?}"
CTX=${CTX:-65536}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/smoke.log"; }
rs() { ssh "$RIG" "LOGDIR=$SMOKE_LOGDIR MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} PORT=$RIG_PORT $1 bash -s" < "$BENCH/rig_server.sh"; }
host() { ssh "$RIG" "cat /proc/meminfo; cat /proc/\$(cat $SMOKE_LOGDIR/server.pid)/status" > "$OUT/host_$1.txt"; }

ssh "$RIG" "mkdir -p $SMOKE_LOGDIR && cat > $SMOKE_LOGDIR/mem_guard.sh" < "$GUARD"
ssh "$RIG" "LOG=$SMOKE_LOGDIR/mem_guard.log setsid nohup chrt -f 10 bash $SMOKE_LOGDIR/mem_guard.sh > /dev/null 2>&1 \
  < /dev/null & sleep 1; cat $SMOKE_LOGDIR/mem_guard.log.pid" > "$OUT/guard.pid" && say "guard up ($(cat "$OUT/guard.pid"))"
say "starting: $REL_BIN $COMMON $ARGS -m $MODEL"
rs "ACTION=start ARM=smoke BIN=$REL_BIN MODEL=$MODEL SERVER_ENV='' SERVER_ARGS='$COMMON $ARGS'" 2>&1 | tee -a "$OUT/smoke.log"
up=${PIPESTATUS[0]}
if [ "$up" = 0 ]; then
  ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$OUT/vram_loaded.csv"
  host loaded
  n=0
  for r in $REQUESTS; do
    n=$((n + 1))
    say "request $n: $(basename "$r")"
    ssh "$RIG" "curl -s -m 900 127.0.0.1:$RIG_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
      < "$r" > "$OUT/reply_$n.json"
    say "request $n done ($(wc -c < "$OUT/reply_$n.json") bytes)"
    host "after_request_$n"
  done
  ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$OUT/vram_after.csv"
  ssh "$RIG" "curl -s 127.0.0.1:$RIG_PORT/metrics" > "$OUT/metrics.txt"
else
  say "the server did not come up"
fi
rs "ACTION=end" 2>&1 | tee -a "$OUT/smoke.log"
ssh "$RIG" "kill \$(cat $SMOKE_LOGDIR/mem_guard.log.pid) 2> /dev/null; rm -f $SMOKE_LOGDIR/mem_guard.sh"
ssh "$RIG" "tar -C $SMOKE_LOGDIR -cf - ." | tar -C "$OUT" -xf - && ssh "$RIG" "rm -rf $SMOKE_LOGDIR" \
  && say "rig logs in $OUT, removed on the rig"
[ "$up" = 0 ]
