#!/usr/bin/env bash
# smoke_dflash.sh OUT: Q561's smoke test of a draft line before its timed agent check (smoke_mtp.sh from Q226, with
# several requests). llama-server starts on the rig with release_check.sh's COMMON settings and ARGS (through
# bench/agent_check/rig_server.sh, so with the same monitors and its RAM watchdog, scripts/ram_watchdog.sh: the stop on
# MemAvailable minus CmaFree, memory pressure and swap-out, Q559; it took over from S15's mem_guard.sh, which this script
# used to start). Each card's memory and the host's (/proc/meminfo, the server's /proc status) are recorded once loaded.
# Then the agent's first-turn request is answered (the first task's whole request, ~2,780 tokens, built at run time from the
# task files by bench/agent_check/first_turn_request.sh, Q607; LONG_REQUEST=off skips it), then each request in REQUESTS
# (chat completion bodies, files here, sent in order; the replies' timings carry draft_n and draft_n_accepted), and the host
# is recorded again after each. The first-turn request goes first, as it does in the agent check: the first request after a
# load is the one that crossed -b 2048's memory in Q561. Then the server and the watchdog stop; the rig's logs (with
# ram_watchdog-smoke.tsv) come back to OUT and are removed on the rig. Exit status: 0 when the server came up and every
# request got a reply. Runs only ssh here. Environment: RIG, BENCH (bench/agent_check), REL_BIN, MODEL, RIG_PORT, ARGS,
# SMOKE_LOGDIR (a folder on the rig), REQUESTS, BUILD_SSH or CLIENT (where first_turn_request.sh builds the request),
# optional SMOKE_ENV (environment variables for the server, "VAR=value ...", e.g. "LLAMA_SPEC_ADAPT=1"; none), CTX
# (65536), MIN_AVAIL_MB (400), WD_ENV (limits for the watchdog, e.g. "PSI_FULL_PCT=30 MIN_USABLE_MB=40"),
# LONG_REQUEST (on; off), LONG_TOKENS and the other LONG_* of first_turn_request.sh, DRY_RUN=1 (build the first-turn
# request and say what would be sent, without touching the rig).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: smoke_dflash.sh OUT}")
: "${RIG:?}" "${BENCH:?}" "${REL_BIN:?}" "${MODEL:?}" "${RIG_PORT:?}" "${ARGS:?}" "${SMOKE_LOGDIR:?}" "${REQUESTS:?}"
CTX=${CTX:-65536}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
LONG=${LONG_REQUEST:-on}
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/smoke.log"; }
rs() { ssh "$RIG" "LOGDIR=$SMOKE_LOGDIR MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} ${WD_ENV:-} PORT=$RIG_PORT $1 bash -s" < "$BENCH/rig_server.sh"; }
host() { ssh "$RIG" "cat /proc/meminfo; cat /proc/\$(cat $SMOKE_LOGDIR/server.pid)/status" > "$OUT/host_$1.txt"; }
field() { local v; v=$(grep -o "\"$1\":[0-9.]*" "$2" | head -n 1); echo "${v#*:}"; }  # first "name":number in a reply
send() {  # send REQUEST-FILE REPLY-FILE: the request to the server on the rig, the reply here
  ssh "$RIG" "curl -s -m 900 127.0.0.1:$RIG_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" < "$1" > "$2"
}

if [ "$LONG" != off ]; then
  say "building the first-turn request (task files rebuilt on the build machine)"
  bash "$BENCH/first_turn_request.sh" "$OUT/long_request.json" 2>&1 | tee -a "$OUT/smoke.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "the first-turn request could not be built; stopping before the rig is touched"; exit 1; }
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
  say "dry run: would start ${REL_BIN##*/} on port $RIG_PORT with $COMMON $ARGS"
  [ "$LONG" != off ] && say "dry run: then send the first-turn request, $OUT/long_request.json ($(wc -c < "$OUT/long_request.json") bytes)"
  say "dry run: then send $REQUESTS"
  exit 0
fi

say "starting: ${SMOKE_ENV:+$SMOKE_ENV }$REL_BIN $COMMON $ARGS -m $MODEL"
rs "ACTION=start ARM=smoke BIN=$REL_BIN MODEL=$MODEL SERVER_ENV='${SMOKE_ENV:-}' SERVER_ARGS='$COMMON $ARGS'" 2>&1 | tee -a "$OUT/smoke.log"
up=${PIPESTATUS[0]}
bad=0
if [ "$up" = 0 ]; then
  ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$OUT/vram_loaded.csv"
  host loaded
  if [ "$LONG" != off ]; then
    say "request long: the first turn ($(wc -c < "$OUT/long_request.json") bytes)"
    send "$OUT/long_request.json" "$OUT/reply_long.json"
    say "request long done ($(wc -c < "$OUT/reply_long.json") bytes): prompt tokens $(field prompt_tokens "$OUT/reply_long.json")" \
      "(target ${LONG_TOKENS:-2780}), generated $(field completion_tokens "$OUT/reply_long.json")," \
      "drafted $(field draft_n "$OUT/reply_long.json"), accepted $(field draft_n_accepted "$OUT/reply_long.json")"
    [ -s "$OUT/reply_long.json" ] || bad=1
    host after_request_long
  fi
  n=0
  for r in $REQUESTS; do
    n=$((n + 1))
    say "request $n: $(basename "$r")"
    send "$r" "$OUT/reply_$n.json"
    say "request $n done ($(wc -c < "$OUT/reply_$n.json") bytes)"
    [ -s "$OUT/reply_$n.json" ] || bad=1
    host "after_request_$n"
  done
  ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$OUT/vram_after.csv"
  ssh "$RIG" "curl -s 127.0.0.1:$RIG_PORT/metrics" > "$OUT/metrics.txt"
else
  say "the server did not come up"
fi
rs "ACTION=end" 2>&1 | tee -a "$OUT/smoke.log"
ssh "$RIG" "tar -C $SMOKE_LOGDIR -cf - ." | tar -C "$OUT" -xf - && ssh "$RIG" "rm -rf $SMOKE_LOGDIR" \
  && say "rig logs in $OUT, removed on the rig"
if [ -f "$OUT/ram_watchdog-smoke.tsv" ]; then
  say "RAM watchdog: $(grep -vc '^#' "$OUT/ram_watchdog-smoke.tsv") samples, $(grep -c '^# [0-9]* KILL ' "$OUT/ram_watchdog-smoke.tsv") kills"
else
  say "RAM watchdog: no ram_watchdog-smoke.tsv came back"
  bad=1
fi
[ "$up" = 0 ] && [ "$bad" = 0 ]
