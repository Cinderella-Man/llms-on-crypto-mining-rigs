#!/usr/bin/env bash
# verify_prof.sh: R49 (Q1211). The verify pass of a served speculative step, launch group by launch group: one load of
# the Miner Llama v3 preset under the fork's profiler in group mode that answers a greedy request, so that the passes
# come from generation (with the MTP draft loaded a prompt gives no pass under five tokens). Runs ON the rig,
# detached, the rig idle; ends with $OUT/DONE.
# Every load is the preset exactly (settings.env: PRESET_ENV, PRESET_ARGS, the MTP draft), started through
# bench/agent_check/rig_server.sh (its monitors and RAM watchdog), as 2026-11-10_served_step_split/run/step_loads.sh
# started its loads. Loads (LOADS, in this order):
#   warm  no profile: warms the cards (the first load after an idle hour can run slow) and gives the server's own
#         timings of the same requests without the profiler
#   prof  under GGML_CUDA_PROFILE=$OUT/prof/prof-verify.csv GGML_CUDA_PROFILE_MODE=group: one CSV line per launch group
#         of every graph evaluation, an event between every two groups, the stream synchronized after every evaluation
# Each load answers REQ (2026-11-04_draft_one_card/run/greedy_request.json: temperature 0, 256 tokens) REPEATS times,
# PAUSE seconds apart (the gap cuts the trace into requests). The request draws nothing at random, so every repeat
# runs the same steps: a load's later requests are further samples of each part, without the one-off launches of a
# load's first request.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<settings.env>
#   RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<greedy_request.json> OUT=<dir>
#   [LOADS='warm prof'] [REPEATS=2] [PORT=8193] [PAUSE=4] [MIN_AVAIL_MB=60] bash verify_prof.sh
# Output: per load OUT/<load>/ with reply_greedy-<i>.json, server-<load>.log, gpu.csv, vmstat.log, meminfo.log and
#   ram_watchdog-<load>.tsv, and OUT/prof/prof-verify.csv; OUT/status.txt (with the epoch time around every request);
#   OUT/cards.txt (memory, temperature and clock per card before each load, once loaded and after its requests)
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${OUT:?}"
LOADS=${LOADS:-warm prof}
REPEATS=${REPEATS:-2}
PORT=${PORT:-8193}; PAUSE=${PAUSE:-4}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
cards() { # <label>: memory used, temperature and clock per card
    { echo "# $(date -Is) $1"; nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.gr --format=csv,noheader; } >> "$OUT/cards.txt"
}
rs() { # <load> VAR=value ...: rig_server.sh in the load's folder
    local d=$OUT/$1
    shift
    env LOGDIR="$d" PORT="$PORT" MIN_AVAIL_MB="${MIN_AVAIL_MB:-60}" "$@" bash "$RIG_SERVER" >> "$OUT/status.txt" 2>&1
}

for load in $LOADS; do
    d=$OUT/$load
    mkdir -p "$d"
    case $load in
        warm) prof="" ;;
        prof) prof="GGML_CUDA_PROFILE=$d/prof-verify.csv GGML_CUDA_PROFILE_MODE=group" ;;
        *) log "unknown load $load"; continue ;;
    esac
    pkill -x llama-server; sleep 2; quiet; cards "$load: before the load"
    log "$load: server start $(date +%s.%3N)"
    if ! rs "$load" ACTION=start ARM="$load" BIN="$BIN/llama-server" MODEL="$MODEL" \
            "SERVER_ENV=$PRESET_ENV $prof" "SERVER_ARGS=$PRESET_ARGS -md $DRAFT"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        continue
    fi
    log "$load: server up ($PRESET_ENV $prof; $PRESET_ARGS -md $DRAFT)"
    cards "$load: loaded"
    for i in $(seq 1 "$REPEATS"); do
        sleep "$PAUSE"
        log "$load greedy-$i: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$REQ" \
            > "$d/reply_greedy-$i.json"
        log "$load greedy-$i: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_greedy-$i.json") bytes)"
    done
    cards "$load: after the requests"
    # the group-mode trace is flushed after every evaluation, so it is whole whichever way the server ends
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    log "$load: stopped"
done
sleep 2; quiet; cards "end"
touch "$OUT/DONE"
