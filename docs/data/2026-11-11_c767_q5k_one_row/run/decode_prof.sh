#!/usr/bin/env bash
# decode_prof.sh: C767 (Q1348). A served plain decode step, launch group by launch group: R49's verify_prof.sh
# (2026-11-10_served_verify_parts/run/verify_prof.sh) without the MTP draft, so that every generated token is one pass
# of one token and the output layer's product is a one-token product. One load of the Miner Llama v3 preset's line
# (PRESET_ENV, PRESET_ARGS) without the draft and its arguments (--spec-*, -otd, -md, as
# 2026-11-10_release_profile/run/release_prof.sh drops them), started through bench/agent_check/rig_server.sh (its
# monitors and RAM watchdog). Runs ON the rig, detached, the rig idle; ends with $OUT/DONE. Loads (LOADS, in this order):
#   warm  no profile: warms the cards and gives the server's own timings of the same requests without the profiler
#   prof  under GGML_CUDA_PROFILE=$OUT/prof/prof-verify.csv GGML_CUDA_PROFILE_MODE=group: one CSV line per launch group
#         of every graph evaluation (the file name is verify_prof.sh's, so that its tools read it as they are)
# Each load answers REQ (2026-11-04_draft_one_card/run/greedy_request.json: temperature 0, 256 tokens) REPEATS times,
# PAUSE seconds apart. Greedy, so every repeat and every load runs the same steps, and a switch whose sums are the
# one-token kernel's bit for bit gives the same reply text.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> PRESET=<settings.env>
#   RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<greedy_request.json> OUT=<dir>
#   [LOADS='warm prof'] [REPEATS=2] [PORT=8193] [PAUSE=4] [MIN_AVAIL_MB=60] bash decode_prof.sh
# Output: as verify_prof.sh: per load OUT/<load>/ with reply_greedy-<i>.json, the server's log and monitors, and
#   OUT/prof/prof-verify.csv; OUT/status.txt; OUT/cards.txt
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${OUT:?}"
LOADS=${LOADS:-warm prof}
REPEATS=${REPEATS:-2}
PORT=${PORT:-8193}; PAUSE=${PAUSE:-4}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
# the preset's arguments without the draft's
args=()
# shellcheck disable=SC2086
set -- $PRESET_ARGS
while [ $# -gt 0 ]; do
    case $1 in --spec-*|-otd|-md) shift 2 ;; *) args+=("$1"); shift ;; esac
done
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
            "SERVER_ENV=$PRESET_ENV $prof" "SERVER_ARGS=${args[*]}"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        continue
    fi
    log "$load: server up ($PRESET_ENV $prof; ${args[*]})"
    cards "$load: loaded"
    for i in $(seq 1 "$REPEATS"); do
        sleep "$PAUSE"
        log "$load greedy-$i: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$REQ" \
            > "$d/reply_greedy-$i.json"
        log "$load greedy-$i: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_greedy-$i.json") bytes)"
    done
    cards "$load: after the requests"
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    log "$load: stopped"
done
sleep 2; quiet; cards "end"
touch "$OUT/DONE"
