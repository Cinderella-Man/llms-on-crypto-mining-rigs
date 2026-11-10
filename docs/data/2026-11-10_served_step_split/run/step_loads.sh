#!/usr/bin/env bash
# step_loads.sh: S37 (Q961) on the release line. What one served speculative step of the Miner Llama v3 preset costs
# beyond its verify pass, and the MTP draft's two one-card switches as arms. Runs ON the rig, detached, the rig idle;
# ends with $OUT/DONE.
# Every load is the preset exactly (settings.env: PRESET_ENV, PRESET_ARGS, the MTP draft), started through
# bench/agent_check/rig_server.sh (its monitors and RAM watchdog, as the release's agent arms ran), plus the load's
# switches and the fork's host profile (GGML_CUDA_PROFILE_HOST: one line per graph evaluation and card, two events an
# evaluation, nothing synchronized). A load's name up to its "-" is its arm:
#   warm     the preset, no profile: warms the cards (the first load after an idle hour can run slow) and gives the
#            server's own timings without the profile
#   d1off    the preset
#   d1cache  + LLAMA_DRAFT_EMBD_CACHE=8192  the draft's token rows cached on its layer's card
#   d1carry  + LLAMA_DRAFT_H_CARRY=1        the draft's hidden rows kept on that card between a step's passes
#   d1on     both
#   d1off-2  the preset again: the spread of two loads of one arm
# Each load answers three requests in order, PAUSE seconds apart (the gap cuts the trace into requests): REQ_LONG (the
# agent's first-turn request, bench/agent_check/first_turn_request.sh), REQ_GREEDY
# (2026-11-04_draft_one_card/run/greedy_request.json) and REQ_SMOKE (2026-11-03_flash_next_mtp/run/smoke_request.json).
# The step is read from their generation: with the draft loaded a prompt gives no pass under five tokens.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<settings.env>
#   RIG_SERVER=<bench/agent_check/rig_server.sh> REQ_LONG=<json> REQ_GREEDY=<json> REQ_SMOKE=<json> OUT=<dir>
#   [LOADS='warm d1off d1cache d1carry d1on d1off-2'] [PORT=8193] [PAUSE=4] [MIN_AVAIL_MB=60] bash step_loads.sh
# Output: per load OUT/<load>/ with host_profile.csv, reply_{long,greedy,smoke}.json, server-<load>.log, gpu.csv,
#   vmstat.log, meminfo.log and ram_watchdog-<load>.tsv; OUT/status.txt (with the epoch time around every request);
#   OUT/cards.txt (memory, temperature and clock per card before each load, once loaded and after its requests)
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ_LONG:?}" "${REQ_GREEDY:?}" "${REQ_SMOKE:?}" "${OUT:?}"
LOADS=${LOADS:-warm d1off d1cache d1carry d1on d1off-2}
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
arm_env() {
    case $1 in
        warm|d1off) echo "" ;;
        d1cache) echo "LLAMA_DRAFT_EMBD_CACHE=8192" ;;
        d1carry) echo "LLAMA_DRAFT_H_CARRY=1" ;;
        d1on) echo "LLAMA_DRAFT_EMBD_CACHE=8192 LLAMA_DRAFT_H_CARRY=1" ;;
        *) return 1 ;;
    esac
}

for load in $LOADS; do
    arm=${load%%-*}
    sw=$(arm_env "$arm") || { log "unknown load $load"; continue; }
    d=$OUT/$load
    mkdir -p "$d"
    prof=""
    [ "$arm" = warm ] || prof="GGML_CUDA_PROFILE_HOST=$d/host_profile.csv"
    pkill -x llama-server; sleep 2; quiet; cards "$load: before the load"
    log "$load: server start $(date +%s.%3N)"
    if ! rs "$load" ACTION=start ARM="$load" BIN="$BIN/llama-server" MODEL="$MODEL" \
            "SERVER_ENV=$PRESET_ENV $sw $prof" "SERVER_ARGS=$PRESET_ARGS -md $DRAFT"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        continue
    fi
    log "$load: server up ($PRESET_ENV $sw $prof; $PRESET_ARGS -md $DRAFT)"
    cards "$load: loaded"
    for r in long greedy smoke; do
        case $r in long) f=$REQ_LONG ;; greedy) f=$REQ_GREEDY ;; smoke) f=$REQ_SMOKE ;; esac
        sleep "$PAUSE"
        log "$load $r: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$f" \
            > "$d/reply_$r.json"
        log "$load $r: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_$r.json") bytes)"
    done
    cards "$load: after the requests"
    # rig_server.sh ends the server with SIGTERM and waits for it, so that the trace is whole
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    log "$load: stopped"
done
sleep 2; quiet; cards "end"
touch "$OUT/DONE"
