#!/usr/bin/env bash
# rig_trace.sh: C405 (Q960) on a P106 rig, untimed: the server's step trace (LLAMA_STEP_TRACE) beside the fork's host
# profile (GGML_CUDA_PROFILE_HOST), S37's method (2026-11-10_served_step_split/run/step_loads.sh) on a build that has
# the trace. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Every load is the preset exactly (settings.env: PRESET_ENV, PRESET_ARGS, the MTP draft), started through
# bench/agent_check/rig_server.sh (its monitors and RAM watchdog), plus the host profile. Loads:
#   trace  + LLAMA_STEP_TRACE=<load>/step_trace.jsonl: the step's stages on the host
#   prof   the profile alone: the step the trace is checked against (step_split.py on both, paired on the greedy
#          request, which draws nothing at random)
# Each load answers REQ_GREEDY then REQ_LONG, PAUSE seconds apart (the gap cuts the profile into requests).
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<settings.env>
#   RIG_SERVER=<bench/agent_check/rig_server.sh> REQ_GREEDY=<json> REQ_LONG=<json> OUT=<dir>
#   [LOADS='trace prof'] [PORT=8193] [PAUSE=4] [MIN_AVAIL_MB=60] bash rig_trace.sh
# Output: per load OUT/<load>/ with host_profile.csv, step_trace.jsonl (trace), reply_{greedy,long}.json,
#   server-<load>.log and rig_server.sh's monitor files; OUT/status.txt (the epoch time around every request)
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ_GREEDY:?}" "${REQ_LONG:?}" "${OUT:?}"
LOADS=${LOADS:-trace prof}
PORT=${PORT:-8193}; PAUSE=${PAUSE:-4}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
rs() { # <load> VAR=value ...: rig_server.sh in the load's folder
    local d=$OUT/$1
    shift
    env LOGDIR="$d" PORT="$PORT" MIN_AVAIL_MB="${MIN_AVAIL_MB:-60}" "$@" bash "$RIG_SERVER" >> "$OUT/status.txt" 2>&1
}

for load in $LOADS; do
    d=$OUT/$load
    mkdir -p "$d"
    sw="GGML_CUDA_PROFILE_HOST=$d/host_profile.csv"
    case $load in
        trace) sw="$sw LLAMA_STEP_TRACE=$d/step_trace.jsonl" ;;
        prof) ;;
        *) log "unknown load $load"; continue ;;
    esac
    pkill -x llama-server; sleep 2; quiet
    log "$load: server start $(date +%s.%3N)"
    if ! rs "$load" ACTION=start ARM="$load" BIN="$BIN/llama-server" MODEL="$MODEL" \
            "SERVER_ENV=$PRESET_ENV $sw" "SERVER_ARGS=$PRESET_ARGS -md $DRAFT"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        continue
    fi
    log "$load: server up ($PRESET_ENV $sw; $PRESET_ARGS -md $DRAFT)"
    for r in greedy long; do
        case $r in greedy) f=$REQ_GREEDY ;; long) f=$REQ_LONG ;; esac
        sleep "$PAUSE"
        log "$load $r: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$f" \
            > "$d/reply_$r.json"
        log "$load $r: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_$r.json") bytes)"
    done
    # rig_server.sh ends the server with SIGTERM and waits for it, so that both files are whole
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    log "$load: stopped"
done
sleep 2; quiet
touch "$OUT/DONE"
