#!/usr/bin/env bash
# rig_swap.sh: C752 (Q1328) / R96 (Q1337) on a P106 rig, untimed: what the ~117 ms host waits before card 0's launch
# wait on when the long request swaps. R86's driver (2026-11-11_c405_server_gap/run/rig_trace.sh) with the greedy
# request first (R86's order, which swapped in both of its loads), two traces and a sampler of the server's main
# thread. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Every load is the preset exactly (settings.env: PRESET_ENV, PRESET_ARGS, the MTP draft), started through
# bench/agent_check/rig_server.sh (its monitors and RAM watchdog), plus:
#   GGML_CUDA_PROFILE_HOST    the host profile (card 0's graph_compute entry)
#   LLAMA_STEP_TRACE          the step's host stages (C405: the wait sits between v.ub.in and card 0's entry)
#   LLAMA_UBATCH_TRACE=1      each split's wait, copy and launch in the server log (C701: which part of that stage)
#   proc_sampler.sh           the main thread's state, kernel wait channel and faults, the host's swap and fault
#                             counters, every SAMPLE seconds
# Loads:
#   swap   the preset
#   vram   the preset and --ctx-checkpoints-vram 4 (Q440: a slot's checkpoints on the cards, 112.6 MiB a
#          checkpoint off the host)
# Each load answers REQ_GREEDY then REQ_LONG, PAUSE seconds apart.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<settings.env>
#   RIG_SERVER=<bench/agent_check/rig_server.sh> SAMPLER=<run/proc_sampler.sh> REQ_GREEDY=<json> REQ_LONG=<json>
#   OUT=<dir> [LOADS='swap vram'] [PORT=8193] [PAUSE=4] [SAMPLE=0.05] [MIN_AVAIL_MB=60] bash rig_swap.sh
# Output: per load OUT/<load>/ with host_profile.csv, step_trace.jsonl, proc.tsv, reply_{greedy,long}.json,
#   server-<load>.log and rig_server.sh's monitor files; OUT/status.txt (the epoch time around every request)
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${SAMPLER:?}" "${REQ_GREEDY:?}" "${REQ_LONG:?}" "${OUT:?}"
LOADS=${LOADS:-swap vram}
PORT=${PORT:-8193}; PAUSE=${PAUSE:-4}; SAMPLE=${SAMPLE:-0.05}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^(MemAvailable|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; }
rs() { # <load> VAR=value ...: rig_server.sh in the load's folder
    local d=$OUT/$1
    shift
    env LOGDIR="$d" PORT="$PORT" MIN_AVAIL_MB="${MIN_AVAIL_MB:-60}" "$@" bash "$RIG_SERVER" >> "$OUT/status.txt" 2>&1
}
for load in $LOADS; do
    d=$OUT/$load
    mkdir -p "$d"
    sw="GGML_CUDA_PROFILE_HOST=$d/host_profile.csv LLAMA_STEP_TRACE=$d/step_trace.jsonl LLAMA_UBATCH_TRACE=1"
    args=""
    case $load in
        swap) ;;
        vram) args="--ctx-checkpoints-vram 4" ;;
        *) log "unknown load $load"; continue ;;
    esac
    pkill -x llama-server; sleep 2; quiet
    log "$load: server start $(date +%s.%3N)"
    if ! rs "$load" ACTION=start ARM="$load" BIN="$BIN/llama-server" MODEL="$MODEL" \
            "SERVER_ENV=$PRESET_ENV $sw" "SERVER_ARGS=$PRESET_ARGS $args -md $DRAFT"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        continue
    fi
    pid=$(pgrep -x llama-server | head -n 1)
    PID=$pid OUT=$d/proc.tsv INTERVAL=$SAMPLE setsid nohup bash "$SAMPLER" > /dev/null 2>&1 < /dev/null &
    log "$load: server up, pid $pid, sampler every $SAMPLE s ($PRESET_ENV $sw; $PRESET_ARGS $args -md $DRAFT)"
    for r in greedy long; do
        case $r in greedy) f=$REQ_GREEDY ;; long) f=$REQ_LONG ;; esac
        sleep "$PAUSE"
        log "$load $r: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$f" \
            > "$d/reply_$r.json"
        log "$load $r: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_$r.json") bytes)"
    done
    # rig_server.sh ends the server with SIGTERM and waits for it, so that every file is whole; the sampler ends
    # with the server
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    log "$load: stopped"
done
sleep 2; quiet
touch "$OUT/DONE"
