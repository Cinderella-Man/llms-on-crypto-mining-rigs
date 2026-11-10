#!/usr/bin/env bash
# r42_run.sh: R42 (Q1156), one long prompt's per-card timeline on the Miner Llama v3.0 release line. Runs ON the rig,
# detached, the rig idle; ends with $OUT/DONE. Each load is one call of R43's r43_run.sh (the preset's environment and
# line, MTP draft included, through rig_taper.sh and taper_replay.py of docs/data/2026-11-04_prompt_taper/run/) with
# LOADS=A (--ctx-checkpoints 4, as the preset), FRESH=2 fresh prompts of FRESH_N tokens, each with the micro-batch
# taper off and on, and the replay cut to one task (TASK) and one pass with the taper off (PASSES_A=off). The loads, in
# this order (R42_LOADS):
#   warm4096   unprofiled: warms the cards (Q1055) and gives the unprofiled times of the 4,097-token prompts
#   graph4096  under GGML_CUDA_PROFILE=<load>/prof-graph.csv GGML_CUDA_PROFILE_MODE=graph (an event pair around each
#              graph evaluation, no synchronization)
#   graph450   the same with 451-token prompts
#   host4096   under GGML_CUDA_PROFILE_HOST=<load>/prof-host.csv (the host's entry to and exit from each graph
#              evaluation and its GPU time)
#   host450    the same with 451-token prompts
# Usage: R43=<folder with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv, settings.env> BUILDS=<builds>
#   BUILD=<release build folder> MODEL=<Swift Q2_0 first shard> DRAFT=<MTP draft> OUT=<dir> [TASK=...] [R42_LOADS=...]
#   setsid nohup bash r42_run.sh > <dir>.out 2>&1 < /dev/null &
set -uo pipefail
: "${R43:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}"
TASK=${TASK:-bugfix-javascript-triangle}
R42_LOADS=${R42_LOADS:-warm4096 graph4096 graph450 host4096 host450}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
cards() { { echo "# $(date -Is) $1"; nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.gr --format=csv,noheader; } >> "$OUT/cards.txt"; }
md5sum "$0" "$R43"/r43_run.sh "$R43"/rig_taper.sh "$R43"/taper_replay.py "$R43"/turns.csv "$R43"/settings.env > "$OUT/md5.txt"
for L in $R42_LOADS; do
    n=${L##*[a-z]}
    mode=${L%%[0-9]*}
    d=$OUT/$L
    mkdir -p "$d"
    prof=()
    case $mode in
        warm) ;;
        graph) prof=(GGML_CUDA_PROFILE="$d/prof-graph.csv" GGML_CUDA_PROFILE_MODE=graph) ;;
        host) prof=(GGML_CUDA_PROFILE_HOST="$d/prof-host.csv") ;;
        *) log "unknown load $L"; continue ;;
    esac
    # the previous load's server must be gone (rig_taper.sh refuses to start next to a compute process)
    for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ] && break; sleep 2; done
    cards "$L: before"
    log "$L: start $(date +%s.%3N); FRESH_N=$n ${prof[*]:-unprofiled}; load $(cut -d' ' -f1-3 /proc/loadavg); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"
    env "${prof[@]+"${prof[@]}"}" BUILDS="$BUILDS" BUILD="$BUILD" MODEL="$MODEL" DRAFT="$DRAFT" OUT="$d" LOADS=A FRESH=2 \
        FRESH_N="$n" TASKS="$TASK" PASSES_A=off bash "$R43/r43_run.sh" > "$d/r43_run.out" 2>&1 < /dev/null
    log "$L: exit $? end $(date +%s.%3N)"
done
pkill -x llama-server
sleep 3
cards "end"
log "done; llama processes $(ps -eo comm | grep -c '^llama-')"
touch "$OUT/DONE"
