#!/usr/bin/env bash
# rig_taper.sh: C218r (Q771, Q770's first test), ON the rig and detached, from a folder that holds this script,
# taper_replay.py and the agent check's turns.csv (docs/data/2026-11-03_retype_agent/agent/turns.csv). Ends with OUT/DONE.
# Two server loads of the taper build (pascal-p106-v3-b12076-taper) with the agent check's line; the taper is off on the
# server and set per request ("ubatch_taper"), so each load runs both arms:
#   A  --ctx-checkpoints 4 (as the agent check): the replay of the check's 62 prompts in PASSES_A (default off, on, on,
#      off), then FRESH fresh prompts of FRESH_N tokens with the taper off and on
#   B  --ctx-checkpoints 0 (no checkpoint drains, Q770's no-code test): the same with PASSES_B (default off, on), so
#      that A's and B's fresh prompts give the drains' cost, and B with the taper on approximates no drains plus taper
# Every load logs its time, memory per card and on the host, the checkpoint and taper lines of its server log, and a
# watchdog stops the server when MemAvailable falls below MIN_AVAIL_MB.
# Since C804 (Q1408, 2026-11-12) every load also counts the target graph's reallocations per request from its server
# log's print_timing lines (`graph reallocs = S / D`, as scripts/realloc_counts.py counts them): the table in
# load-<L>/reallocs.tsv and one line in run.log ("load <L>: graph reallocations: W of N requests (S at the same size,
# D changed)"). The loads and the requests are the same as before; a switch whose loads reallocate where its line
# without it does not is marked with scripts/realloc_counts.py --marks.
# Env: BUILDS (folder of the builds), BUILD (the taper build's folder), MODEL (first shard), OUT (default ./out),
#      CTX (65536), ARGS (default '-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio'), PORT (8095),
#      LOADS ("A B"), PASSES_A ("off on on off"; a spec such as short=2048:128 is a pass too), PASSES_B ("off on"),
#      FRESH (4), FRESH_N (2048), GEN (8), PAD (0),
#      TASKS (all), MIN_AVAIL_MB (150), FORCE=1 starts even with a compute process on the cards.
#   ssh $RIG "BUILDS=<builds> BUILD=<taper build> MODEL=<first shard> OUT=<work>/out \
#     setsid nohup bash <work>/rig_taper.sh > <work>/rig_taper.out 2>&1 < /dev/null &"
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${BUILDS:?BUILDS=<folder of the builds>}" "${BUILD:?BUILD=<the taper build>}" "${MODEL:?MODEL=<first shard>}"
OUT=${OUT:-$here/out}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
PORT=${PORT:-8095}
LOADS=${LOADS:-A B}
PASSES_A=${PASSES_A:-off on on off}
PASSES_B=${PASSES_B:-off on}
FRESH=${FRESH:-4}
FRESH_N=${FRESH_N:-2048}
GEN=${GEN:-8}
PAD=${PAD:-0}
TASKS=${TASKS:-}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-150}
FORCE=${FORCE:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
# realloc_count SERVER_LOG TSV (C804): the target graph's reallocations per request from print_timing's running count
# (`graph reallocs = S / D` since the load, after the request's `prompt eval time = X ms / N tokens`); writes one row
# per request to TSV and prints the summary
realloc_count() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" 2> /dev/null | awk -v out="$2" '
        BEGIN { print "req\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed" > out }
        /prompt eval time =/ { t = $0; sub(/.*prompt eval time = */, "", t); split(t, p, " "); ms = p[1]; n = p[4]; cur = 1; next }
        cur && /graph reallocs =/ { t = $0; sub(/.*graph reallocs = */, "", t); split(t, q, " ")
            s = q[1] - ps; d = q[3] - pd; ps = q[1]; pd = q[3]; cur = 0
            printf "%d\t%s\t%s\t%d\t%d\n", k, n, ms, s, d > out
            k++; ts += s; td += d; if (s + d > 0) w++ }
        END { printf "graph reallocations: %d of %d requests (%d at the same size, %d changed)\n", w, k, ts, td }'
}

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$here/rig_taper.sh" "$here/taper_replay.py" "$here/turns.csv" "$BUILDS/$BUILD/bin/llama-server" > "$OUT/md5.txt" 2>&1
grep -E '^(name|commit)' "$BUILDS/$BUILD/BUILDINFO.txt" >> "$OUT/md5.txt" 2>/dev/null

for L in $LOADS; do
    case $L in
        A) CK=4; PASSES=$PASSES_A ;;
        B) CK=0; PASSES=$PASSES_B ;;
        *) log "unknown load $L"; continue ;;
    esac
    d=$OUT/load-$L
    mkdir -p "$d"
    log "load $L: --ctx-checkpoints $CK, passes $PASSES, $FRESH fresh prompts of $FRESH_N tokens"
    pkill -x llama-server; sleep 2
    # shellcheck disable=SC2086
    "$BUILDS/$BUILD/bin/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" $ARGS -np 1 -bs --ctx-checkpoints "$CK" \
        --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5 --host 127.0.0.1 --port "$PORT" \
        > "$d/server.log" 2>&1 < /dev/null &
    srv=$!
    # host-memory watchdog: the 4 GB host must not swap the server out or start the OOM killer
    ( while kill -0 $srv 2> /dev/null; do
          a=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
          if [ "$a" -lt "$MIN_AVAIL_MB" ]; then
              echo "$(date -Is) watchdog: MemAvailable $a MiB < $MIN_AVAIL_MB, stopping the server" >> "$OUT/run.log"
              kill -INT $srv; break
          fi
          echo "$(date +%s) $a" >> "$d/memavail.txt"
          sleep 2
      done ) &
    wd=$!
    t0=$(date +%s)
    for _ in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    log "load $L: up after $(( $(date +%s) - t0 )) s"
    grep -E 'model buffer size|KV buffer size|RS buffer size|compute buffer size|n_ctx *=|n_ubatch|checkpoints|out of memory|failed|error' \
        "$d/server.log" | grep -v 'unused tensor' | cut -c1-220 | head -60 > "$d/load_lines.txt"
    { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; free -m; } > "$d/memory_loaded.txt"
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        python3 -I "$here/taper_replay.py" --port "$PORT" --turns "$here/turns.csv" --passes "$PASSES" --gen "$GEN" \
            --pad "$PAD" --fresh "$FRESH" --fresh-n "$FRESH_N" ${TASKS:+--tasks "$TASKS"} --label "$L" --out "$d" \
            > "$d/replay.out" 2>&1
        log "load $L: replay exit $?; $(tr '\n' ' ' < "$d/summary.txt" 2>/dev/null | cut -c1-600)"
    else
        log "load $L: server did not start"; tail -20 "$d/server.log" >> "$OUT/run.log"
    fi
    { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; free -m; } > "$d/memory_after.txt"
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server; kill $wd 2> /dev/null
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -oE '(created|restored) context checkpoint.*' > "$d/checkpoints.txt"
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -o 'ubatch taper: .*' > "$d/taper_lines.txt"
    log "load $L: $(wc -l < "$d/checkpoints.txt") checkpoint lines, $(wc -l < "$d/taper_lines.txt") taper lines, lowest MemAvailable $(sort -k2 -n "$d/memavail.txt" | head -n 1 | cut -d' ' -f2) MiB"
    log "load $L: $(realloc_count "$d/server.log" "$d/reallocs.tsv")"
done
log "done"
touch "$OUT/DONE"
