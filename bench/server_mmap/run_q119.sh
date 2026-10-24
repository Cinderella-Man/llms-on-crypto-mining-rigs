#!/usr/bin/env bash
# run_q119.sh: does llama-server lose time to page faults on the model file, and do --no-mmap or --mlock help? (Q119)
# Run as root on the rig, from this folder, with nothing else running (~15 min).
#
# Q118 found ~0.23 major page faults per generated token on the first pass over new text: the token-embedding rows
# stay on the host and are read through mmap from the model stick. For each mode (default mmap, --no-mmap, --mlock)
# this drops the page cache, starts llama-server on all cards (UD-Q4_K_XL, backend sampling, as in Q118), and sends
# the same request twice (1,000 new tokens, fixed seed): the first pass faults in the rows the text needs, the
# repeat shows the warm case. Per request it records tok/s, major and minor faults per token, and the server's
# resident and locked memory; /proc/meminfo is sampled every 2 s through the whole mode, since the host has 4 GB.
#
# What --mlock locks: the loader locks the model mapping from its start to the end of the last tensor kept on the
# host. With every layer on the cards that is token_embd, the third tensor of the UD-Q4_K_XL file, ending at
# 0.61 GiB; the loader then unmaps the range before it (the output head, which lives on a card), leaving ~0.3 GiB
# locked. The mode is skipped if MemAvailable is below MLOCK_MIN_MB before the start.
#
# Env [defaults]: B [v2 base build bin], M [UD-Q4_K_XL], E [LLAMA_NO_MOE_DOWN_RESCALE=tg], PORT [8089], N [1000],
#   MODES [mmap nommap mlock], RUNS [2], T [2] (server threads), MLOCK_MIN_MB [1200], OUT [out-q119/ next to this script]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
B=${B:-/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin}
M=${M:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
E=${E:-LLAMA_NO_MOE_DOWN_RESCALE=tg}
PORT=${PORT:-8089}
N=${N:-1000}
MODES=${MODES:-mmap nommap mlock}
RUNS=${RUNS:-2}
T=${T:-2}
MLOCK_MIN_MB=${MLOCK_MIN_MB:-1200}
OUT=${OUT:-$here/out-q119}
mkdir -p "$OUT"
LOG=$OUT/q119.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
mem_avail_mb() { awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo; }
server_pid() { pgrep -f -- "llama-server .*--port $PORT" | head -1; }
stop_server() {
    local p
    p=$(server_pid); [ -n "$p" ] && kill "$p" 2>/dev/null
    for _ in $(seq 1 30); do [ -z "$(server_pid)" ] && return; sleep 1; done
}

SAMPLER=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
    stop_server
    exit "$rc"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

[ "$(id -u)" = 0 ] || { echo "run as root (drop_caches, /proc of the server)"; exit 2; }
if pgrep -x llama-server > /dev/null || [ -n "$(server_pid)" ]; then echo "a llama-server is already running"; exit 2; fi

sample_mem() {  # sample_mem FILE: /proc/meminfo every 2 s (MiB)
    echo "time,mem_available,mem_free,cached,mlocked,swap_free,dirty" > "$1"
    while :; do
        awk -v t="$(date +%T)" '/^(MemAvailable|MemFree|Cached|Mlocked|SwapFree|Dirty):/ { v[$1] = int($2 / 1024) }
            END { print t "," v["MemAvailable:"] "," v["MemFree:"] "," v["Cached:"] "," v["Mlocked:"] "," v["SwapFree:"] "," v["Dirty:"] }' \
            /proc/meminfo >> "$1"
        sleep 2
    done
}
proc_stat() { awk '{ print $10, $12 }' "/proc/$1/stat"; }                     # minflt majflt
proc_mem() { awk '/^(VmRSS|VmLck):/ { printf "%s %d MiB ", $1, $2 / 1024 }' "/proc/$1/status"; }

echo "mode,run,tokens,tok_per_s,maj_faults,maj_per_token,min_faults,min_per_token,mem_available_before_mb,mem_available_after_mb,server_mem" \
    > "$OUT/q119.csv"
log "Q119: modes $MODES, $RUNS runs of $N tokens each, -t $T, model $M"
for mode in $MODES; do
    case $mode in
    mmap)   flags="" ;;
    nommap) flags="--no-mmap" ;;
    mlock)  flags="--mlock" ;;
    *)      log "unknown mode $mode"; continue ;;
    esac
    sync; echo 3 > /proc/sys/vm/drop_caches
    if [ "$mode" = mlock ] && [ "$(mem_avail_mb)" -lt "$MLOCK_MIN_MB" ]; then
        log "$mode: MemAvailable $(mem_avail_mb) MiB is below $MLOCK_MIN_MB: skipped"; continue
    fi
    sample_mem "$OUT/mem_$mode.csv" &
    SAMPLER=$!
    log "$mode: starting llama-server $flags (MemAvailable $(mem_avail_mb) MiB)"
    env $E setsid nohup "$B/llama-server" -m "$M" -ngl 99 -fa on -c 16384 -np 1 -bs --cache-ram 0 --ctx-checkpoints 0 \
        -t "$T" --no-webui --host 127.0.0.1 --port "$PORT" $flags > "$OUT/server_$mode.log" 2>&1 < /dev/null &
    up=0
    for _ in $(seq 1 120); do
        curl -sf "127.0.0.1:$PORT/health" > /dev/null && { up=1; break; }
        [ -n "$(server_pid)" ] || break
        sleep 5
    done
    P=$(server_pid)
    if [ "$up" != 1 ] || [ -z "$P" ]; then
        log "$mode: the server did not come up (see $OUT/server_$mode.log)"; kill "$SAMPLER"; SAMPLER=""; stop_server; continue
    fi
    log "$mode: loaded; $(proc_mem "$P")MemAvailable $(mem_avail_mb) MiB"
    for run in $(seq 1 "$RUNS"); do
        before=$(proc_stat "$P"); m0=$(mem_avail_mb)
        python3 "$here/client.py" nostream "$PORT" "$N" > "$OUT/run_${mode}_$run.json" 2>> "$LOG"
        after=$(proc_stat "$P"); m1=$(mem_avail_mb); pm=$(proc_mem "$P")
        python3 - "$OUT/run_${mode}_$run.json" "$mode" "$run" "$before" "$after" "$m0" "$m1" "$pm" >> "$OUT/q119.csv" <<'PY'
import json, sys
path, mode, run, before, after, m0, m1, pm = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception:
    d = {}
tok = (d.get("usage") or {}).get("completion_tokens") or 0
tps = (d.get("timings") or {}).get("predicted_per_second") or 0
b, a = [int(x) for x in before.split()], [int(x) for x in after.split()]
maj, mnr = a[1] - b[1], a[0] - b[0]
print(f"{mode},{run},{tok},{tps:.2f},{maj},{maj / tok if tok else 0:.3f},{mnr},{mnr / tok if tok else 0:.2f},{m0},{m1},{pm.strip()}")
PY
        log "$mode run $run: $(tail -n 1 "$OUT/q119.csv")"
        sleep 10
    done
    stop_server
    kill "$SAMPLER" 2>/dev/null; SAMPLER=""
    log "$mode: done; lowest MemAvailable $(awk -F, 'NR > 1 && ($2 < m || m == "") { m = $2 } END { print m }' "$OUT/mem_$mode.csv") MiB"
done
log "Q119 done:"
{ column -s, -t < "$OUT/q119.csv" 2>/dev/null || cat "$OUT/q119.csv"; } | tee -a "$LOG"
