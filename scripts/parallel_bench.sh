#!/usr/bin/env bash
# Starts one or more llama-server instances on disjoint GPU sets, one after another (loading two at once would
# thrash the 4 GB of host RAM), records memory per card and free host RAM, then measures each server alone and all
# together with scripts/chat_load.py. Servers listen on 127.0.0.1 only and are stopped at the end.
#
#   MODEL=/mnt/models/x.gguf TAG=two_instances \
#   SERVERS="0,1,2,3,4|8089|-c 262144 -b 512 -ub 128;5,6,7,8|8090|-c 131072 -b 512 -ub 128" \
#   bash scripts/parallel_bench.sh
#
#   MODEL=... TAG=two_slots PER_ENDPOINT=2 SERVERS="0,1,2,3,4,5,6,7,8|8089|-np 2 -c 524288 -b 512 -ub 128" bash scripts/parallel_bench.sh
#
# SERVERS is a ';'-separated list of "<CUDA_VISIBLE_DEVICES>|<port>|<extra llama-server args>". Set CHAT_LOAD to the
# path of chat_load.py when running the script through 'ssh ... bash -s' (default: next to this script, or /tmp).
set -u

BIN=${BIN:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:?set TAG}
SERVERS=${SERVERS:?set SERVERS}
PER_ENDPOINT=${PER_ENDPOINT:-1}
CHAT_LOAD=${CHAT_LOAD:-$(dirname "$0")/chat_load.py}
[ -f "$CHAT_LOAD" ] || CHAT_LOAD=/tmp/chat_load.py
OUT=/tmp/parallel_bench/$TAG
mkdir -p "$OUT"

pids=()
endpoints=()
cleanup() { kill "${pids[@]}" 2> /dev/null; wait 2> /dev/null; }
trap cleanup EXIT

# Watchdog: kill all servers if host RAM runs low, instead of letting the kernel's OOM killer take the desktop down.
(
    while sleep 0.5; do
        avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
            echo "watchdog: MemAvailable ${avail} MiB, killing llama-server instances" >&2
            pkill -9 -f "llama-server.*--port 80[0-9][0-9]"; break
        fi
        pgrep -f "llama-server.*--port 80[0-9][0-9]" > /dev/null || [ ! -f "$OUT/.started" ] || break
    done
) &

IFS=';' read -ra specs <<< "$SERVERS"
for i in "${!specs[@]}"; do
    IFS='|' read -r gpus port args <<< "${specs[$i]}"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpus "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$port" \
        --no-webui --cache-ram 0 --ctx-checkpoints 0 $args > "$OUT/server$i.log" 2>&1 &
    pids+=($!)
    touch "$OUT/.started"
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "${pids[$i]}" 2> /dev/null; then echo "server $i (GPUs $gpus) died during load:"; tail -5 "$OUT/server$i.log"; exit 1; fi
        sleep 2
    done
    ctx=$(grep -o "n_ctx_slot = [0-9]*" "$OUT/server$i.log" | head -1 | grep -o "[0-9]*$")
    echo "server $i: GPUs $gpus, port $port, context per slot $ctx, loaded in $(( $(date +%s) - t0 )) s ($args)"
    endpoints+=("127.0.0.1:$port")
done

echo "memory per card (MiB):"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "host MemAvailable: $(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) MiB"

if [ "${#endpoints[@]}" -gt 1 ]; then
    echo "== each server alone"
    python3 "$CHAT_LOAD" --endpoints "${endpoints[@]}" --mode alone
fi
echo "== all together (${PER_ENDPOINT} request(s) per server)"
python3 "$CHAT_LOAD" --endpoints "${endpoints[@]}" --mode together --per-endpoint "$PER_ENDPOINT"
echo "host MemAvailable after load: $(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) MiB"
