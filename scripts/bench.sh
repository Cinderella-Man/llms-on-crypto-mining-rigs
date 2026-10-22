#!/usr/bin/env bash
# Runs llama-bench once per GPU set while logging every GPU's P-state, clocks, power, temperature, utilisation and
# throttle reasons every 0.5 s. Results go to /tmp/bench/<TAG>/: one JSONL and one markdown table per GPU set,
# plus smi.csv and any GPU (Xid) errors from the kernel log.
#
#   MODEL=/mnt/models/x.gguf TAG=mytest GPU_SETS="0 1 2" bash scripts/bench.sh          # each card on its own
#   MODEL=/mnt/models/x.gguf TAG=split GPU_SETS="0,1 0,1,2,3,4,5,6,7,8" bash scripts/bench.sh   # layer splits
#   ssh root@rig "MODEL=... TAG=... LLAMA_DIR=/path/to/llama.cpp bash -s" < scripts/bench.sh
#
# Each GPU set is passed to CUDA_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES, so "0,1" means a model split across
# devices 0 and 1. A CUDA build reads the first, a Vulkan build the second; Vulkan numbers devices in its own order
# (docs/vulkan_on_the_rig.md).
set -u

BIN=${BIN:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin}
MODEL=${MODEL:?set MODEL}
GPU_SETS=${GPU_SETS:-"0,1,2,3,4,5,6,7,8"}
TAG=${TAG:?set TAG}
BENCH_ARGS=${BENCH_ARGS:-"-ngl 99 -fa on -p 512 -n 128 -r 3"}
OUT=/tmp/bench/$TAG
mkdir -p "$OUT"

start=$(date -Is)
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/smi.csv" &
smi=$!

for set in $GPU_SETS; do
    echo "== GPUs $set $(date +%T)"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$set GGML_VK_VISIBLE_DEVICES=$set "$BIN/llama-bench" -m "$MODEL" $BENCH_ARGS -o jsonl -oe md \
        > "$OUT/gpus_$set.jsonl" 2> "$OUT/gpus_$set.log"
    grep -E "^\| [a-zA-Z]" "$OUT/gpus_$set.log" | grep -v "^| model"
    grep -E "error|failed" "$OUT/gpus_$set.log" | head -3
    echo "   ($(( $(date +%s) - t0 )) s including model load)"
done

kill $smi
journalctl -k --since "$start" --no-pager | grep -i "xid" > "$OUT/xid.txt"
echo "== done $(date +%T); Xid lines: $(wc -l < "$OUT/xid.txt")"
