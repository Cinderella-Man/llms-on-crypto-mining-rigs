#!/usr/bin/env bash
# Measures what limits token generation for small models on one GPU. For each model:
#   1. a verbose 4-token run records the compute-graph size (nodes) and whether CUDA graphs get disabled,
#   2. generation with an idle CPU: 1 vs 2 threads, flash attention off/on, empty cache and DEPTHS of context,
#   3. generation while busy loops occupy one CPU core, then both cores.
# Results go to /tmp/launch_probe/. Run as root or any user that can read the models.
#
#   MODELS="/mnt/models/a.gguf /mnt/models/b.gguf" GPU=0 bash scripts/launch_overhead_probe.sh
#
# DEPTHS defaults to 0,4096. Use DEPTHS=0,1024 for dense 7B models: an f16 KV cache for 4096 tokens doesn't fit
# next to their weights on a 6 GB card, and llama-bench aborts the whole run at the first test that fails.
set -u

BIN=${BIN:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin}
GPU=${GPU:-0}
DEPTHS=${DEPTHS:-0,4096}
MODELS=${MODELS:?set MODELS to one or more GGUF paths}
OUT=/tmp/launch_probe
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU

burn() {  # burn <core>: busy loop pinned to one core, self-limited to 15 minutes
    timeout 900 taskset -c "$1" bash -c 'while :; do :; done' &
}

bench() {  # bench <tag> <name> <model> <llama-bench args...>
    local tag=$1 name=$2 model=$3; shift 3
    "$BIN/llama-bench" -m "$model" -ngl 99 -p 0 -r 3 "$@" -o jsonl -oe md \
        > "$OUT/$name.$tag.jsonl" 2> "$OUT/$name.$tag.log"
    grep -E "^\| [a-zA-Z]" "$OUT/$name.$tag.log" | grep -v "^| model" | sed "s/^/[$tag] /"
}

start=$(date -Is)
nvidia-smi -i "$GPU" --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
    --format=csv -lms 250 > "$OUT/smi.csv" &
smi=$!

for model in $MODELS; do
    name=$(basename "$model" .gguf)
    echo "== $name $(date +%T)"

    timeout 300 "$BIN/llama-completion" -m "$model" -ngl 99 -n 4 -p "Hello" -no-cnv -v < /dev/null \
        > "$OUT/$name.verbose.log" 2>&1
    grep -E "graph nodes|graph splits|disabling CUDA graphs" "$OUT/$name.verbose.log" | sort -u -k4 | sed "s/^/[verbose] /"

    bench idle "$name" "$model" -n 64 -d "$DEPTHS" -t 1,2 -fa off,on

    burn 1; b1=$!
    bench busy1 "$name" "$model" -n 64 -d 0 -t 2 -fa on
    kill $b1

    burn 0; b0=$!; burn 1; b1=$!
    bench busy2 "$name" "$model" -n 64 -d 0 -t 2 -fa on
    kill $b0 $b1
done

kill $smi
pkill -f "while :; do :; done" 2>/dev/null
journalctl -k --since "$start" --no-pager | grep -i "xid" > "$OUT/xid.txt"
echo "== done $(date +%T); Xid lines: $(wc -l < "$OUT/xid.txt")"
