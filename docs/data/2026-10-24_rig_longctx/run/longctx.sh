#!/bin/bash
# Long context (Q79, Q110, Q116): K2h-r16 on nine cards, the gpumask2 build (prompt and decode masks built on each
# card), llama-batched-bench at 32k / 64k / 131k, host mask (LLAMA_GPU_KQ_MASK=0) against card mask (1), with host
# memory and swap logged every second (Q79: is the 64k drop host swapping?). No saved state. Ran 2026-10-24.
# Usage: [RIG=rig] [OUT=dir] longctx.sh   (the logs land in $OUT/raw; ../batched_bench.csv and ../memory_swap.csv sum them up)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
B=/opt/llama-builds/llama.cpp-20261024-6f943f792-pascal-p106-v2-gpumask2/bin
M=/mnt/models/flint/flint-q113-K2h-r16.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
ssh "$RIG" "mkdir -p /tmp/lc"
echo "== op test KQ_MASK on GPU0 $(date +%T)"
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $B/test-backend-ops -b CUDA0 -o KQ_MASK 2>&1 | tail -3"
for npp in 32768 65536 131072; do
  c=$((npp + 512)); ub=256; [ $npp -le 32768 ] && ub=512
  for m in 0 1; do
    echo "== npp=$npp ub=$ub mask=$m $(date +%T)"
    ssh "$RIG" "(vmstat 1 900 > /tmp/lc/vmstat_${npp}_$m.txt 2>/dev/null &); (timeout 900 bash -c \"while true; do grep -E 'MemAvailable|SwapFree' /proc/meminfo | tr '\\n' ' '; echo; sleep 1; done\" > /tmp/lc/mem_${npp}_$m.txt 2>/dev/null &); \
      $E LLAMA_GPU_KQ_MASK=$m $B/llama-batched-bench -m $M -ngl 99 -fa on -c $c -b 2048 -ub $ub -npp $npp -ntg 64 -npl 1 2>&1 | tail -4; pkill -x vmstat; pkill -x timeout; true"
  done
done
scp -q -r "$RIG":/tmp/lc "$OUT/raw"; ssh "$RIG" "rm -rf /tmp/lc; pkill -x vmstat; true"
echo LONGCTX_DONE
