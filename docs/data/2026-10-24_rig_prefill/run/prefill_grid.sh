#!/bin/bash
# Prompt-processing grid (Q108/Q109/Q89): which -ub and scheduler-copy count give the fastest prefill on the combined
# build. K2h-r16, pp2048 and pp8192, -b 8192, -ub 128/256/512/1024; 9 then 6 cards; then the 1-copy prefill build
# against the 4-copy one at 9 cards. llama-bench, warm (the first pp run warms the cards). Ran 2026-10-24.
# Usage: [RIG=rig] [OUT=dir] prefill_grid.sh   (writes prefill_grid.md; ../llama_bench.csv holds it)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
C=/opt/llama-builds/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin
P4=/opt/llama-builds/llama.cpp-20261024-82bbb3aa4-pascal-p106-v2-prefill/bin
P1=/opt/llama-builds/llama.cpp-20261024-82bbb3aa4-pascal-p106-v2-prefill-sched1/bin
M=/mnt/models/flint/flint-q113-K2h-r16.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
{
for cards in 0,1,2,3,4,5,6,7,8 0,1,2,3,4,5; do
  n=$(echo $cards | tr ',' '\n' | wc -l)
  echo "== combined, ${n} cards, -ub grid $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards $E $C/llama-bench -m $M -ngl 99 -fa on -p 2048,8192 -n 0 -b 8192 -ub 128,256,512,1024 -r 2 -o md 2>&1" | grep -E 'pp[0-9]+|error|fail|out of memory'
done
for b in "prefill-4copies|$P4" "prefill-1copy|$P1"; do
  IFS='|' read -r label bin <<< "$b"
  echo "== $label, 9 cards $(date +%T)"
  ssh "$RIG" "$E $bin/llama-bench -m $M -ngl 99 -fa on -p 8192 -n 0 -b 8192 -ub 128,512 -r 2 -o md 2>&1" | grep -E 'pp[0-9]+|error|fail|out of memory'
done
echo PREFILL_GRID_DONE
} | tee "$OUT/prefill_grid.md"
