#!/bin/bash
# Q125: the fair release comparison on the combined build. UD-Q4_K_XL with and without the screened head, K2h-r16
# (GPTQ Q5_0 head, no screen), K2-r16 with and without the screen (with = K2sr); 6 then 9 cards. llama-bench runs
# pp512 before tg128 in the same process, so decode starts on warm clocks (Q141). Correctness of the screen first.
# Ran 2026-10-24. Usage: [RIG=rig] [OUT=dir] q125.sh   (writes q125.md; ../llama_bench.csv holds its tables)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
B=/opt/llama-builds/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin
UD=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
K2H=/mnt/models/flint/flint-q113-K2h-r16.gguf
K2=/mnt/models2/flint/flint-q126-K2-r16.gguf
SC=/mnt/models/flint/flint-output-screen-gptq-q4_0.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=tg LLAMA_SCREEN_HEAD_FILE=$SC"
ssh "$RIG" "mkdir -p /tmp/q125 && printf 'Explain how a B-tree splits nodes.\nWrite a Python function that merges two sorted lists.\n' > /tmp/q125/prompts.txt"
{
echo "== screen correctness on UD (P106) $(date +%T)"
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E $B/llama-screen-check -m $UD -f /tmp/q125/prompts.txt -n 64 -ngl 99 -fa on -c 4096 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234 2>&1 | tail -6"
echo "== screen correctness on K2-r16 (P106) $(date +%T)"
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E $B/llama-screen-check -m $K2 -f /tmp/q125/prompts.txt -n 64 -ngl 99 -fa on -c 4096 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234 2>&1 | tail -6"
for cards in 0,1,2,3,4,5 0,1,2,3,4,5,6,7,8; do
  n=$(echo $cards | tr ',' '\n' | wc -l)
  for arm in "UD|$UD|0" "UD+screen|$UD|1" "K2h-r16|$K2H|0" "K2-r16|$K2|0" "K2sr(K2-r16+screen)|$K2|1"; do
    IFS='|' read -r label m s <<< "$arm"
    echo "== $label ${n} cards $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards LLAMA_SCREEN_HEAD=$s $E $B/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 -o md 2>&1" | grep -E 'pp512|tg128|error|fail'
  done
done
ssh "$RIG" "rm -rf /tmp/q125"
echo Q125_DONE
} | tee "$OUT/q125.md"
