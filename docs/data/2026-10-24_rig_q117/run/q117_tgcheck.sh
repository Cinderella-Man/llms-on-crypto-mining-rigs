#!/bin/bash
# Q117 follow-up: is the combined build's decode loss real or an order effect? tg128 only (no prompt runs before it),
# combined with Q108 on and off and the v2 base, alternating, three rounds. UD-Q4_K_XL, 6 cards. Ran 2026-10-24.
# Usage: [RIG=rig] [OUT=dir] q117_tgcheck.sh   (writes tgcheck.md; ../llama_bench.csv holds it)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
R=/opt/llama-builds
M=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
C=$R/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin
V=$R/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin
run() { # $1 label, $2 bin, $3 env
  echo "== $1 $(date +%T) $(ssh "$RIG" "nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | head -6 | tr '\n' ' '")"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $3 $2/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5 -o md" 2>&1 | grep -E 'tg128|error'
}
{
  for round in 1 2 3; do
    run "combined-q108-on r$round" $C "LLAMA_NO_MOE_DOWN_RESCALE=tg"
    run "combined-q108-off r$round" $C "LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_MMQ_MOE_LEGACY=1"
    run "v2-base r$round" $V "LLAMA_NO_MOE_DOWN_RESCALE=tg"
  done
  echo TGCHECK_DONE
} | tee "$OUT/tgcheck.md"
