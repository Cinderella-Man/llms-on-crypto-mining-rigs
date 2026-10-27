#!/bin/bash
# Q38/Q49 (2026-10-23), run on the rig after run4.sh: host and GPU timeline of S4-L0 on 6 cards, Pascal kernels on and
# off, with the fork's light host profiler (GGML_CUDA_PROFILE_HOST, tg64). Gives phost-s4l0-mode0/1.csv.
# Env: BIN (fork build at 7d1f070fd), MODEL, OUT.
OUT=${OUT:-/tmp/q38}
while ! grep -q DONE $OUT/run4.log 2>/dev/null; do sleep 10; done
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-7d1f070fd-pascal-p106/bin}
M=${MODEL:-/mnt/models2/flint/flint-S4L0.gguf}
cd "$OUT"
for mode in 1 0; do
  echo "== host prof mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE_HOST=$OUT/phost-s4l0-mode$mode.csv \
    $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 64 -r 1 -o md > $OUT/phost-s4l0-mode$mode.txt 2>&1
  echo "   exit $? $(date +%T)"; grep tg64 $OUT/phost-s4l0-mode$mode.txt
done
echo DONE $(date +%T)
