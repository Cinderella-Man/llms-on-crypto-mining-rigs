#!/bin/bash
# Q38 (2026-10-23), run on the rig: one-token test-backend-ops cases on one P106-090, then Miner S4-L0 on 6 cards
# profiled (GGML_CUDA_PROFILE, tg32) and benchmarked (scripts/bench.sh, tg128) with the Pascal kernels off and on.
# Env: BIN (fork build at f254fc7f6), MODEL (S4-L0 GGUF), OUT (work dir), BENCH (a copy of scripts/bench.sh on the rig).
# Started as `nohup bash run2.sh > $OUT/run2.log 2>&1 &`; run3.sh-run6.sh wait for the DONE line of the previous log.
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-f254fc7f6-pascal-p106/bin}
M=${MODEL:-/mnt/models2/flint/flint-S4L0.gguf}
OUT=${OUT:-/tmp/q38}
BENCH=${BENCH:-$OUT/bench.sh}
mkdir -p "$OUT"; cd "$OUT"
echo "== tests $(date +%T)"
for t in q8_0 q4_0 q4_1; do
  CUDA_VISIBLE_DEVICES=0 $B/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ADD,MUL_MAT_ID,MUL_MAT_ID_SHARED -p "type_a=$t,.*,n=1," > tbo-$t.txt 2>&1
  CUDA_VISIBLE_DEVICES=0 $B/test-backend-ops -b CUDA0 -o MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION -p "type=$t,.*m=1," > tbo-fusion-$t.txt 2>&1
  grep -h "tests passed" tbo-$t.txt tbo-fusion-$t.txt
done
for mode in 0 1; do
  echo "== prof mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUT/prof-s4l0-mode$mode.csv \
    $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 32 -r 1 -o md > $OUT/prof-s4l0-mode$mode.txt 2>&1
  echo "   exit $? $(date +%T)"; grep tg32 $OUT/prof-s4l0-mode$mode.txt
done
for mode in 1 0; do
  echo "== tg128 mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M TAG=q38-s4l0-mode$mode GPU_SETS=0,1,2,3,4,5 BENCH_ARGS="-ngl 99 -fa on -p 512 -n 128 -r 3" bash $BENCH
done
echo DONE $(date +%T)
