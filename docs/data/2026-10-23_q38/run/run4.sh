#!/bin/bash
# Q38 (2026-10-23), run on the rig after run3.sh: tg128 of four more files with the Pascal kernels on (1) and off (0),
# same binary (112cdc597). Env: BIN, OUT, BENCH (as run2.sh); D and D2 (the two model folders, one per USB stick).
OUT=${OUT:-/tmp/q38}
while ! grep -q DONE $OUT/run3.log 2>/dev/null; do sleep 10; done
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-112cdc597-pascal-p106/bin}
BENCH=${BENCH:-$OUT/bench.sh}
D=${D:-/mnt/models}
D2=${D2:-/mnt/models2}
run() { # tag model gpus rescale_off
  for mode in 1 0; do
    echo "== $1 mode $mode $(date +%T)"
    if [ "$4" = 1 ]; then export LLAMA_NO_MOE_DOWN_RESCALE=1; else unset LLAMA_NO_MOE_DOWN_RESCALE; fi
    GGML_CUDA_MMVQ_PASCAL=$mode BIN=$B MODEL=$2 TAG=q38-$1-mode$mode GPU_SETS=$3 BENCH_ARGS="-ngl 99 -fa on -p 512 -n 128 -r 3" bash $BENCH
  done
}
run makers-q41 $D/RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf 0,1,2,3,4 0
run ud-q4kxl $D/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf 0,1,2,3,4,5 0
run qwen36-q40 $D2/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf 0,1,2,3 0
run flint-q80 $D2/unsloth/flint-GGUF/flint-Q8_0.gguf 0,1,2,3,4,5,6,7,8 1
echo DONE $(date +%T)
