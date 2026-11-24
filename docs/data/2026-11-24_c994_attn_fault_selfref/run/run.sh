#!/usr/bin/env bash
# run.sh: C994 (Q1727), each attention fault on Ornith-1.5-122B Pascal N2P GGUF scored against its own unfaulted run,
# the form a new-kernel check takes. C989's run.sh with one pass, s, whose reference is the Q8_0 cache itself: it runs
# kld_depth.sh (beside it, C976's driver unchanged) over C976's 12 stretches of 8,192 tokens of the agent text
# (-c 8192 --chunks 12, each chunk scored on its second half), with C976's and C989's settings:
#   s  the Q8_0 cache at -ub 512 (q8) writes the reference, then the Q8_0 cache with the Pascal prompt kernel's fault
#      switch at 1 (q8f1: its output x 1.05) and at 3 (q8f3: each KV split's last key tile dropped), the Q8_0 cache read
#      in micro-batches of 128 (q8u: the rounding yardstick, the same sums in another order), and q8 again (q8r: the
#      repeat, which shows whether the reference run reproduces itself exactly).
# Every arm sets the fork's Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes, and the
# v5 preset's engine variables, as C976's and C989's run.sh do, and logs at -lv 4 (log level only): the log then names
# the attention kernel that took the prompt passes and the cache type it read.
# WORK/DONE at the end; WORK/FAILED if the pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the build's bin), MODEL, TEXT, GPUS, MOUNTS, DOCKER_ARGS, CNAME, PREFLIGHT (a CUDA check of the
#      cards, run before the pass; optional), IMAGE, ARMS (default "q8 q8f1 q8f3 q8u q8r"; the first writes the
#      reference).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export CNAME=${CNAME:-kld-depth}
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
P106="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
Q8="-ctk q8_0 -ctv q8_0 -lv 4"
export ARM_ARGS_q8="$Q8"    ARM_UB_q8=512
export ARM_ARGS_q8f1="$Q8"  ARM_UB_q8f1=512 ARM_ENV_q8f1="GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
export ARM_ARGS_q8f3="$Q8"  ARM_UB_q8f3=512 ARM_ENV_q8f3="GGML_CUDA_FATTN_TILE_PASCAL_FAULT=3"
export ARM_ARGS_q8u="$Q8"   ARM_UB_q8u=128
export ARM_ARGS_q8r="$Q8"   ARM_UB_q8r=512
ARMS=${ARMS:-q8 q8f1 q8f3 q8u q8r}
rm -f "$WORK/DONE" "$WORK/FAILED"
echo "==== pass s $(date -Is)"
if [ -n "${PREFLIGHT:-}" ] && ! "$PREFLIGHT" -g "$GPUS"; then
    echo "preflight failed before pass s"; touch "$WORK/FAILED"
else
    OUT=$WORK/s CHUNKS=12 DEPTHS=8192 ARMS="$ARMS" bash "$HERE/kld_depth.sh"
    [ -f "$WORK/s/failed" ] && touch "$WORK/FAILED"
fi
echo "==== end $(date -Is)"
touch "$WORK/DONE"
