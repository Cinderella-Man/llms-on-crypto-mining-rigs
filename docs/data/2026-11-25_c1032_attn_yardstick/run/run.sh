#!/usr/bin/env bash
# run.sh: C1032 (Q1765), a rounding yardstick on Ornith-1.5-122B Pascal N2P GGUF that changes only the attention
# kernel. C994's run.sh (docs/data/2026-11-24_c994_attn_fault_selfref/run/run.sh) with one change of arms: it runs
# kld_depth.sh (beside it, C976's driver unchanged) over C976's 12 stretches of 8,192 tokens of the agent text
# (-c 8192 --chunks 12, each chunk scored on its second half), with C976's, C989's and C994's settings:
#   s  the Q8_0 cache at -ub 512 with the fork's compute 6.1 prompt attention kernel (q8) writes the reference, then the
#      same cache and micro-batch with that kernel off (q8g: GGML_CUDA_FATTN_TILE_PASCAL=0, so the build's generic tile
#      kernel takes the prompt passes, on K/V converted to F16; everything outside the attention is unchanged), and the
#      Pascal prompt kernel's fault switch at 1 again (q8f1: its output x 1.05, C994's 5 % error).
# Every arm sets the fork's other Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes,
# and the v5 preset's engine variables, as C994's run.sh does; the prompt attention kernel's switch is set per arm
# (=2 in q8 and q8f1, =0 in q8g) so that no arm carries the variable twice. Every arm logs at -lv 4 (log level only):
# the fork's prompt kernel then logs one line when it takes a pass, and an arm without that line ran the generic one.
# WORK/DONE at the end; WORK/FAILED if the pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the build's bin), MODEL, TEXT, GPUS, MOUNTS, DOCKER_ARGS, CNAME, PREFLIGHT (a CUDA check of the
#      cards, run before the pass; optional), IMAGE, ARMS (default "q8 q8g q8f1"; the first writes the reference).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export CNAME=${CNAME:-kld-depth}
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
P106="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
Q8="-ctk q8_0 -ctv q8_0 -lv 4"
export ARM_ARGS_q8="$Q8"   ARM_UB_q8=512   ARM_ENV_q8="GGML_CUDA_FATTN_TILE_PASCAL=2"
export ARM_ARGS_q8g="$Q8"  ARM_UB_q8g=512  ARM_ENV_q8g="GGML_CUDA_FATTN_TILE_PASCAL=0"
export ARM_ARGS_q8f1="$Q8" ARM_UB_q8f1=512 ARM_ENV_q8f1="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
ARMS=${ARMS:-q8 q8g q8f1}
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
