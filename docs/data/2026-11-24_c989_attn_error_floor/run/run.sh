#!/usr/bin/env bash
# run.sh: C989 (Q1722), how small an attention-kernel error the per-stretch quality check sees on Ornith-1.5-122B
# Pascal N2P GGUF. Runs kld_depth.sh (beside it, C976's driver unchanged) in two passes, each in its own OUT folder
# under WORK, over C976's 12 stretches of 8,192 tokens of the agent text (-c 8192 --chunks 12, each chunk scored on its
# second half), with C976's settings:
#   g  against the F16 cache: the F16 cache writes the reference, then the Q8_0 cache (q8), the Q8_0 cache with the
#      Pascal prompt kernel's fault switch at 1 (q8f1: its output x 1.05) and at 3 (q8f3: each KV split's last key tile
#      dropped), and the F16 cache read in micro-batches of 128 (f16u: the rounding yardstick);
#   h  against Ornith-1.5-122B Q8_0's own reference on the same stretches (AGENT_BASE, C976's pass a): q8, q8f1, q8f3.
#      C976's pass a holds the F16 cache's run against the same reference.
# Every arm sets the fork's Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes, and the
# v5 preset's engine variables, as C976's run.sh does. Every arm also logs at -lv 4 (log level only): the log then
# names the attention kernel that took the prompt passes and the cache type it read.
# WORK/DONE at the end; WORK/FAILED if any pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the build's bin), MODEL, TEXT, AGENT_BASE, GPUS, MOUNTS, DOCKER_ARGS, CNAME, PREFLIGHT (a CUDA check
#      of the cards, run before each pass; optional), IMAGE, PASSES (default "g h").
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${AGENT_BASE:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PASSES=${PASSES:-g h}
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export CNAME=${CNAME:-kld-depth}
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
P106="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
F16="-ctk f16 -ctv f16 -lv 4" Q8="-ctk q8_0 -ctv q8_0 -lv 4"
export ARM_ARGS_f16="$F16"  ARM_UB_f16=512
export ARM_ARGS_f16u="$F16" ARM_UB_f16u=128
export ARM_ARGS_q8="$Q8"    ARM_UB_q8=512
export ARM_ARGS_q8f1="$Q8"  ARM_UB_q8f1=512 ARM_ENV_q8f1="GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
export ARM_ARGS_q8f3="$Q8"  ARM_UB_q8f3=512 ARM_ENV_q8f3="GGML_CUDA_FATTN_TILE_PASCAL_FAULT=3"
rm -f "$WORK/DONE" "$WORK/FAILED"
for p in $PASSES; do
    echo "==== pass $p $(date -Is)"
    if [ -n "${PREFLIGHT:-}" ]; then
        "$PREFLIGHT" -g "$GPUS" || { echo "preflight failed before pass $p"; touch "$WORK/FAILED"; break; }
    fi
    case $p in
        g)  OUT=$WORK/g CHUNKS=12 DEPTHS=8192 ARMS="f16 q8 q8f1 q8f3 f16u" bash "$HERE/kld_depth.sh" ;;
        h)  OUT=$WORK/h BASE_FILE=$AGENT_BASE CHUNKS=12 DEPTHS=8192 ARMS="q8 q8f1 q8f3" bash "$HERE/kld_depth.sh" ;;
        *)  echo "unknown pass $p"; touch "$WORK/FAILED"; continue ;;
    esac
    [ -f "$WORK/$p/failed" ] && touch "$WORK/FAILED"
done
echo "==== end $(date -Is)"
touch "$WORK/DONE"
