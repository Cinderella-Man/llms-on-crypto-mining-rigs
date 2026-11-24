#!/usr/bin/env bash
# run.sh: C993 (Q1726), the per-stretch quality check's floor on Ornith-1.5-122B Pascal N2P GGUF, measured. C989's pass
# g (docs/data/2026-11-24_c989_attn_error_floor/run/run.sh) with the size of the Pascal prompt kernel's test fault set
# per arm (GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 with GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=<factor>: every output of
# the kernel x factor; the switch is on the fork branch pascal-p106-v5-fault-scale). Runs kld_depth.sh (beside it,
# C976's driver unchanged) once, in WORK/g, over C976's 12 stretches of 8,192 tokens of the agent text
# (-c 8192 --chunks 12, each chunk scored on its second half), with C976's and C989's settings:
#   the F16 cache writes the reference (f16), then the Q8_0 cache (q8), and the Q8_0 cache with the fault at x 1.05
#   (q8f105: C989's size, a repeat of its q8f1), x 1.01 (q8f101), x 1.02 (q8f102) and x 1.03 (q8f103).
# Every arm sets the fork's Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes, and the
# v5 preset's engine variables, as C976's and C989's run.sh do. Every arm logs at -lv 4 (log level only): the log then
# names the attention kernel that took the prompt passes, the cache type it read and, with the fault on, its factor.
# WORK/DONE at the end; WORK/FAILED if the pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the build's bin), MODEL, TEXT, GPUS, MOUNTS, DOCKER_ARGS, CNAME, PREFLIGHT (a CUDA check of the
#      cards, run before the pass; optional), IMAGE, ARMS (default "f16 q8 q8f105 q8f101 q8f102 q8f103"; the first
#      arm writes the reference).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARMS=${ARMS:-f16 q8 q8f105 q8f101 q8f102 q8f103}
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export CNAME=${CNAME:-kld-depth}
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
P106="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
F16="-ctk f16 -ctv f16 -lv 4" Q8="-ctk q8_0 -ctv q8_0 -lv 4"
FAULT=GGML_CUDA_FATTN_TILE_PASCAL_FAULT
export ARM_ARGS_f16="$F16"   ARM_UB_f16=512
export ARM_ARGS_f16u="$F16"  ARM_UB_f16u=128
export ARM_ARGS_q8="$Q8"     ARM_UB_q8=512
for s in 101 102 103 105; do
    export "ARM_ARGS_q8f$s=$Q8" "ARM_UB_q8f$s=512" "ARM_ENV_q8f$s=$FAULT=1 ${FAULT}_SCALE=1.${s#1}"
done
rm -f "$WORK/DONE" "$WORK/FAILED"
echo "==== pass g $(date -Is)"
if [ -n "${PREFLIGHT:-}" ]; then
    "$PREFLIGHT" -g "$GPUS" || { echo "preflight failed before pass g"; touch "$WORK/FAILED" "$WORK/DONE"; exit 1; }
fi
OUT=$WORK/g CHUNKS=12 DEPTHS=8192 ARMS="$ARMS" bash "$HERE/kld_depth.sh"
[ -f "$WORK/g/failed" ] && touch "$WORK/FAILED"
echo "==== end $(date -Is)"
touch "$WORK/DONE"
