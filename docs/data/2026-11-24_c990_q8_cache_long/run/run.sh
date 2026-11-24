#!/usr/bin/env bash
# run.sh: C990 (Q1723), the quality of a Q8_0 attention cache on Ornith-1.5-122B Pascal N2P GGUF at the longer context
# it allows. Runs kld_depth.sh (beside it: C976's driver, unchanged) once per depth, the longest first, so that its
# reference run is also the fit check; three chunks a depth (-c D --chunks 3: three stretches of text, each scored on
# its second half), four arms, each one llama-perplexity load:
#   f16   the F16 cache, prompt micro-batches of 512 (the v5 preset's): the reference, which writes the log-probabilities
#         every other arm is scored against on the same tokens
#   q8    the Q8_0 cache, micro-batches of 512, with pull request 9's two switches as the agent runs set them
#         (GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4): the cache through the Pascal prompt kernel
#   q8k   the Q8_0 cache with the same switches, one token a pass (-ub 1): the cache through the decode kernel at every
#         position of the chunk
#   f16d  the F16 cache, one token a pass, no switch (as the preset decodes): the decode path's F16 control, so that
#         q8k - f16d is the cache's cost through the decode path
# Every arm sets the fork's Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes, the v5
# preset's engine variables (C976's settings) and -lv 4, so each log names the cache type and the attention kernels.
# If a depth's reference fails (it does not fit), kld_depth.sh skips that depth's other arms and the next depth runs.
# A memory log (host MemAvailable and SwapFree, each card's used memory, every MEMLOG seconds; 0 turns it off) goes to
# WORK/mem.tsv while the run lasts. WORK/DONE at the end; WORK/FAILED if any depth failed. Correctness only: no time is
# read.
# Env: WORK (outputs: base files of up to 73 GB at 98,304 tokens), BIN (the build's bin), MODEL, TEXT, GPUS, MOUNTS,
#      DOCKER_ARGS, PREFLIGHT (a CUDA check of the cards, run before each depth; optional), IMAGE, DEPTHS (default
#      "98304 65536"), CHUNKS (3), ARMS (default "f16 q8 q8k f16d"), MEMLOG (30).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DEPTHS=${DEPTHS:-98304 65536}
export CHUNKS=${CHUNKS:-3}
RUN_ARMS=${ARMS:-f16 q8 q8k f16d}
MEMLOG=${MEMLOG:-30}
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -lv 4"
P106="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
F16="-ctk f16 -ctv f16" Q8="-ctk q8_0 -ctv q8_0" PR9="GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
export ARM_ARGS_f16="$F16"  ARM_UB_f16=512
export ARM_ARGS_q8="$Q8"    ARM_UB_q8=512  ARM_ENV_q8="$PR9"
export ARM_ARGS_q8k="$Q8"   ARM_UB_q8k=1   ARM_ENV_q8k="$PR9"
export ARM_ARGS_f16d="$F16" ARM_UB_f16d=1
mkdir -p "$WORK"
rm -f "$WORK/DONE" "$WORK/FAILED"
memlog() {
    [ -s "$WORK/mem.tsv" ] || printf 'time\tmem_available_kib\tswap_free_kib\tgpu_used_mib\n' > "$WORK/mem.tsv"
    while [ ! -f "$WORK/DONE" ]; do
        printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" \
            "$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)" \
            "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr -d ' ' | paste -sd,)" \
            >> "$WORK/mem.tsv"
        sleep "$MEMLOG"
    done
}
[ "$MEMLOG" = 0 ] || memlog &
for d in $DEPTHS; do
    echo "==== depth $d $(date -Is)"
    if [ -n "${PREFLIGHT:-}" ]; then
        "$PREFLIGHT" -g "$GPUS" || { echo "preflight failed before depth $d"; touch "$WORK/FAILED"; break; }
    fi
    OUT=$WORK/p DEPTHS=$d ARMS="$RUN_ARMS" bash "$HERE/kld_depth.sh"
    [ -f "$WORK/p/failed" ] && { echo "depth $d: a run failed"; touch "$WORK/FAILED"; }
done
echo "==== end $(date -Is)"
touch "$WORK/DONE"
wait
