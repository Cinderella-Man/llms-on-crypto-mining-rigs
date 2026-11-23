#!/usr/bin/env bash
# run.sh: C976 (Q1707), the quality of a Q8_0 cache on Ornith-1.5-122B Pascal N2P GGUF. Runs kld_depth.sh (beside it)
# in five passes, in this order, each in its own OUT folder under WORK:
#   a  both caches scored against Ornith-1.5-122B Q8_0's own reference on the agent text (12 chunks of 8,192 tokens,
#      AGENT_BASE): Q62's method, the quality bar's comparison;
#   b  the Q8_0 cache against the F16 cache at 8,192 and 49,152 tokens (prompt micro-batches of 512, the preset's), with
#      a second F16 load (the floor) and F16 read in micro-batches of 128 (the rounding yardstick);
#   b2 the Q8_0 cache with the Pascal prompt kernel's output scaled by 1.05 (its fault switch) at 8,192: the check sees a
#      kernel error, and the kernel read the Q8_0 cache;
#   c  the decode path at 8,192: every token its own pass (-ub 1), the F16 cache as reference, the Q8_0 cache through the
#      base's kernels, F16 in micro-batches of 512 (the yardstick), and the Q8_0 cache through pull request 9's decode
#      kernel (BIN_PR9, GGML_CUDA_FATTN_DEC_PASCAL=4);
#   d  pass b's three arms at 16,384 and 32,768.
# Three passes added after d, not in the default list (run as PASSES="e v", then PASSES=f, then PASSES=v again with
# its -b 512):
#   e  pass b's three arms on six chunks of 16,384 tokens (the text's first 98,304 tokens), so each chunk is its own
#      repeat: d's perplexity rise at 16,384 tested across independent stretches of text;
#   v  one chunk of 512 tokens, one token a pass, with -lv 4: the logs show the cache's type and pull request 9's decode
#      kernel's one-time line, which the default log level hides (F16 the reference, then the decode kernel's arm).
#      With -b 512, one sequence a pass: at -b 4096 llama-perplexity reads eight 512-token chunks at once, and the
#      scheduler's reallocation guard (GGML_SCHED_DEBUG_REALLOC=1) stops the warm-up of that layout. The same window
#      through the base's decode path (q8vd) and as one prompt batch (f16vp) gives the decode kernel's KLD a scale.
#   f  the F16 and Q8_0 caches on four chunks of 32,768 tokens (the first 131,072 tokens): the perplexity rises of d at
#      32,768 and b at 49,152 tested across stretches of text with a long context.
# Every arm sets the fork's Pascal kernels on as a P106 runs them (=2: any GPU) with a P106-090's launch shapes, and the
# v5 preset's engine variables. WORK/DONE at the end; WORK/FAILED if any pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the base build's bin), BIN_PR9 (pull request 9's), MODEL, TEXT, AGENT_BASE, GPUS, MOUNTS, DOCKER_ARGS,
#      PREFLIGHT (a CUDA check of the cards, run before each pass; optional), IMAGE, PASSES (default "a b b2 c d").
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${BIN_PR9:?}" "${MODEL:?}" "${TEXT:?}" "${AGENT_BASE:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PASSES=${PASSES:-a b b2 c d}
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
P106="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5"
P106="$P106 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
export PRESET_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 $P106"
F16="-ctk f16 -ctv f16" Q8="-ctk q8_0 -ctv q8_0"
export ARM_ARGS_f16="$F16"  ARM_UB_f16=512
export ARM_ARGS_f16b="$F16" ARM_UB_f16b=512
export ARM_ARGS_f16u="$F16" ARM_UB_f16u=128
export ARM_ARGS_q8="$Q8"    ARM_UB_q8=512
export ARM_ARGS_q8f="$Q8"   ARM_UB_q8f=512  ARM_ENV_q8f="GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
export ARM_ARGS_f16d="$F16" ARM_UB_f16d=1
export ARM_ARGS_q8d="$Q8"   ARM_UB_q8d=1
export ARM_ARGS_f16p="$F16" ARM_UB_f16p=512
export ARM_ARGS_q8k="$Q8"   ARM_UB_q8k=1    ARM_BIN_q8k="$BIN_PR9"
export ARM_ENV_q8k="GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
export ARM_ARGS_f16v="$F16 -lv 4" ARM_UB_f16v=1
export ARM_ARGS_q8kv="$Q8 -lv 4"  ARM_UB_q8kv=1 ARM_BIN_q8kv="$BIN_PR9" ARM_ENV_q8kv="$ARM_ENV_q8k"
export ARM_ARGS_q8vd="$Q8"   ARM_UB_q8vd=1
export ARM_ARGS_f16vp="$F16" ARM_UB_f16vp=512
rm -f "$WORK/DONE" "$WORK/FAILED"
for p in $PASSES; do
    echo "==== pass $p $(date -Is)"
    if [ -n "${PREFLIGHT:-}" ]; then
        "$PREFLIGHT" -g "$GPUS" || { echo "preflight failed before pass $p"; touch "$WORK/FAILED"; break; }
    fi
    case $p in
        a)  OUT=$WORK/a BASE_FILE=$AGENT_BASE CHUNKS=12 DEPTHS=8192 ARMS="f16 q8" bash "$HERE/kld_depth.sh" ;;
        b)  OUT=$WORK/b DEPTHS="8192 49152" ARMS="f16 q8 f16u f16b" bash "$HERE/kld_depth.sh" ;;
        b2) OUT=$WORK/b DEPTHS=8192 ARMS="f16 q8f" bash "$HERE/kld_depth.sh" ;;
        c)  OUT=$WORK/c DEPTHS=8192 ARMS="f16d q8d f16p q8k" bash "$HERE/kld_depth.sh" ;;
        d)  OUT=$WORK/b DEPTHS="16384 32768" ARMS="f16 q8 f16u" bash "$HERE/kld_depth.sh" ;;
        e)  OUT=$WORK/e CHUNKS=6 DEPTHS=16384 ARMS="f16 q8 f16u" bash "$HERE/kld_depth.sh" ;;
        v)  OUT=$WORK/v BATCH=512 DEPTHS=512 ARMS="f16v q8kv q8vd f16vp" bash "$HERE/kld_depth.sh" ;;
        f)  OUT=$WORK/f CHUNKS=4 DEPTHS=32768 ARMS="f16 q8" bash "$HERE/kld_depth.sh" ;;
        *)  echo "unknown pass $p"; touch "$WORK/FAILED"; continue ;;
    esac
    case $p in a|c|e|v|f) o=$p ;; *) o=b ;; esac
    [ -f "$WORK/$o/failed" ] && touch "$WORK/FAILED"
done
echo "==== end $(date -Is)"
touch "$WORK/DONE"
