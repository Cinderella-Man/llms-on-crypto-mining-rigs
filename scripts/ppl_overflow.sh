#!/usr/bin/env bash
# Real-data overflow check for the fork's Pascal matrix-vector kernels without a Pascal card: the last 4 layers of a
# model (for Flint 45-48, whose routed down-projections carry the largest activations) on a newer NVIDIA GPU with the
# Pascal kernels forced (GGML_CUDA_MMVQ_PASCAL=2), the rest on the CPU; one 512-token chunk of wikitext-2. Prints the
# chunk's perplexity; "nan" means a 16-bit sum overflowed. Used for the FP32 activation sums (registry row Q76,
# 2026-10-23): with and without the MoE down-projection rescale, one token per ubatch (-ub 1, the one-token kernels) and
# full batches (-ub 512).
#
#   scripts/ppl_overflow.sh <llama.cpp dir> <model.gguf> <ub> <LLAMA_NO_MOE_DOWN_RESCALE value, or "unset"> <tag>
#   scripts/ppl_overflow.sh ~/src/llama.cpp flint-int4-q4_1.gguf 1 unset q41-on-ub1
#   scripts/ppl_overflow.sh ~/src/llama.cpp flint-int4-q4_1.gguf 1 tg q41-tg-ub1
#
# The build must contain sm_61 or compute_61 code and the GPU's own (e.g. CMAKE_CUDA_ARCHITECTURES="61;120").
# Env: TEXT (wikitext-2 test file, default <llama.cpp dir>/wikitext-2-raw/wiki.test.raw), OUT (log folder, default
# /tmp/ppl_overflow), THREADS (default: all cores), NGL (GPU layers, default 4), LD_LIBRARY_PATH as the build needs.
set -uo pipefail
BUILD=$1 MODEL=$2 UB=$3 RESCALE=$4 TAG=$5
TEXT=${TEXT:-$BUILD/wikitext-2-raw/wiki.test.raw}
OUT=${OUT:-/tmp/ppl_overflow}
THREADS=${THREADS:-$(nproc)}
mkdir -p "$OUT"
export GGML_CUDA_MMVQ_PASCAL=2
if [ "$RESCALE" = unset ]; then unset LLAMA_NO_MOE_DOWN_RESCALE; else export LLAMA_NO_MOE_DOWN_RESCALE=$RESCALE; fi
"$BUILD/build/bin/llama-perplexity" -m "$MODEL" -f "$TEXT" \
    -c 512 --chunks 1 -b 512 -ub "$UB" -ngl "${NGL:-4}" -t "$THREADS" > "$OUT/$TAG.log" 2>&1
echo "$TAG: $(grep -E 'Final estimate|^\[1\]' "$OUT/$TAG.log" | tr '\n' ' ')"
