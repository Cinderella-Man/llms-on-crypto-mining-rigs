#!/usr/bin/env bash
# Times Flint's matrix products at 1 to 4 tokens per step (one token, and a speculative-decoding verify step) on one
# card, with the fork's test-backend-ops in perf mode (branch pascal-p106-v2-multicol or later, built with TESTS=1).
# Three runs of the same cases:
#   off    GGML_CUDA_MMVQ_PASCAL=0: llama.cpp's own kernels (mul_mat_vec_q, and mul_mat_vec_q_moe for the experts)
#   on     the Pascal kernels (single-token kernel at 1 token, multi-token kernel at 2-4)
#   stage  the Pascal kernels with GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1 (activations staged through shared memory)
# The cases (make_test_cases_perf in tests/test-backend-ops.cpp): dense products at Flint's shapes (attention q/k/v and
# output, shared expert; the 150,144-row output head for Q8_0 only) and the expert products (gate or up with one input
# per token, down with one input per expert), for Q8_0, Q5_0, Q4_1, Q4_0, Q5_K and IQ4_XS. Each token picks 6 of 48
# random experts, so 2 and 4 tokens touch 11.2 and 19.9 distinct experts, as Flint's own routing does (11.1, 20.1).
# Fused products are not timed here: perf mode times only a test's last node.
#
#   ssh rig "BIN=/opt/llama-builds/<build>/bin bash -s" < bench/mmvq_pascal/multitoken_perf.sh
#   scp -r rig:/tmp/mmvq_multitoken . && bench/mmvq_pascal/multitoken_perf_table.py mmvq_multitoken
#
# Env: BIN (required), GPU (CUDA_VISIBLE_DEVICES, default 0), OUT (default /tmp/mmvq_multitoken), TYPES (regex of the
# weight types, default all six), MODE_ON (GGML_CUDA_MMVQ_PASCAL for "on" and "stage": default 1, compute 6.1 only;
# 2 forces the kernels on any NVIDIA GPU), RUNS (which of off, on, stage to run; default all three).
# Most of the time goes into initializing the random weights on the host. Nothing else should run meanwhile.
set -euo pipefail

BIN=${BIN:?set BIN}
GPU=${GPU:-0}
OUT=${OUT:-/tmp/mmvq_multitoken}
MODE_ON=${MODE_ON:-1}
T=${TYPES:-'(q8_0|q5_0|q4_1|q4_0|q5_K|iq4_xs)'}
RUNS=${RUNS:-off on stage}
mkdir -p "$OUT"

# op|filter on the case's parameters
CASES=(
    "MUL_MAT|type_a=$T,type_b=f32,m=(4096|512|2048|1024|150144),n=[1-4],k=(1024|2048|4096),bs=\[1,1\]"
    "MUL_MAT_ID|type_a=$T,type_b=f32,n_mats=48,n_used=6,"
)

run() { # <label> <env assignments...>
    local label=$1
    shift
    echo "== $label ($*) $(date -Is)"
    for c in "${CASES[@]}"; do
        env "$@" CUDA_VISIBLE_DEVICES="$GPU" "$BIN/test-backend-ops" perf -b CUDA0 -o "${c%%|*}" -p "${c#*|}"
    done > "$OUT/$label.log" 2>&1
    grep -c "us/run" "$OUT/$label.log" | sed "s/^/   cases timed: /"
}

nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.mem --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true
for r in $RUNS; do
    case $r in
    off)   run off   GGML_CUDA_MMVQ_PASCAL=0 ;;
    on)    run on    GGML_CUDA_MMVQ_PASCAL="$MODE_ON" ;;
    stage) run stage GGML_CUDA_MMVQ_PASCAL="$MODE_ON" GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1 ;;
    *)     echo "unknown run $r" >&2; exit 1 ;;
    esac
done
echo "logs in $OUT"
