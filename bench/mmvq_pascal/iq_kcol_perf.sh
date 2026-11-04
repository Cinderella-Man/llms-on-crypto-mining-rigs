#!/usr/bin/env bash
# Times the grid types' and IQ1_M's products at Qwen3.8-Flash-Next's shapes on one card at 1 to 8 tokens (a k-token
# verify pass), with the fork's test-backend-ops (branch pascal-p106-v3-b12076-iqkcol or later, built with TESTS=1): the
# k-column and multi-token kernels of Q713 against the kernels before them, on one binary:
#   off  GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP and GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP list the types: dense products of 2 to 4
#        tokens on mul_mat_vec_q_pascal_mc, 5 to 8 on llama.cpp's mul_mat_vec_q; experts of 2 to 4 tokens on
#        mul_mat_vec_q_pascal_mc, 5 to 8 (IQ1_M 7 to 8) on llama.cpp's matrix-matrix kernels (mul_mat_q); one token on
#        the one-token Pascal kernels (IQ1_M: llama.cpp's), as in the build of branch -iq2xs
#   on   the k-column kernels (dense, 2 to 8 tokens) and the multi-token expert kernels (2 to 8 tokens)
# The cases (make_test_cases_perf in tests/test-backend-ops.cpp): one routed gate or up projection (2560 -> 640, 10 of
# 128 random experts per token) and the dense attn_qkv, attn_q, attn_gate (2560 -> 10240, 12288, 6144) and
# attn_output / ssm_out (6144 -> 2560) in perf mode; the fused routed gate/up + SwiGLU under GGML_CUDA_PROFILE (perf
# mode repeats only a graph's last node), summed per graph evaluation over the launches that start with MUL_MAT_ID, the
# median over the evaluations. Weights are random bytes (GGML_TEST_RAW_QUANT=1: valid blocks for these types; perf mode
# checks no values). Every call reads 4 MB or more, beyond a P106's 1.5 MB L2.
#
#   ssh rig "BIN=/opt/llama-builds/<build>/bin OUT=/tmp/iq_kcol_perf bash -s" < bench/mmvq_pascal/iq_kcol_perf.sh
#   then: python3 -I bench/mmvq_pascal/iq_kcol_perf_table.py <OUT> [--layout <iq_layout.py output> ...] > iq_kcol_perf.csv
#
# Env: BIN (required), GPU (CUDA_VISIBLE_DEVICES, default 0), OUT (default /tmp/iq_kcol_perf), TYPES (default the six),
# TOKENS (default "1 2 3 4 5 6 7 8"), RUNS (default "off on"), MODE_ON (GGML_CUDA_MMVQ_PASCAL for both runs: 1, compute
# 6.1 only; 2 forces the Pascal kernels on any NVIDIA GPU, whose timings mean nothing for a P106). ~25 minutes on a P106;
# nothing else may run on the card meanwhile.
set -euo pipefail
BIN=${BIN:?set BIN}
GPU=${GPU:-0}
OUT=${OUT:-/tmp/iq_kcol_perf}
TYPES=${TYPES:-iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs iq1_m}
TOKENS=${TOKENS:-1 2 3 4 5 6 7 8}
RUNS=${RUNS:-off on}
MODE_ON=${MODE_ON:-1}
mkdir -p "$OUT"
re=$(echo $TYPES | tr ' ' '|')
nre=$(echo $TOKENS | tr ' ' '|')
list=$(echo $TYPES | tr ' ' ',')
export CUDA_VISIBLE_DEVICES=$GPU GGML_TEST_RAW_QUANT=1 GGML_CUDA_MMVQ_PASCAL=$MODE_ON
nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.mem --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true
for r in $RUNS; do
    case $r in
        off) skip=$list ;;
        on)  skip="" ;;
        *)   echo "unknown run $r" >&2; exit 1 ;;
    esac
    echo "== $r $(date -Is)"
    export GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=$skip GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=$skip
    {
        "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT_ID \
            -p "type_a=($re),type_b=f32,n_mats=128,n_used=10,b=1,m=640,n=($nre),k=2560"
        "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT \
            -p "type_a=($re),type_b=f32,m=(10240|12288|6144|2560),n=($nre),k=(2560|6144),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,"
    } > "$OUT/$r.log" 2>&1
    echo "   cases timed: $(grep -c 'us/run' "$OUT/$r.log")"
    : > "$OUT/$r-fused.txt"
    for t in $TYPES; do
        for m in $TOKENS; do
            rm -f "$OUT/prof.csv"
            GGML_CUDA_PROFILE="$OUT/prof.csv" "$BIN/test-backend-ops" perf -b CUDA0 \
                -o MUL_MAT_VEC_FUSION -p "type=$t,.*m=$m,n=640,k=2560,use_id=1,n_mats=128" > /dev/null 2>&1 || true
            med=$(awk -F, '$5 ~ /^MUL_MAT_ID/ { s[$2] += $6 } END { for (e in s) print s[e] }' "$OUT/prof.csv" 2>/dev/null \
                | sort -n | awk '{ a[NR] = $1 } END { if (NR) printf "%.2f %d", a[int((NR + 1) / 2)], NR; else print "- 0" }')
            lab=$(awk -F, '$5 ~ /^MUL_MAT_ID/ { l[$5] = 1 } END { for (k in l) printf "%s; ", k }' "$OUT/prof.csv" 2>/dev/null)
            echo "$t $m $med $lab" >> "$OUT/$r-fused.txt"
        done
    done
    rm -f "$OUT/prof.csv"
    echo "   fused cases: $(wc -l < "$OUT/$r-fused.txt")"
done
unset GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP
echo "logs in $OUT"
