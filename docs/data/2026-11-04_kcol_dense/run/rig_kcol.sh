#!/usr/bin/env bash
# C70 / Q591 on one P106-090 of the rig: (1) correctness of the k-column kernels (mmvq-pascal-kcol.cu) and of the BF16
# kernel at 1 to 8 tokens through test-backend-ops (the P106 against the rig's CPU), in every launch-shape setting;
# (2) their kernel times at Qwen3.8-Flash-Next's dense shapes (bench/mmvq_pascal build/mmvq_kcol); (3) the whole
# product path on one binary, the new kernels on and off (test-backend-ops perf). Nothing else may run on the rig.
# Env: BIN (the build's bin folder on the rig), MMVQ (mmvq_kcol on the rig), OUT (output folder on the rig), CARD
# (default 0), PARTS (default "check bench perf"; "bf16" adds the BF16 kernel's launch-shape sweep with MMVQ_PASCAL,
# bench/mmvq_pascal's build/mmvq_pascal), BENCH_ARGS (extra mmvq_kcol arguments, e.g. "--check-only --nsm 5" for a
# correctness run through the compute 6.1 PTX on another GPU, with GGML_CUDA_MMVQ_PASCAL=2, GGML_CUDA_MMVF_PASCAL=2,
# CUDA_FORCE_PTX_JIT=1 and the _NSM=5 overrides in the environment).
# Usage: ssh $RIG "BIN=<build>/bin MMVQ=<dir>/mmvq_kcol OUT=<dir>/out setsid nohup bash -s" < run/rig_kcol.sh
set -uo pipefail
BIN=${BIN:?BIN=<build bin folder>}
MMVQ=${MMVQ:?MMVQ=<mmvq_kcol binary>}
OUT=${OUT:?OUT=<output folder>}
CARD=${CARD:-0}
PARTS=${PARTS:-check bench perf}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
log() { echo "$*" | tee -a "$OUT/console.txt"; }
quiet() {
    log "== $(date -Is) $1: $(pgrep -c -f 'llama-|test-backend-ops|mmvq_' || true) GPU processes, load $(cut -d' ' -f1 /proc/loadavg)," \
        "card $CARD $(nvidia-smi --id="$CARD" --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader)"
}
# the k-column kernels' types at 1 to 8 tokens (test_mul_mat and test_mul_mat_large_act cases)
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs),(type_b=f32,)?m=[0-9]+,n=[1-8],'
tbo() { # <tag> <env> <test-backend-ops args...>
    local tag=$1 envs=$2
    shift 2
    quiet "tbo $tag"
    env $envs "$BIN/test-backend-ops" -b CUDA0 "$@" > "$OUT/tbo_$tag.log" 2>&1
    log "$tag: $(grep -E 'tests passed' "$OUT/tbo_$tag.log" | tail -1), $(grep -c FAIL "$OUT/tbo_$tag.log") failed"
}

if [[ " $PARTS " == *" check "* ]]; then
    # default: 2 to 8 tokens, Q3_K and short-row Q8_0 at one token; then every product of 1 to 8 tokens at the default
    # shape, with R = 1 and R = 2 forced, with one iteration per activation tile (several tiles per row: the K-tile
    # loop), and the control with the k-column kernels off (the earlier kernels must pass the same cases)
    tbo kcol      ""                                                       -o MUL_MAT -p "$KT"
    tbo kcol_all  "GGML_CUDA_MMVQ_PASCAL_KCOL=2"                           -o MUL_MAT -p "$KT"
    tbo kcol_r1   "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=1"   -o MUL_MAT -p "$KT"
    tbo kcol_r2   "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=2"   -o MUL_MAT -p "$KT"
    tbo kcol_itt1 "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_ITT=1" -o MUL_MAT -p "$KT"
    tbo kcol_off  "GGML_CUDA_MMVQ_PASCAL_KCOL=0"                           -o MUL_MAT -p "$KT"
    # BF16 at 1 to 8 tokens, the default shapes and R = 8 forced at G = 8 and G = 128
    tbo bf16      ""                                   -o MUL_MAT,MUL_MAT_VEC_FUSION -p 'type_a=bf16|type=bf16'
    tbo bf16_g8r8 "GGML_CUDA_MMVF_PASCAL_SHAPE=8,8"     -o MUL_MAT -p 'type_a=bf16'
    tbo bf16_g128r8 "GGML_CUDA_MMVF_PASCAL_SHAPE=128,8" -o MUL_MAT -p 'type_a=bf16'
fi

if [[ " $PARTS " == *" bench "* ]]; then
    quiet "mmvq_kcol"
    "$MMVQ" ${BENCH_ARGS:-} --csv "$OUT/kcol.csv" > "$OUT/kcol.txt" 2>&1
    log "mmvq_kcol: $(tail -1 "$OUT/kcol.txt")"
fi

if [[ " $PARTS " == *" bf16 "* ]]; then
    # the BF16 kernel at 1 to 8 tokens and every launch shape (gG_rR), against llama.cpp's mul_mat_vec_f
    quiet "mmvq_pascal --bf16"
    "${MMVQ_PASCAL:?MMVQ_PASCAL=<mmvq_pascal binary>}" --bf16 --tokens 1,2,3,4,5,6,7,8 --csv "$OUT/bf16.csv" > "$OUT/bf16.txt" 2>&1
    log "mmvq_pascal --bf16: $(tail -1 "$OUT/bf16.txt")"
fi

if [[ " $PARTS " == *" perf "* ]]; then
    # Flash-Next's dense products at 1 to 8 tokens through ggml (quantizer + product): the new kernels off and on
    PT='type_a=(q8_0|q3_K|iq4_xs|q4_K|q2_0|q5_K|q6_K),type_b=f32,m=(10240|320|6144|12288|2560|248320),n=[1-8],k=(320|10240|2560|6144|640),'
    for arm in off on; do
        quiet "perf kcol $arm"
        env $([ $arm = off ] && echo GGML_CUDA_MMVQ_PASCAL_KCOL=0) "$BIN/test-backend-ops" -b CUDA0 perf -o MUL_MAT -p "$PT" \
            > "$OUT/perf_kcol_$arm.log" 2>&1
    done
    # BF16: llama.cpp's mul_mat_vec_f against the Pascal kernel
    for arm in off on; do
        quiet "perf bf16 $arm"
        env $([ $arm = off ] && echo GGML_CUDA_MMVF_PASCAL=0) "$BIN/test-backend-ops" -b CUDA0 perf -o MUL_MAT -p 'type_a=bf16,' \
            > "$OUT/perf_bf16_$arm.log" 2>&1
    done
    log "perf: done"
fi
quiet "end"
touch "$OUT/done"
