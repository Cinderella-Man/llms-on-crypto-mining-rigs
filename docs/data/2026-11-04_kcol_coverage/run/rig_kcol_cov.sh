#!/usr/bin/env bash
# Q630, Q611, Q612 on one P106-090 of the rig, after docs/data/2026-11-04_kcol_dense/run/rig_kcol.sh has run on the same
# build (that script's BF16 correctness, launch-shape sweep and perf runs already cover Q611's 3- and 4-token shapes, and
# its mmvq_kcol run times the new types' shapes). This one adds:
#   (1) correctness of the k-column kernels for Q4_0, Q5_0 and IQ4_NL at 1 to 8 tokens through test-backend-ops (the P106
#       against the rig's CPU), in every launch-shape setting, and of the Q2_0 fused gate/up at one token with the
#       small-grid rule (Q612) and with R = 2 forced;
#   (2) the one-token Q2_0 bench at the experts' shapes (bench/mmvq_pascal --q2_0: `pascal` is now R = 1 for the shared
#       expert's gate/up);
#   (3) the whole product path for the three types through ggml, the k-column kernels off and on (test-backend-ops perf).
# Nothing else may run on the rig.
# Env: BIN (the build's bin folder on the rig), MMVQ_PASCAL (bench/mmvq_pascal's build/mmvq_pascal on the rig), OUT
# (output folder on the rig), CARD (default 0), PARTS (default "check q2_0 perf"). For a correctness run through the
# compute 6.1 PTX on another GPU: PARTS="check q2_0" with GGML_CUDA_MMVQ_PASCAL=2, CUDA_FORCE_PTX_JIT=1,
# GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 and GGML_CUDA_MMVQ_PASCAL_NSM=5 in the environment (and Q2_BENCH_ARGS="--check-only").
# Usage: ssh $RIG "BIN=<build>/bin MMVQ_PASCAL=<dir>/mmvq_pascal OUT=<dir>/out setsid nohup bash <dir>/rig_kcol_cov.sh \
#   > <dir>/rig_kcol_cov.out 2>&1 < /dev/null &"
set -uo pipefail
BIN=${BIN:?BIN=<build bin folder>}
OUT=${OUT:?OUT=<output folder>}
CARD=${CARD:-0}
PARTS=${PARTS:-check q2_0 perf}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
log() { echo "$*" | tee -a "$OUT/console.txt"; }
quiet() {
    log "== $(date -Is) $1: $(pgrep -c -f 'llama-|test-backend-ops|mmvq_' || true) GPU processes, load $(cut -d' ' -f1 /proc/loadavg)," \
        "card $CARD $(nvidia-smi --id="$CARD" --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null)"
}
# the new types at 1 to 8 tokens (test_mul_mat and test_mul_mat_large_act cases)
KT='type_a=(q4_0|q5_0|iq4_nl),(type_b=f32,)?m=[0-9]+,n=[1-8],'
tbo() { # <tag> <env> <test-backend-ops args...>
    local tag=$1 envs=$2
    shift 2
    quiet "tbo $tag"
    env $envs "$BIN/test-backend-ops" -b CUDA0 "$@" > "$OUT/tbo_$tag.log" 2>&1
    log "$tag: $(grep -E 'tests passed' "$OUT/tbo_$tag.log" | tail -1), $(grep -c FAIL "$OUT/tbo_$tag.log") failed"
}

if [[ " $PARTS " == *" check "* ]]; then
    # default (2 to 8 tokens, IQ4_NL also at one), every product of 1 to 8 tokens, R = 1 and R = 2 forced, one iteration
    # per activation tile, and the control with the k-column kernels off
    tbo cov       ""                                                              -o MUL_MAT -p "$KT"
    tbo cov_all   "GGML_CUDA_MMVQ_PASCAL_KCOL=2"                                  -o MUL_MAT -p "$KT"
    tbo cov_r1    "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=1"   -o MUL_MAT -p "$KT"
    tbo cov_r2    "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=2"   -o MUL_MAT -p "$KT"
    tbo cov_itt1  "GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_ITT=1" -o MUL_MAT -p "$KT"
    tbo cov_off   "GGML_CUDA_MMVQ_PASCAL_KCOL=0"                                  -o MUL_MAT -p "$KT"
fi

if [[ " $PARTS " == *" q2_0 "* ]]; then
    # Q612: every Q2_0 fused gate/up case with the default shapes (R = 1 below 4 blocks per SM) and with R = 2 forced
    tbo q2f       ""                                  -o MUL_MAT_VEC_FUSION -p 'type=q2_0'
    tbo q2f_r2    "GGML_CUDA_MMVQ_PASCAL_Q2_0=2"      -o MUL_MAT_VEC_FUSION -p 'type=q2_0'
    if [ -n "${MMVQ_PASCAL:-}" ]; then
        quiet "mmvq_pascal --q2_0"
        "$MMVQ_PASCAL" --q2_0 ${Q2_BENCH_ARGS:-} --csv "$OUT/q2_0.csv" > "$OUT/q2_0.txt" 2>&1
        log "mmvq_pascal --q2_0: $(tail -1 "$OUT/q2_0.txt")"
    fi
fi

if [[ " $PARTS " == *" perf "* ]]; then
    # the shared expert's down projection in the three types at 1 to 8 tokens through ggml: k-column kernels off and on
    PT='type_a=(q4_0|q5_0|iq4_nl),type_b=f32,m=2560,n=[1-8],k=640,'
    for arm in off on; do
        quiet "perf cov $arm"
        env $([ $arm = off ] && echo GGML_CUDA_MMVQ_PASCAL_KCOL=0) "$BIN/test-backend-ops" -b CUDA0 perf -o MUL_MAT -p "$PT" \
            > "$OUT/perf_kcol_$arm.log" 2>&1
    done
    # no BF16 runs here (rig_kcol.sh has them); 2026-11-04_kcol_dense/run/perf_table.py reads both kinds
    touch "$OUT/perf_bf16_off.log" "$OUT/perf_bf16_on.log"
    log "perf: done"
fi
quiet "end"
touch "$OUT/done"
