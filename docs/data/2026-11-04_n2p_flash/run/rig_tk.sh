#!/usr/bin/env bash
# Q928 on the rig: N2P, N3P, N2P-128 and N3P-128 on a P106, and what the N2P Flash-Next files cost a whole-model k-token
# pass against pmix-q2-fast. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
#   check  test-backend-ops on one P106 (gpu_checks.sh MODE=native, the runs in CHECK_RUNS: the four types at the
#          defaults, MMQ, which the PTX runs could not check, and the regression of the half-word types and Q2_0)
#   perf   test-backend-ops perf on one P106 (CARD) at Flash-Next's expert shapes, 1-8 tokens, 10 of 128 experts: the
#          four types' routed gate/up, fused gate/up and down projections beside Q2_0's, with the Pascal kernels at their
#          defaults, with MTOK=0 (2-8 tokens on the earlier kernels) and on llama.cpp's kernels (PASCAL_SKIP)
#   tk     T(k) of each file in FILES at k = 1-8, Q591's method (S21's verify_cost.sh: llama-bench -p 1,...,8 -n 0,
#          five repetitions after a warm-up, under the host-memory guard after a logged quiet check); llama-bench's
#          prompt tokens are random ids, so every arm routes alike. The N2P files also with MTOK=0.
# Usage: BIN=<bin/ of the unpacked n2p tarball, the extras in it> FILES='<name>=<first shard> ...' \
#   VC=<folder with verify_cost.sh, mem_guard.sh, smoke_request.json> OUT=<dir> [STEPS='check perf tk'] \
#   [CHECK_RUNS='np np-mmq reg-half'] [CARD=0] [TS=<-ts of the files>] bash rig_tk.sh
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
STEPS=${STEPS:-check perf tk}
CHECK_RUNS=${CHECK_RUNS:-np np-mmq reg-half}
CARD=${CARD:-0}
P=GGML_CUDA_MMVQ_PASCAL
mkdir -p "$OUT"
rm -f "$OUT/DONE"
st() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }

if step check; then
    st "check start"
    MODE=native BIN="$BIN" OUT="$OUT/check" DEV="$CARD" RUNS="$CHECK_RUNS" bash "$HERE/gpu_checks.sh" > "$OUT/check.out" 2>&1
    st "check exit $? ($(grep -c ' 0 FAIL ' "$OUT/check/console.txt" 2>/dev/null) runs without FAIL lines)"
fi

if step perf; then
    # the routed experts at Flash-Next's shapes, 10 of 128 (random weights: GGML_TEST_RAW_QUANT skips quantizing them)
    PE='type_a=(q2_0|n2p|n3p|n2p_128|n3p_128),type_b=f32,n_mats=128,n_used=10,.*n=[1-8],|type=(q2_0|n2p|n3p),.*n=640,k=2560,use_id=1,n_mats=128,'
    for arm in "def:" "mtok0:${P}_MTOK=0" "llama:${P}_SKIP=q2_0,n2p,n3p,n2p_128,n3p_128"; do
        tag=${arm%%:*}
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES=$CARD GGML_TEST_RAW_QUANT=1 ${arm#*:} "$BIN/test-backend-ops" -b CUDA0 perf \
            -o MUL_MAT_ID,MUL_MAT_VEC_FUSION -p "$PE" > "$OUT/perf_$tag.log" 2>&1
        st "perf $tag exit $?"
    done
fi

if step tk; then
    : "${FILES:?}" "${VC:?}"
    for f in $FILES; do
        name=${f%%=*} model=${f#*=}
        arms="def"
        case $name in *n2p*) arms="def mtok0" ;; esac
        for arm in $arms; do
            envs=""
            [ "$arm" = mtok0 ] && envs="${P}_MTOK=0"
            mkdir -p "$OUT/tk_${name}_$arm"
            # shellcheck disable=SC2086
            env $envs BIN="$BIN" MODEL="$model" DRAFT="$model" OUT="$OUT/tk_${name}_$arm/out" STEPS=wall KS=1,2,3,4,5,6,7,8 \
                ${TS:+TS=$TS} timeout 2400 bash "$VC/verify_cost.sh" > "$OUT/tk_${name}_$arm/verify_cost.out" 2>&1
            st "tk $name $arm (${envs:-defaults}) exit $?"
            pkill -x llama-bench
        done
    done
fi

touch "$OUT/DONE"
st "done"
