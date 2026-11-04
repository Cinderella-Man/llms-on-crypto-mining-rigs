#!/usr/bin/env bash
# Q671 / C131: the compiler-flag and launch-bound arms of timing_arms.tsv on one P106-090, kernel by kernel: the fork's own
# kernels through test-backend-ops perf (the cases of make_test_cases_perf at Qwen3.8-Flash-Next's and Ornith-1.5-122B's
# shapes) and the standalone kernels through bench/mmvq_pascal. Runs ON the rig, as its own timed task, with nothing else
# running: it refuses to start when a process uses the card or the load is not low.
#
# Layout under ROOT: <arm>/bin/test-backend-ops for every arm (build_timing_arms.sh), and bench/<arm>/{mmvq_pascal,mmvq_kcol}
# for the arms with bench flags (build_bench_arms.sh).
# Usage (from the workstation, no files left on the rig but $OUT):
#   ssh rig "ROOT=<dir> ARMS='base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg' REPS=3 OUT=/tmp/ccflags_timing bash -s" < p106_timing.sh
#   scp -r rig:/tmp/ccflags_timing . && ssh rig 'rm -rf /tmp/ccflags_timing'     # then timing_table.py ccflags_timing
# Env: ROOT (required), ARMS (default: base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg), REPS passes per arm (default 3; the arms run in
# order on odd passes and in reverse on even ones, so that drift shows), CARD (default 0), OUT (default
# /tmp/ccflags_timing), LOAD_MAX (the load average above which it refuses to start, default 0.5), CASES (default: every group below). Output: $OUT/<arm>/<group>.<pass>.{txt,csv}, $OUT/console.txt,
# $OUT/done when finished. Most of a test-backend-ops case's time goes into initializing random weights on the host: time
# one group with REPS=1 first (CASES=iq_down).
set -u
ROOT=${ROOT:?ROOT=<folder with the arms>}
ARMS=${ARMS:-base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg}
REPS=${REPS:-3}
CARD=${CARD:-0}
LOAD_MAX=${LOAD_MAX:-0.5}
OUT=${OUT:-/tmp/ccflags_timing}
CASES=${CASES:-iq_down iq_gateup iq_dense q2_0_mtok q2_0_one kcol bf16 bench_mtok bench_q2_0 bench_bf16 bench_kcol}
mkdir -p "$OUT"
rm -f "$OUT/done"

# test-backend-ops cases: group|op|regex over the case's parameters (make_test_cases_perf in tests/test-backend-ops.cpp)
TBO=(
    # IQ2_S and IQ3_XXS at Ornith-1.5-122B's shapes (hidden 3072, experts of width 1024, 8 of 64 used), 1, 2, 4 tokens:
    # the expert down projection, the fused expert gate/up with SwiGLU, an 8192-row dense product. Tokens 2 and 4 run the
    # multi-token kernels that iqmc2 changes; token 1 runs the one-token kernels it leaves alone.
    "iq_down|MUL_MAT_ID|type_a=(iq2_s|iq3_xxs),type_b=f32,n_mats=64,n_used=8,b=0,m=3072,n=[124],k=1024"
    "iq_gateup|MUL_MAT_VEC_FUSION|type=(iq2_s|iq3_xxs),.*m=1024,n=[124],k=3072,use_id=1"
    "iq_dense|MUL_MAT|type_a=(iq2_s|iq3_xxs),type_b=f32,m=8192,n=[124],k=3072"
    # Flash-Next's Q2_0 routed experts (10 of 128 used): gate/up (2560 -> 640) and down (640 -> 2560) at 2 to 8 tokens,
    # where the multi-token kernel runs (mtokmb4), and at one token
    "q2_0_mtok|MUL_MAT_ID|type_a=q2_0,type_b=f32,n_mats=128,n_used=10,b=(0|1),m=(640|2560),n=[2-8],k=(2560|640)"
    "q2_0_one|MUL_MAT_ID|type_a=q2_0,type_b=f32,n_mats=128,n_used=10,b=(0|1),m=(640|2560),n=1,k=(2560|640)"
    # the dense products of a 1 to 8 token verify step (the k-column kernels)
    "kcol|MUL_MAT|type_a=(q8_0|q3_K|iq4_xs|q4_K|q2_0|q5_K|q6_K),type_b=f32,m=(10240|320|6144|12288|2560|248320),n=[1-8],k=(2560|10240|320|6144|640),bs=\[1,1\]"
    # the BF16 products (hyper-connection projections, indexer, recurrent gates) at 1 to 8 tokens
    "bf16|MUL_MAT|type_a=bf16,type_b=f32,m=(10240|320|4|512|48),n=[1-8],k=(320|10240|2560),bs=\[1,1\]"
)
# bench/mmvq_pascal cases: group|binary|arguments (the CSV is added)
BENCH=(
    "bench_mtok|mmvq_pascal|--q2_0-mtok --variants mtok,mtok_mb4 --tokens 2,4,8 --routes random"
    "bench_q2_0|mmvq_pascal|--q2_0 --variants pascal"
    "bench_bf16|mmvq_pascal|--bf16 --variants pascal --tokens 1,2,4,8"
    "bench_kcol|mmvq_kcol|--types q2_0,q3_K,iq4_xs,q8_0 --variants pascal --tokens 1,2,4,8"
)
wanted() { case " $CASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# the rig must be quiet: no process on the card, a low load
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$CARD" 2>/dev/null | grep -c .)
load=$(cut -d' ' -f1 /proc/loadavg)
{
    echo "start $(date -Is) ROOT=$ROOT ARMS='$ARMS' REPS=$REPS CARD=$CARD CASES='$CASES'"
    echo "processes on the card: $busy, load $load"
    nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu --format=csv,noheader -i "$CARD"
} > "$OUT/console.txt"
if [ "$busy" != 0 ] || awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { exit !(l > m) }'; then
    echo "not quiet (processes on the card: $busy, load $load): stop before timing" | tee -a "$OUT/console.txt"
    exit 2
fi

for pass in $(seq 1 "$REPS"); do
    order=$ARMS
    if [ $((pass % 2)) = 0 ]; then order=$(for a in $ARMS; do echo "$a"; done | tac | tr '\n' ' '); fi
    for arm in $order; do
        mkdir -p "$OUT/$arm"
        for c in "${TBO[@]}"; do
            IFS='|' read -r g op rx <<< "$c"
            wanted "$g" || continue
            CUDA_VISIBLE_DEVICES=$CARD "$ROOT/$arm/bin/test-backend-ops" perf -b CUDA0 -o "$op" -p "$rx" > "$OUT/$arm/$g.$pass.txt" 2>&1
            echo "pass $pass $arm $g exit $? $(date +%T)" >> "$OUT/console.txt"
        done
        for c in "${BENCH[@]}"; do
            IFS='|' read -r g bin args <<< "$c"
            wanted "$g" || continue
            [ -x "$ROOT/bench/$arm/$bin" ] || continue   # no bench build for this arm
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES=$CARD "$ROOT/bench/$arm/$bin" $args --csv "$OUT/$arm/$g.$pass.csv" > "$OUT/$arm/$g.$pass.txt" 2>&1
            echo "pass $pass $arm $g exit $? $(date +%T)" >> "$OUT/console.txt"
        done
    done
done
echo "end $(date -Is)" >> "$OUT/console.txt"
touch "$OUT/done"
