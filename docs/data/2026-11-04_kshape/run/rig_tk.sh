#!/usr/bin/env bash
# C267-C269 (Q823-Q825) on the rig: what the retuned k-column shapes, the dispatch table and the multi-token down
# projection do to whole-model k-token passes. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
#   check  test-backend-ops on one P106 first (gpu_checks.sh MODE=native, the runs in CHECK_RUNS: the defaults, every
#          product on the k-column kernels, the experts with tiles and with direct loads, the IQ experts, the bench's
#          CPU-reference check; the other runs passed through the PTX, and the rig's two cores make each run slow)
#   tk     T(k) of the published Swift Q2_0 at k = 1-8, one arm per switch setting, Q591's method (S21's
#          verify_cost.sh: llama-bench -p 1,...,8 -n 0, five repetitions after a warm-up, under the host-memory guard
#          after a logged quiet check); llama-bench's prompt tokens are random ids, so every arm routes alike
#   bench  mmvq_kcol's timings of every launch-shape variant at Flash-Next's shapes (pascal = this build's defaults,
#          old = the shapes before Q823), as Q591 ran it, and test-backend-ops perf of the k-column products with the
#          table on, off and the k-column kernels off (the data a refit of Q824's table needs)
# Arms of tk (ARMS, in this order; all GGML_CUDA_MMVQ_PASCAL_... switches):
#   new        every switch at its default (Q823's shapes, Q824's table, Q825's down projection)
#   tab0       KCOL_TABLE=0: the k-column kernels wherever they fit (Q823's shapes alone)
#   s32k       KCOL_SMEM=32768 KCOL_TABLE=0: Q591's fastest arm (74.9 ... 297.3 ms on -ladder3), now with Q825
#   s32k_mdir  the same with MTOK=1,0,0,0,0: the down projection's direct loads instead of its tiles
#   s24k       KCOL_SMEM=24576 KCOL_TABLE=0: a quarter of the SM per block (Q912)
#   old        KCOL_SMEM=49152 KCOL_TABLE=0: the k-column shapes before Q823, with Q825
#   mtok0      MTOK=0: the routed experts on the earlier kernels
# Usage: BIN=<bin/ of the unpacked kshape tarball, the extras in it> MODEL=<first shard> DRAFT=<any gguf; unused> \
#   VC=<folder with verify_cost.sh, mem_guard.sh, smoke_request.json> OUT=<dir> [STEPS='check tk bench'] [ARMS=...] [CHECK_RUNS=...] \
#   [KCOL_BENCH=<mmvq_kcol>] [CARD=0] [KCOL_DENSE=<docs/data/2026-11-04_kcol_dense of this repository>] bash rig_tk.sh
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
STEPS=${STEPS:-check tk bench}
ARMS=${ARMS:-new tab0 s32k s32k_mdir s24k old mtok0}
CHECK_RUNS=${CHECK_RUNS:-kcol kcol-all mtok mtok-direct mtok-iq bench}
CARD=${CARD:-0}
P=GGML_CUDA_MMVQ_PASCAL
mkdir -p "$OUT"
rm -f "$OUT/DONE"
st() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }

if step check; then
    st "check start"
    MODE=native BIN="$BIN" OUT="$OUT/check" DEV="$CARD" RUNS="$CHECK_RUNS" KCOL_BENCH="${KCOL_BENCH:-}" bash "$HERE/gpu_checks.sh" > "$OUT/check.out" 2>&1
    st "check exit $? ($(grep -c ' 0 FAIL ' "$OUT/check/console.txt" 2>/dev/null) runs without FAIL lines)"
fi

if step tk; then
    : "${MODEL:?}" "${VC:?}"
    for arm in $ARMS; do
        case $arm in
            new)       envs="" ;;
            tab0)      envs="${P}_KCOL_TABLE=0" ;;
            s32k)      envs="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" ;;
            s32k_mdir) envs="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0 ${P}_MTOK=1,0,0,0,0" ;;
            s24k)      envs="${P}_KCOL_SMEM=24576 ${P}_KCOL_TABLE=0" ;;
            old)       envs="${P}_KCOL_SMEM=49152 ${P}_KCOL_TABLE=0" ;;
            mtok0)     envs="${P}_MTOK=0" ;;
            *) st "unknown arm $arm"; continue ;;
        esac
        mkdir -p "$OUT/tk_$arm"
        # shellcheck disable=SC2086
        env $envs BIN="$BIN" MODEL="$MODEL" DRAFT="${DRAFT:-$MODEL}" OUT="$OUT/tk_$arm/out" STEPS=wall KS=1,2,3,4,5,6,7,8 \
            timeout 2400 bash "$VC/verify_cost.sh" > "$OUT/tk_$arm/verify_cost.out" 2>&1
        st "tk $arm (${envs:-defaults}) exit $?"
        pkill -x llama-bench
    done
fi

if step bench; then
    if [ -n "${KCOL_BENCH:-}" ]; then
        CUDA_VISIBLE_DEVICES=$CARD "$KCOL_BENCH" --tokens 1,2,3,4,5,6,7,8 --csv "$OUT/kcol_bench.csv" > "$OUT/kcol_bench.out" 2>&1
        st "bench mmvq_kcol exit $?"
    fi
    # Flash-Next's dense products at 1 to 8 tokens through ggml (Q591's perf arms, the shared expert's down types added)
    PT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs|q4_0|q5_0|iq4_nl),type_b=f32,m=(320|2560|6144|10240|12288|248320),n=[1-8],k=(320|640|2560|6144|10240),'
    for arm in "on:" "tab0:${P}_KCOL_TABLE=0" "off:${P}_KCOL=0"; do
        tag=${arm%%:*}
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES=$CARD ${arm#*:} "$BIN/test-backend-ops" -b CUDA0 perf -o MUL_MAT -p "$PT" > "$OUT/perf_kcol_$tag.log" 2>&1
        st "perf $tag exit $?"
    done
    if [ -n "${KCOL_DENSE:-}" ] && [ -f "$KCOL_DENSE/run/perf_table.py" ]; then
        # perf_table.py reads perf_kcol_{off,on}.log (and perf_bf16_*, empty here): the defaults, then the table off
        for a in on tab0; do
            mkdir -p "$OUT/perf_$a" && cp "$OUT/perf_kcol_off.log" "$OUT/perf_$a/perf_kcol_off.log"
            cp "$OUT/perf_kcol_$a.log" "$OUT/perf_$a/perf_kcol_on.log"
            : > "$OUT/perf_$a/perf_bf16_off.log"; : > "$OUT/perf_$a/perf_bf16_on.log"
            python3 -I "$KCOL_DENSE/run/perf_table.py" "$OUT/perf_$a" > "$OUT/perf_dense_$a.tsv" 2> "$OUT/perf_table_$a.err"
            st "perf table $a exit $?"
        done
    fi
fi

touch "$OUT/DONE"
st "done"
