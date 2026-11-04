#!/usr/bin/env bash
# GPU correctness of the Q713 kernels (the grid types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_M in the k-column
# dense products and the multi-token expert products, 2 to 8 tokens): test-backend-ops against the CPU on one card, for
# every MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION and MUL_MAT_ID_SHARED case of these types at 1 to 8 tokens (Q222's
# cases at 1 to 4 tokens included: their 2 to 4 dense tokens now run the k-column kernels). Prints pass counts and
# failures only; no timings (bench/mmvq_pascal/iq_kcol_perf.sh times them).
#   MODE=ptx     a newer NVIDIA GPU (run inside nvidia/cuda:12.9.2-devel-ubuntu24.04 with one GPU): the compute 6.1 PTX
#                the P106 runs, through the driver's JIT (CUDA_FORCE_PTX_JIT=1), with the Pascal kernels forced on any
#                GPU (GGML_CUDA_MMVQ_PASCAL=2, GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 for the P106's launch shapes); every case
#   MODE=native  a P106 (compute 6.1, its own sm_61 code); the dense cases of 6144 rows or more and K = 6144, the
#                consecutive-routing cases (keep=) and the 32-of-64-expert cases are left out, because a 2-core host
#                quantizes their IQ weights for hours (the PTX run covers them); RUNS=on is enough there
# Runs (RUNS, default "on kcol_r2 cap1 rb1 off skip memcheck"):
#   on        the kernels as built
#   kcol_r2   GGML_CUDA_MMVQ_PASCAL_KCOL_R=2: the k-column kernels with 2 rows per lane group (R = 1 by default for these
#             types), MUL_MAT only
#   cap1      GGML_CUDA_MMVQ_PASCAL_MTOK=1,1: expert groups of one entry (no sharing), the expert ops only
#   rb1       GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1: one row block per block, the expert ops only
#   off       GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP and GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP list the six types: the kernels before
#             Q713 (and llama.cpp's matrix-matrix kernels for the experts of 5 to 8 tokens)
#   skip      GGML_CUDA_MMVQ_PASCAL_SKIP lists the six types: llama.cpp's kernels throughout (checks the cases rather
#             than the kernels)
#   memcheck  ptx only: compute-sanitizer memcheck on one case per type and kind
# Env: BIN (folder with test-backend-ops), OUT (output folder), MODE, RUNS, GPU (CUDA_VISIBLE_DEVICES for native,
# default 0), TYPES (default "iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs iq1_m").
# Output: $OUT/<run>.txt (the raw test-backend-ops lines of failures and summaries), $OUT/console.txt (one line per run,
# type and op: passed/total and failures), $OUT/memcheck.txt, $OUT/done.
set -uo pipefail
BIN=${BIN:?}; OUT=${OUT:?}; MODE=${MODE:-native}
RUNS=${RUNS:-on kcol_r2 cap1 rb1 off skip memcheck}
TYPES=${TYPES:-iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs iq1_m}
LIST=$(echo $TYPES | tr ' ' ',')
mkdir -p "$OUT"
rm -f "$OUT/done" "$OUT/console.txt"
T="$BIN/test-backend-ops -b CUDA0"
if [ "$MODE" = ptx ]; then
    export GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
    EXCL=""
else
    export CUDA_VISIBLE_DEVICES=${GPU:-0}
    EXCL='(?!.*(m=6144|m=10240|m=12288|k=6144|keep=|n_used=32))'
fi
summ() { grep -E "FAIL|tests passed|Abort|CUDA error|terminate called|Segmentation|not supported" | sed -E 's/^ +//'; }
line() { # <run> <type> <op> <log>
    local p f
    p=$(grep -E "tests passed" "$4" | tail -n 1 | grep -oE "[0-9]+/[0-9]+")
    f=$(grep -cE "FAIL|Abort|CUDA error|terminate called|Segmentation" "$4")
    echo "$1 $2 $3: ${p:-NO CASES RAN} passed, $f failure lines" | tee -a "$OUT/console.txt"
}
run_ops() { # <run> <ops...>
    local run=$1 t op log
    shift
    : > "$OUT/$run.txt"
    for t in $TYPES; do
        for op in "$@"; do
            log=$OUT/$run-$t-$op.log
            $T -o "$op" -p "^$EXCL.*=$t," > "$log" 2>&1
            { echo "== $run $t $op"; summ < "$log"; } >> "$OUT/$run.txt"
            line "$run" "$t" "$op" "$log"
        done
    done
}
EXPERT_OPS="MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_ID_SHARED"
for run in $RUNS; do
    case $run in
        on)      run_ops on MUL_MAT $EXPERT_OPS ;;
        kcol_r2) GGML_CUDA_MMVQ_PASCAL_KCOL_R=2 run_ops kcol_r2 MUL_MAT ;;
        cap1)    GGML_CUDA_MMVQ_PASCAL_MTOK=1,1 run_ops cap1 $EXPERT_OPS ;;
        rb1)     GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1 run_ops rb1 $EXPERT_OPS ;;
        off)     GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=$LIST GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=$LIST run_ops off MUL_MAT $EXPERT_OPS ;;
        skip)    GGML_CUDA_MMVQ_PASCAL_SKIP=$LIST run_ops skip MUL_MAT $EXPERT_OPS ;;
        memcheck)
            [ "$MODE" = ptx ] || continue
            CS=${CS:-/usr/local/cuda/bin/compute-sanitizer}
            : > "$OUT/memcheck.txt"
            n=0
            for t in $TYPES; do
                for c in "MUL_MAT_ID|=$t,.*m=640,n=8,k=2560" "MUL_MAT_ID|=$t,.*m=1000,n=5,k=1024" \
                         "MUL_MAT_VEC_FUSION|=$t,.*m=7,n=641,k=2560" "MUL_MAT|=$t,.*m=641,n=6,k=2560" \
                         "MUL_MAT|=$t,.*m=2560,n=8,k=6144"; do
                    echo "== memcheck ${c%%|*} ${c#*|}" >> "$OUT/memcheck.txt"
                    $CS --tool memcheck --print-limit 5 $T -o "${c%%|*}" -p "${c#*|}" 2>&1 \
                        | grep -E "ERROR SUMMARY|Invalid|out of bounds|tests passed|FAIL" | sort | uniq -c >> "$OUT/memcheck.txt"
                    n=$((n + 1))
                done
            done
            echo "memcheck: $(grep -c 'ERROR SUMMARY: 0 errors' "$OUT/memcheck.txt") of $n cases with 0 errors" | tee -a "$OUT/console.txt" ;;
        *) echo "unknown run $run" >&2 ;;
    esac
done
touch "$OUT/done"
