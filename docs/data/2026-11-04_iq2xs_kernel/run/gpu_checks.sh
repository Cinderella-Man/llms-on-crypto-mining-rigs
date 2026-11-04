#!/usr/bin/env bash
# GPU correctness of the Pascal IQ2_XXS / IQ2_XS / IQ3_S kernels (Q222), with IQ2_S and IQ3_XXS at the same shapes:
# test-backend-ops against the CPU on one card, for the five grid types' MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION and
# MUL_MAT_ID_SHARED cases of 1 to 4 tokens (larger batches run llama.cpp's own batched kernels, which this branch does
# not touch). Prints pass counts and failures only; no timings (bench/mmvq_pascal/iq_perf.sh times them).
#   MODE=ptx     a newer NVIDIA GPU (run inside nvidia/cuda:12.9.2-devel-ubuntu24.04 with one GPU): the compute 6.1 PTX
#                the P106 runs, through the driver's JIT (a rig build holds no other device code; PTX_JIT=1 adds
#                CUDA_FORCE_PTX_JIT=1 for a build that does, which also aborts every case cuBLAS takes), with the Pascal
#                kernels forced on any GPU (GGML_CUDA_MMVQ_PASCAL=2); every case
#   MODE=native  a P106 (compute 6.1, its own sm_61 code); the dense cases of 6144 rows or more are left out, because a
#                2-core host quantizes their IQ weights for many minutes (the PTX run covers them)
# Runs (RUNS, default "on skip memcheck"):
#   on        the kernels as built
#   skip      GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_xxs,iq2_xs,iq3_s: the new types on llama.cpp's mul_mat_vec_q (the same cases
#             must pass there too, which checks the cases rather than the kernels)
#   memcheck  ptx only: compute-sanitizer memcheck on one case per new type and kind
# Env: BIN (folder with test-backend-ops), OUT (output folder), MODE, RUNS, GPU (CUDA_VISIBLE_DEVICES for native,
# default 0), TYPES (default "iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs").
# Output: $OUT/<run>.txt (the raw test-backend-ops lines of failures and summaries), $OUT/console.txt (one line per run,
# type and op: passed/total and failures), $OUT/done.
set -uo pipefail
BIN=${BIN:?}; OUT=${OUT:?}; MODE=${MODE:-native}
RUNS=${RUNS:-on skip memcheck}
TYPES=${TYPES:-iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs}
mkdir -p "$OUT"
rm -f "$OUT/done" "$OUT/console.txt"
T="$BIN/test-backend-ops -b CUDA0"
if [ "$MODE" = ptx ]; then
    export GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=${PTX_JIT:-0}
    EXCL=""
else
    export CUDA_VISIBLE_DEVICES=${GPU:-0}
    EXCL='(?!.*(m=6144|m=10240|m=12288|k=6144))'
fi
# 1 to 4 tokens: the fusion test names tokens m, the shared-expert test n, the others n
tok() { case $1 in MUL_MAT_VEC_FUSION) echo "m=[1-4],";; MUL_MAT_ID_SHARED) echo "n=[1-4]";; *) echo "n=[1-4],";; esac; }
summ() { grep -E "FAIL|tests passed|Abort|CUDA error|terminate called|Segmentation|not supported" | sed -E 's/^ +//'; }
line() { # <run> <type> <op> <log>
    local p f
    p=$(grep -E "tests passed" "$4" | tail -n 1 | grep -oE "[0-9]+/[0-9]+")
    f=$(grep -cE "FAIL|Abort|CUDA error|terminate called|Segmentation" "$4")
    echo "$1 $2 $3: ${p:-NO CASES RAN} passed, $f failure lines" | tee -a "$OUT/console.txt"
}
for run in $RUNS; do
    case $run in
        on|skip)
            [ "$run" = skip ] && export GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_xxs,iq2_xs,iq3_s || unset GGML_CUDA_MMVQ_PASCAL_SKIP
            : > "$OUT/$run.txt"
            for t in $TYPES; do
                for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_ID_SHARED; do
                    log=$OUT/$run-$t-$op.log
                    $T -o $op -p "^$EXCL.*=$t,.*$(tok $op)" > "$log" 2>&1
                    { echo "== $run $t $op"; summ < "$log"; } >> "$OUT/$run.txt"
                    line "$run" "$t" "$op" "$log"
                done
            done
            unset GGML_CUDA_MMVQ_PASCAL_SKIP ;;
        memcheck)
            [ "$MODE" = ptx ] || continue
            CS=${CS:-/usr/local/cuda/bin/compute-sanitizer}
            : > "$OUT/memcheck.txt"
            for t in iq2_xxs iq2_xs iq3_s; do
                for c in "MUL_MAT_ID|=$t,.*m=640,n=1,k=2560" "MUL_MAT_ID|=$t,.*m=2560,n=4,k=1024" \
                         "MUL_MAT_VEC_FUSION|=$t,.*m=2,n=640,k=2560,use_id=1" "MUL_MAT|=$t,.*m=641,n=1,k=2560" \
                         "MUL_MAT|=$t,.*m=2560,n=3,k=6144"; do
                    echo "== memcheck ${c%%|*} ${c#*|}" >> "$OUT/memcheck.txt"
                    $CS --tool memcheck --print-limit 5 $T -o "${c%%|*}" -p "${c#*|}" 2>&1 \
                        | grep -E "ERROR SUMMARY|Invalid|out of bounds|tests passed|FAIL" | sort | uniq -c >> "$OUT/memcheck.txt"
                done
            done
            echo "memcheck: $(grep -c 'ERROR SUMMARY: 0 errors' "$OUT/memcheck.txt") of 15 cases with 0 errors" | tee -a "$OUT/console.txt" ;;
    esac
done
touch "$OUT/done"
