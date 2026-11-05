#!/usr/bin/env bash
# T0b's diagnostics on one recorded draw (case_reps.sh's OUT/rec/<build>-<run>): the draw replayed REPS times on each
# build (the same inputs every time, so any spread is the kernel's), then compute-sanitizer's memcheck, initcheck and
# racecheck on the first build, with the same PTX settings as case_reps.sh.
# Usage (inside a container of the build image that sees one card):
#   BINS='l5=<bin/> l4=<bin/>' REC=<a record> OUT=<dir> [REPS=10] [CASES=<-p regex>] bash case_diag.sh
# Output: OUT/det-<build>-<pass>.replay and OUT/san-<tool>.replay (index, case, verdict, per-node NMSE), OUT/*.log
set -uo pipefail
OUT=${OUT:?OUT=<dir>}
REC=${REC:?REC=<record>}
BINS=${BINS:?BINS=<tag>=<bin/> ...}
CASES=${CASES:-type_a=n3p_128,type_b=f32,n_mats=32,n_used=10,b=0,m=2561,n=1,k=640,m_v=0,ids_offset=0,skip=0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ON="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
    GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
for p in $BINS; do
    tag=${p%%=*}; bin=${p#*=}
    for i in $(seq 1 "${REPS:-10}"); do
        rm -f "$OUT/det-$tag-$i.replay"
        # shellcheck disable=SC2086
        env CUDA_CACHE_PATH="$OUT/nvcache-$tag" CUDA_CACHE_MAXSIZE=4294967296 $ON "$bin/test-backend-ops" -b CUDA0 \
            -o MUL_MAT_ID -p "$CASES" --replay "$REC" --replay-log "$OUT/det-$tag-$i.replay" > "$OUT/det-$tag-$i.log" 2>&1
        echo "$tag $i $(cut -f3,4 "$OUT/det-$tag-$i.replay")"
    done
done
p=${BINS%% *}; tag=${p%%=*}; bin=${p#*=}
for t in memcheck initcheck racecheck; do
    # shellcheck disable=SC2086
    env CUDA_CACHE_PATH="$OUT/nvcache-$tag" CUDA_CACHE_MAXSIZE=4294967296 $ON timeout 1800 compute-sanitizer --tool $t \
        --print-limit 20 "$bin/test-backend-ops" -b CUDA0 -o MUL_MAT_ID -p "$CASES" --replay "$REC" \
        --replay-log "$OUT/san-$t.replay" > "$OUT/san-$t.log" 2>&1
    echo "$t: $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' "$OUT/san-$t.log"); $(cut -f3,4 "$OUT/san-$t.replay")"
done
touch "$OUT/done"
