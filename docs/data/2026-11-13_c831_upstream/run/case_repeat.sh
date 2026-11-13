#!/usr/bin/env bash
# R153 (Q1463): one recorded case replayed N times with each of two builds, to tell a difference between the builds
# from run-to-run variation: each run prints the case's verdict and every node's error (GGML_TEST_REPLAY_LOG).
# TIP_SET and BASE_SET are the same record under each build's case names (alias_base.py); OPS and P select the case
# (test-backend-ops -o, -p: anchor P so that it selects one case); DEV is the card (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Usage: TIP=<tree> BASE=<tree> TIP_SET=<set> BASE_SET=<set> OPS=MUL_MAT_ID P='^type_a=...$' [N=5] [DEV=1] [ENV='K=V ...'] \
#   OUT=<dir> bash case_repeat.sh
# Output: OUT/repeat.tsv (build, run, case, verdict, errors), OUT/<build>-<run>.log
set -uo pipefail
N=${N:-5}
DEV=${DEV:-1}
OUT=${OUT:?OUT=<dir>}
mkdir -p "$OUT"
: > "$OUT/repeat.tsv"
for b in tip base; do
    if [ $b = tip ]; then tree=${TIP:?}; set=${TIP_SET:?}; else tree=${BASE:?}; set=${BASE_SET:?}; fi
    for i in $(seq 1 "$N"); do
        rm -f "$OUT/$b-$i.rlog"
        # shellcheck disable=SC2086
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEV" ${ENV:-} GGML_TEST_REPLAY="$set" \
            GGML_TEST_REPLAY_LOG="$OUT/$b-$i.rlog" "$tree/bin/test-backend-ops" -b CUDA0 -o "${OPS:?}" -p "${P:?}" \
            > "$OUT/$b-$i.log" 2>&1
        echo "exit $?" >> "$OUT/$b-$i.log"
        awk -F'\t' -v b="$b" -v i="$i" '{ print b "\t" i "\t" $2 "\t" $3 "\t" $4 }' "$OUT/$b-$i.rlog" >> "$OUT/repeat.tsv"
    done
done
cut -f1,4,5 "$OUT/repeat.tsv" | sort | uniq -c
