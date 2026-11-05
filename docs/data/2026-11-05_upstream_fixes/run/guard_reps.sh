#!/usr/bin/env bash
# Upstream fixes, the MMVQ row guard: run the same test-backend-ops cases REPS times per build on one GPU and record
# each case's verdict and NMSE. Every run draws new random inputs (test-backend-ops seeds from std::random_device), so
# a race shows as a failure rate. Builds alternate within each repetition, so they see the same machine state.
# Usage (inside a container that sees one CUDA GPU):
#   BINS='<tag>=<dir of test-backend-ops> ...' OUT=<dir> [REPS=30] [OPS=MUL_MAT_ID]
#   [CASES=<-p regex>] [JIT=0|1] [LABEL=<name>] bash guard_reps.sh
#   JIT=1 sets CUDA_FORCE_PTX_JIT=1: the driver compiles the build's compute_61 PTX instead of loading its sm_86 code
# Output: OUT/<LABEL>.tsv (label, build, run, case, verdict, nmse), OUT/logs/<LABEL>-<build>-<run>.log
set -uo pipefail
OUT=${OUT:?OUT=<dir>}
BINS=${BINS:?BINS=<tag>=<dir> ...}
REPS=${REPS:-30}
OPS=${OPS:-MUL_MAT_ID}
CASES=${CASES:-n_mats=32,n_used=10,b=0,m=129,n=1,}
JIT=${JIT:-0}
LABEL=${LABEL:-sass}
mkdir -p "$OUT/logs"
T=$OUT/$LABEL.tsv
printf 'label\tbuild\trun\tcase\tverdict\tnmse\n' > "$T"
read -r -a PAIRS <<< "$BINS"
for i in $(seq 1 "$REPS"); do
    for p in "${PAIRS[@]}"; do
        tag=${p%%=*}; bin=${p#*=}
        L=$OUT/logs/$LABEL-$tag-$i.log
        env CUDA_FORCE_PTX_JIT="$JIT" CUDA_CACHE_PATH="$OUT/nvcache-$tag" CUDA_CACHE_MAXSIZE=4294967296 timeout 1800 \
            "$bin/test-backend-ops" -b CUDA0 -o "$OPS" -p "$CASES" > "$L" 2>&1
        rc=$?
        # a case line: "  OP(params): OK", or "[OP] ERR = x > limit   OP(params): FAIL"; colour codes stripped.
        # The error column is ERR (test-backend-ops' error measure for the op, here the NMSE) when the case fails
        sed 's/\x1b\[[0-9;]*m//g' "$L" | awk -v l="$LABEL" -v b="$tag" -v i="$i" -v rc="$rc" '
            /[A-Z_]+\([^)]*\): (OK|FAIL)/ {
                match($0, /[A-Z_]+\([^)]*\)/); c = substr($0, RSTART, RLENGTH)
                v = $NF; n = ""
                if (match($0, /(ERR|NMSE) = [0-9.eE+-]+/)) { n = substr($0, RSTART, RLENGTH); sub(/^[A-Z]+ = /, "", n) }
                print l "\t" b "\t" i "\t" c "\t" v "\t" n; seen = 1
            }
            # a run that stopped early (a crash: exit 134; a timeout: 124) or printed no case: its exit status
            END { if (!seen || (rc != 0 && rc != 1)) print l "\t" b "\t" i "\t(run)\texit " rc "\t" }' >> "$T"
    done
    echo "$(date -Is) $LABEL run $i done"
done
