#!/usr/bin/env bash
# T0b: is one failing test-backend-ops case a draw or a regression? The cases (CASES, a -p regex) run REPS times on one
# card per build, through the compute_61 PTX with the fork's kernel paths forced on and a P106's launch shapes (as
# 2026-11-04_ladder4_merge/run/gpu_checks.sh MODE=ptx). Every run of the first build records its draw (--record), and
# each other build replays that same draw (--replay) as well as drawing its own: a build's NMSE on the other build's
# inputs tells the code apart from the inputs.
# Usage (inside a container of the build image that sees one card):
#   BINS='l5=<bin/> l4=<bin/>' OUT=<dir> [REPS=30] [OPS=MUL_MAT_ID] [CASES=<-p regex>] [KEEP_REC=0|1|none] bash case_reps.sh
# Output: OUT/nmse.tsv, a line per case and run: build, run, draw (own, or the build whose record it replays), case,
#   verdict, NMSE (per compared node, the op first); OUT/logs/; OUT/done. After the replays a run's records are
#   deleted, except (KEEP_REC=0, the default) a run with a FAIL; KEEP_REC=1 keeps all, KEEP_REC=none none.
set -uo pipefail
OUT=${OUT:?OUT=<dir>}
BINS=${BINS:?BINS=<tag>=<bin/> ...}
REPS=${REPS:-30}
OPS=${OPS:-MUL_MAT_ID}
CASES=${CASES:-type_a=n3p_128,type_b=f32,n_mats=32,n_used=10,b=0,m=2561,n=1,k=640,m_v=0,ids_offset=0,skip=0}
KEEP_REC=${KEEP_REC:-0}
mkdir -p "$OUT/logs" "$OUT/rec"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ON="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
    GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
T=$OUT/nmse.tsv
[ -f "$T" ] || printf 'build\trun\tdraw\tcase\tverdict\tnmse\n' > "$T"
read -r -a PAIRS <<< "$BINS"
FIRST=${PAIRS[0]%%=*}
tbo() { # <tag> <log> <args...>
    local tag=$1 log=$2 bin
    shift 2
    for p in "${PAIRS[@]}"; do [ "${p%%=*}" = "$tag" ] && bin=${p#*=}; done
    # shellcheck disable=SC2086
    env CUDA_CACHE_PATH="$OUT/nvcache-$tag" CUDA_CACHE_MAXSIZE=4294967296 $ON timeout 1800 \
        "$bin/test-backend-ops" -b CUDA0 -o "$OPS" -p "$CASES" "$@" > "$log" 2>&1
}
for i in $(seq 1 "$REPS"); do
    for p in "${PAIRS[@]}"; do
        tag=${p%%=*}
        R=$OUT/rec/$tag-$i
        rm -rf "$R"
        tbo "$tag" "$OUT/logs/$tag-$i.log" --record "$R"
        # each record's .meta: the case's name and the live check's verdict and per-node errors
        for m in "$R"/cases/*.meta; do
            [ -f "$m" ] || continue
            awk -F'\t' -v b="$tag" -v i="$i" '$1 == "name" { n = $2 } $1 == "live" { print b "\t" i "\town\t" n "\t" $3 "\t" $4 }' "$m"
        done >> "$T"
    done
    for p in "${PAIRS[@]:1}"; do
        tag=${p%%=*}
        L=$OUT/logs/$tag-on-$FIRST-$i.replay
        rm -f "$L"
        tbo "$tag" "$OUT/logs/$tag-on-$FIRST-$i.log" --replay "$OUT/rec/$FIRST-$i" --replay-log "$L"
        awk -F'\t' -v b="$tag" -v i="$i" -v f="$FIRST" '{ print b "\t" i "\t" f "\t" $2 "\t" $3 "\t" $4 }' "$L" >> "$T"
    done
    # a run's records stay when any of its lines (own draws or replays) failed
    if [ "$KEEP_REC" = none ] || { [ "$KEEP_REC" != 1 ] && ! awk -F'\t' -v i="$i" '$2 == i && $5 == "FAIL" { f = 1 } END { exit !f }' "$T"; }; then
        for p in "${PAIRS[@]}"; do rm -rf "$OUT/rec/${p%%=*}-$i"; done
    fi
    echo "$(date -Is) run $i done"
done
touch "$OUT/done"
