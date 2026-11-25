#!/usr/bin/env bash
# C1038 (Q1771): replays a record set on the CPU backend, in a container without GPUs (test-backend-ops -b CPU
# --replay), with the build that recorded it. Every distinct case filter (-o, -p) of the set's run list runs once:
# a P106 replay of that list (scripts/replay_checks.sh) selects the same cases, as a run's switches change only what
# the card runs, not which cases. Each filter's replay summary says how many selected cases the set holds and how many it
# does not (NOREC); the CPU's own results are checked against the stored references as well.
# Env: SET (a record set: cases/, runs.tsv), BIN (bin/ of the unpacked rig tarball), OUT, JOBS (test-backend-ops -j,
#   default 8), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), DOCKER_ARGS
# Output: OUT/filters.tsv (filter, -o, -p, the runs that use it), OUT/filter-<n>.log (the console), OUT/filter-<n>.tsv
#   (one line per case: id, case, verdict OK, FAIL or NOREC), OUT/summary.tsv (per filter: cases, OK, FAIL, NOREC, the
#   replay summary line, exit status), OUT/done
# Usage: SET=<dir>/v4kern BIN=<tarball>/bin OUT=<dir> nice -n 19 bash cpu_replay.sh
set -uo pipefail
SET=$(realpath "${SET:?SET=<a record set>}")
BIN=$(realpath "${BIN:?BIN=<bin/ of the rig tarball>}")
OUT=${OUT:?OUT=<dir>}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
[ -f "$SET/runs.tsv" ] && [ -d "$SET/cases" ] || { echo "no runs.tsv or cases/ in $SET" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in (CUDA
# then finds no device; only the CPU backend is present), as in scripts/record_checks.sh
tbo() {
    # shellcheck disable=SC2086
    docker run --rm --label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-} -v "$BIN:$BIN:ro" -v "$SET:$SET:ro" \
        -v "$OUT:$OUT" "$IMAGE" sh -c \
        's=/usr/local/cuda/lib64/stubs/libcuda.so
         if [ -e $s ]; then
             mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1
             export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
         fi
         exec nice -n 19 "$@"' sh "$BIN/test-backend-ops" "$@"
}
# runs.tsv: run, switches, -o, -p, expect, wrap[, known]; an empty field is "-"
awk -F'\t' -v OFS='\t' '{ k = $3 "\t" $4; if (!(k in n)) { n[k] = ++c; o[c] = k }; r[k] = r[k] (r[k] == "" ? "" : " ") $1 }
    END { for (i = 1; i <= c; i++) print i, o[i], r[o[i]] }' "$SET/runs.tsv" > "$OUT/filters.tsv"
printf 'filter\tcases\tok\tfail\tnorec\treplay_summary\texit\n' > "$OUT/summary.tsv"
while IFS=$'\t' read -r n ops params runs; do
    A=(-b CPU -j "$JOBS" --replay "$SET" --replay-log "$OUT/filter-$n.tsv")
    [ "$ops" != - ] && A+=(-o "$ops")
    [ "$params" != - ] && A+=(-p "$params")
    rm -f "$OUT/filter-$n.tsv"
    tbo "${A[@]}" > "$OUT/filter-$n.log" 2>&1
    rc=$?
    sum=$(grep -a -m1 '  replay: ' "$OUT/filter-$n.log" | sed -E 's/^ +replay: [^:]*: //')
    awk -F'\t' -v n="$n" -v s="$sum" -v rc="$rc" -v OFS='\t' '{ c++; v[$3]++ }
        END { print n, c + 0, v["OK"] + 0, v["FAIL"] + 0, v["NOREC"] + 0, s, rc }' "$OUT/filter-$n.tsv" >> "$OUT/summary.tsv"
    echo "$(date -Is) filter $n ($runs): $sum, exit $rc"
done < "$OUT/filters.tsv"
date -Is > "$OUT/done"
