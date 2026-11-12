#!/usr/bin/env bash
# R129's tables: C806 on a P106 (rig_c806.sh's output) read against itself and against R112's replay of the same set on
# the -kern build, whose GPU code is the -c806 build's (8,540 of 8,540 functions, same SASS and PTX).
# Runs anywhere with bash and awk (CPU only; run it on a build machine, not on the rig).
# Env:
#   RIG_OUT    rig_c806.sh's OUT/replay (results.tsv, unrecorded.tsv, <run>.tsv)
#   R112_OUT   R112's replay output on -kern (its <run>.tsv; the -kern test-backend-ops writes no NOREC lines)
#   R112_UNREC saved/unrecorded_r112_dmt.tsv beside this folder (C806's list of R112's 30 unrecorded cases)
#   OUT        where the tables go
#   SET_DIR, RUN_DIR (optional) the set's and the run's folders on the rig, written as $SET and $D in the copies
# Output:
#   counts.tsv        per run: expect, verdict, cases, the per-case table's OK, FAIL and NOREC lines, the unrecorded
#                     cases unrecorded.tsv lists, whether the NOREC lines name exactly those, whether they are R112's 30,
#                     and how many NOREC ids were also replayed (must be 0)
#   vs_r112.tsv       per run and part: cases compared with R112's run on -kern, how many have the same verdict and the
#                     same errors to every printed digit, how many differ, how many R112's run lacks
#   vs_r112_diff.tsv  the differing cases, if any
#   failing.tsv       the failing cases of the two dmt-q3f runs (run, id, case, errors)
#   cases.tsv.gz      every case line of every run's per-case table (run, id, case, verdict, errors)
#   console.txt, results.tsv, unrecorded.tsv  replay_checks.sh's console and tables
set -euo pipefail
RIG_OUT=${RIG_OUT:?RIG_OUT=<rig_c806.sh OUT/replay>}; R112_OUT=${R112_OUT:?R112_OUT=<R112 -kern replay out>}
R112_UNREC=${R112_UNREC:?R112_UNREC=<saved/unrecorded_r112_dmt.tsv>}; OUT=${OUT:?OUT=<dir>}
mkdir -p "$OUT"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
awk -F'\t' 'NR > 1 && $1 == "dmt-on" { print $2 }' "$R112_UNREC" | sort > "$T/r112"
[ "$(wc -l < "$T/r112")" -eq 30 ] || { echo "R112_UNREC: expected 30 dmt-on ids" >&2; exit 1; }

printf 'run\texpect\tverdict\tcases\tok_lines\tfail_lines\tnorec_lines\tunrecorded_listed\tnorec_lines_are_the_list\tsame_30_as_r112\tnorec_also_replayed\n' \
    > "$OUT/counts.tsv"
# results.tsv: run switches expect cases fail_lines exit memcheck_errors verdict known_fail unrecorded (empty fields
# occur, so awk picks the columns: a tab IFS in read would merge them)
awk -F'\t' 'NR > 1 { print $1 "|" $3 "|" $4 "|" $8 }' "$RIG_OUT/results.tsv" > "$T/runs"
while IFS='|' read -r run expect cases verdict; do
    t="$RIG_OUT/$run.tsv"
    read -r ok fail norec < <(awk -F'\t' '{ c[$3]++ } END { print c["OK"] + 0, c["FAIL"] + 0, c["NOREC"] + 0 }' "$t")
    awk -F'\t' '$3 == "NOREC" { print $1 }' "$t" | sort > "$T/norec"
    awk -F'\t' '$3 == "OK" || $3 == "FAIL" { print $1 }' "$t" | sort > "$T/replayed"
    awk -F'\t' -v r="$run" 'NR > 1 && $1 == r { print $2 }' "$RIG_OUT/unrecorded.tsv" | sort > "$T/listed"
    same_list=no; cmp -s "$T/norec" "$T/listed" && same_list=yes
    same_r112=no; cmp -s "$T/norec" "$T/r112" && same_r112=yes
    overlap=$(comm -12 "$T/norec" "$T/replayed" | wc -l)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$expect" "$verdict" "$cases" "$ok" "$fail" "$norec" \
        "$(wc -l < "$T/listed")" "$same_list" "$same_r112" "$overlap" >> "$OUT/counts.tsv"
done < "$T/runs"

# compare RUN's replayed cases whose case string matches PAT with R112's REF run, id by id (verdict and errors)
compare() {
    local run=$1 ref=$2 part=$3 pat=$4
    awk -F'\t' -v pat="$pat" -v run="$run" -v ref="$ref" -v diffs="$T/diff" '
        NR == FNR { if (($3 == "OK" || $3 == "FAIL") && $2 ~ pat) r[$1] = $3 "\t" $4; next }
        ($3 == "OK" || $3 == "FAIL") && $2 ~ pat {
            n++
            if (!($1 in r)) miss++
            else if (r[$1] == $3 "\t" $4) same++
            else { d++; print run "\t" ref "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" r[$1] >> diffs }
        }
        END { printf "%d\t%d\t%d\t%d\n", n, same + 0, d + 0, miss + 0 }' "$R112_OUT/$ref.tsv" "$RIG_OUT/$run.tsv" \
        | { IFS=$'\t' read -r n s d m; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$part" "$ref" "$n" "$s" "$d" "$m"; }
}
: > "$T/diff"
DMT='type_a=(iq4_xs|q4_K),'
Q3='type_a=q3_K,'
{
    printf 'run\tpart\tr112_run\tcases\tsame_verdict_and_errors\tdiffer\tnot_in_r112_run\n'
    cut -d'|' -f1 "$T/runs" | while read -r run; do
        case $run in
        dmt-q3f-*)
            compare "$run" dmt-on "iq4_xs and q4_K" "$DMT"
            compare "$run" q3kmt-fault q3_K "$Q3"
            ;;
        *) compare "$run" "$run" all "." ;;
        esac
    done
} > "$OUT/vs_r112.tsv"
{ printf 'run\tr112_run\tid\tcase\tverdict\terrors\tr112_verdict\tr112_errors\n'; cat "$T/diff"; } > "$OUT/vs_r112_diff.tsv"
{
    printf 'run\tid\tcase\terrors\n'
    for run in dmt-q3f-pass dmt-q3f-fail; do
        awk -F'\t' -v run="$run" '$3 == "FAIL" { print run "\t" $1 "\t" $2 "\t" $4 }' "$RIG_OUT/$run.tsv"
    done
} > "$OUT/failing.tsv"
# every replayed and unrecorded case of every run, in the run list's order (run, id, case, verdict, errors)
{
    printf 'run\tid\tcase\tverdict\terrors\n'
    cut -d'|' -f1 "$T/runs" | while read -r run; do
        awk -F'\t' -v run="$run" '{ print run "\t" $0 }' "$RIG_OUT/$run.tsv"
    done
} | gzip -9n > "$OUT/cases.tsv.gz"
# the console and the script's tables, with the rig's folders written as $SET (the set) and $D (the run's folder)
for f in console.txt results.tsv unrecorded.tsv; do
    sed -e "s#${SET_DIR:-@none@}#\$SET#g" -e "s#${RUN_DIR:-@none@}#\$D#g" "$RIG_OUT/$f" > "$OUT/$f"
done
column -t -s $'\t' "$OUT/counts.tsv" 2> /dev/null || cat "$OUT/counts.tsv"
column -t -s $'\t' "$OUT/vs_r112.tsv" 2> /dev/null || cat "$OUT/vs_r112.tsv"
echo "differing cases: $(($(wc -l < "$OUT/vs_r112_diff.tsv") - 1)); failing cases in the two dmt-q3f runs: $(($(wc -l < "$OUT/failing.tsv") - 1))"
