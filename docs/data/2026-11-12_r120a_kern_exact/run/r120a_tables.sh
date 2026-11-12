#!/usr/bin/env bash
# r120a_tables.sh: R120a's tables from rig_kern.sh's output folder (STEPS='replay bitid') and r120a_diag.sh's: the
# replay's runs, every bit-identity arm with its comparisons, differences, exit status, what passing means and the
# verdict, and every product that differs in any arm.
# Usage: OUT=<rig_kern.sh's OUT> DIAG=<r120a_diag.sh's OUT> DEST=<folder for the tables> bash r120a_tables.sh
# Writes DEST/replay.tsv (scripts/replay_checks.sh's results.tsv), DEST/bitid.tsv, DEST/diag.tsv, DEST/diffs.tsv and
# DEST/summary.txt.
set -euo pipefail
OUT=${OUT:?OUT=<rig_kern.sh output folder>}
DIAG=${DIAG:?DIAG=<r120a_diag.sh output folder>}
DEST=${DEST:?DEST=<folder>}
mkdir -p "$DEST"

# the replay: one line per run, as replay_checks.sh wrote it
cp "$OUT/replay/results.tsv" "$DEST/replay.tsv"

# one line per bit-identity arm: <status.txt> <folder of the arms' files> <file name prefix ("" or "bitid/")>.
# status.txt has "<time> bitid <mode> <arm> start (<switches>)" and "<time> bitid <mode> <arm> exit <rc>: <last line>";
# each arm's own file has the counts. Passing: 0 comparisons with differences and exit 0, except the faults and controls
# (arm names fault, *-fault, control), which must have differences and exit 1.
arms() {
    awk -v dir="$2" '
        BEGIN { OFS = "\t"; print "mode", "arm", "switches", "comparisons", "with differences", "declined", "exit", "expect", "verdict" }
        $2 == "bitid" && $5 == "start" {
            sw = $0; sub(/^[^(]*\(/, "", sw); sub(/\)$/, "", sw)
            key = $3 "-" $4; S[key] = sw; order[++n] = key; M[key] = $3; A[key] = $4
        }
        $2 == "bitid" && $5 == "exit" { rc = $6; sub(/:$/, "", rc); R[$3 "-" $4] = rc }
        END {
            for (i = 1; i <= n; i++) {
                k = order[i]; f = dir "/" k ".txt"; cmp = "-"; dif = "-"; dec = "-"
                while ((getline line < f) > 0) {
                    if (line ~ /^[0-9]+ comparisons, [0-9]+ with differences$/) { split(line, w, " "); cmp = w[1]; dif = w[3] }
                    if (line ~ /declined by the read-once kernel/) { split(line, w, " "); dec = w[1] }
                }
                close(f)
                neg = (A[k] ~ /(^|-)fault$/ || A[k] == "control")
                exp_ = neg ? "differ" : "0"
                if (neg) v = (dif != "-" && dif > 0 && R[k] == "1") ? "ok" : "WRONG"
                else     v = (dif == "0" && R[k] == "0") ? "ok" : "WRONG"
                print M[k], A[k], S[k], cmp, dif, dec, R[k], exp_, v
            }
        }' "$1"
}
arms "$OUT/status.txt" "$OUT/bitid" > "$DEST/bitid.tsv"
arms "$DIAG/status.txt" "$DIAG" > "$DEST/diag.tsv"

# every product with differences, in every arm of both runs (the faults' and controls' included)
{
    printf 'run\tmode\tarm\tcomparison\ttypes\tn\tm\tk\tvalues that differ\tvalues\tmax rel\n'
    for f in "$OUT"/bitid/*.txt "$DIAG"/*-*.txt; do
        b=$(basename "$f" .txt)
        case $f in "$OUT"/*) run=main ;; *) run=diag ;; esac
        mode=${b%%-*} arm=${b#*-}
        { grep 'DIFFERENT' "$f" || true; } | sed -E 's/ +/ /g; s/ ?: DIFFERENT \(/ /; s/ of / /; s/ values differ, max rel / /; s/\)$//' |
            awk -v run="$run" -v mode="$mode" -v arm="$arm" 'BEGIN { OFS = "\t" } {
                sub(/^n=/, "", $3); sub(/^m=/, "", $4); sub(/^k=/, "", $5)
                print run, mode, arm, $1, $2, $3, $4, $5, $6, $7, $8 }'
    done
} > "$DEST/diffs.tsv"

{
    echo "replay: $(tail -n 1 "$OUT/replay/console.txt")"
    awk -F'\t' 'NR > 1 { n[$9]++ } END { printf "bitid: %d arms, %d ok, %d WRONG\n", NR - 1, n["ok"], n["WRONG"] }' "$DEST/bitid.tsv"
    awk -F'\t' 'NR > 1 { printf "bitid %s: %s %s, %s of %s with differences (exit %s, expect %s)\n", $9, $1, $2, $5, $4, $7, $8 }' "$DEST/bitid.tsv" |
        grep WRONG || true
    awk -F'\t' 'NR > 1 { printf "diag %s %s: %s of %s with differences\n", $1, $2, $5, $4 }' "$DEST/diag.tsv"
} > "$DEST/summary.txt"
cat "$DEST/summary.txt"
