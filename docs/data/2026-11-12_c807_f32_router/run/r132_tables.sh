#!/usr/bin/env bash
# r132_tables.sh: R132's replay and bit-identity tables from rig_c807.sh's output folder (from R120a's r120a_tables.sh).
# Usage: OUT=<rig_c807.sh's OUT> DEST=<folder for the tables> bash r132_tables.sh
# Writes DEST/replay.tsv (scripts/replay_checks.sh's results.tsv), DEST/bitid.tsv (every arm: comparisons, differences,
# exit status, what passing means, the verdict), DEST/bitid_diffs.tsv (every product that differs in any arm) and
# DEST/rig_summary.txt.
# Passing, per bit-identity arm: on and on2 (f32r) and rows0 (f32) 0 comparisons with differences and exit 0; control
# (f32r, the switch off) differences and exit 1; limit (f32 at the default _ROWS) differences and exit 1, every one of
# them at a matrix of more than 32 rows (the products the limit leaves to cuBLAS).
set -euo pipefail
OUT=${OUT:?OUT=<rig_c807.sh output folder>}
DEST=${DEST:?DEST=<folder>}
mkdir -p "$DEST"

cp "$OUT/replay/results.tsv" "$DEST/replay.tsv"

# every product with differences, in every arm
{
    printf 'mode\tarm\tcomparison\ttypes\tn\tm\tk\tvalues that differ\tvalues\tmax rel\n'
    for f in "$OUT"/bitid/*.txt; do
        b=$(basename "$f" .txt)
        mode=${b%%-*} arm=${b#*-}
        { grep 'DIFFERENT' "$f" || true; } | sed -E 's/ +/ /g; s/ ?: DIFFERENT \(/ /; s/ of / /; s/ values differ, max rel / /; s/\)$//' |
            awk -v mode="$mode" -v arm="$arm" 'BEGIN { OFS = "\t" } {
                sub(/^n=/, "", $3); sub(/^m=/, "", $4); sub(/^k=/, "", $5)
                print mode, arm, $1, $2, $3, $4, $5, $6, $7, $8 }'
    done
} > "$DEST/bitid_diffs.tsv"

awk -v dir="$OUT/bitid" -v diffs="$DEST/bitid_diffs.tsv" '
    BEGIN {
        OFS = "\t"
        while ((getline line < diffs) > 0) {
            split(line, d, "\t"); if (d[1] == "mode") continue
            k = d[1] "-" d[2]; if (d[6] + 0 <= 32) low[k]++; else high[k]++
        }
        print "mode", "arm", "switches", "comparisons", "with differences", "of them at most 32 rows", "exit", "expect", "verdict"
    }
    $2 == "bitid" && $5 == "start" {
        sw = $0; sub(/^[^(]*\(/, "", sw); sub(/\)$/, "", sw)
        key = $3 "-" $4; S[key] = sw; order[++n] = key; M[key] = $3; A[key] = $4
    }
    $2 == "bitid" && $5 == "exit" { rc = $6; sub(/:$/, "", rc); R[$3 "-" $4] = rc }
    END {
        for (i = 1; i <= n; i++) {
            k = order[i]; f = dir "/" k ".txt"; cmp = "-"; dif = "-"
            while ((getline line < f) > 0)
                if (line ~ /^[0-9]+ comparisons, [0-9]+ with differences$/) { split(line, w, " "); cmp = w[1]; dif = w[3] }
            close(f)
            lo = low[k] + 0
            if (A[k] == "control")    { e = "differ"; v = (dif > 0 && R[k] == "1") ? "ok" : "WRONG" }
            else if (A[k] == "limit") { e = "differ above 32 rows only"; v = (dif > 0 && lo == 0 && R[k] == "1") ? "ok" : "WRONG" }
            else                      { e = "0"; v = (dif == "0" && R[k] == "0") ? "ok" : "WRONG" }
            print M[k], A[k], (S[k] == "" ? "(off)" : S[k]), cmp, dif, lo, R[k], e, v
        }
    }' "$OUT/status.txt" > "$DEST/bitid.tsv"

{
    echo "replay: $(tail -n 1 "$OUT/replay/console.txt")"
    awk -F'\t' 'NR > 1 { n[$8]++ } END { printf "replay runs: %d, ok %d\n", NR - 1, n["ok"] }' "$DEST/replay.tsv"
    grep -E ' f32x-fault' "$OUT/status.txt" | cut -d' ' -f2- || true
    awk -F'\t' 'NR > 1 { n[$9]++ } END { printf "bitid: %d arms, %d ok, %d WRONG\n", NR - 1, n["ok"], n["WRONG"] }' "$DEST/bitid.tsv"
    awk -F'\t' 'NR > 1 { printf "bitid %s %s: %s of %s with differences, %s at most 32 rows (exit %s, expect %s): %s\n", $1, $2, $5, $4, $6, $7, $8, $9 }' "$DEST/bitid.tsv"
} > "$DEST/rig_summary.txt"
cat "$DEST/rig_summary.txt"
