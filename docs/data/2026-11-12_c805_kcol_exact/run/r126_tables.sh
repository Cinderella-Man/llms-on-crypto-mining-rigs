#!/usr/bin/env bash
# r126_tables.sh: R126's tables from rig_c805.sh's output folder: every bit-identity arm with its comparisons,
# differences, exit status, what passing means and the verdict; every product that differs in any arm; and the perf
# step's us/run per arm, pass and token count, with each arm against `off` (and the C726 arms against `q3off`).
# Usage: OUT=<rig_c805.sh's OUT> DEST=<folder for the tables> bash r126_tables.sh
# Writes DEST/rig_bitid.tsv, DEST/rig_diffs.tsv, DEST/rig_perf.tsv, DEST/rig_perf_vs_off.tsv and DEST/rig_summary.txt.
set -euo pipefail
OUT=${OUT:?OUT=<rig_c805.sh output folder>}
DEST=${DEST:?DEST=<folder>}
mkdir -p "$DEST"

# one line per bit-identity arm. status.txt has "<time> bitid <mode> <arm> start (<switches>; must: <0|differ>)" and
# "<time> bitid <mode> <arm> exit <rc>: <last line>"; each arm's own file has the counts. Passing: 0 comparisons with
# differences and exit 0 where must is 0; differences and exit 1 where must is differ (the controls and the faults).
awk -v dir="$OUT/bitid" '
    BEGIN { OFS = "\t"; print "mode", "arm", "switches", "comparisons", "with differences", "exit", "must", "verdict" }
    $2 == "bitid" && $5 == "start" {
        sw = $0; sub(/^[^(]*\(/, "", sw); sub(/\)$/, "", sw)
        must = sw; sub(/^.*; must: /, "", must); sub(/; must: [^;]*$/, "", sw)
        key = $3 "-" $4; S[key] = sw; MU[key] = must; order[++n] = key; M[key] = $3; A[key] = $4
    }
    $2 == "bitid" && $5 == "exit" { rc = $6; sub(/:$/, "", rc); R[$3 "-" $4] = rc }
    END {
        for (i = 1; i <= n; i++) {
            k = order[i]; f = dir "/" k ".txt"; cmp = "-"; dif = "-"
            while ((getline line < f) > 0)
                if (line ~ /^[0-9]+ comparisons, [0-9]+ with differences$/) { split(line, w, " "); cmp = w[1]; dif = w[3] }
            close(f)
            if (MU[k] == "differ") v = (dif != "-" && dif > 0 && R[k] == "1") ? "ok" : "WRONG"
            else                   v = (dif == "0" && R[k] == "0") ? "ok" : "WRONG"
            print M[k], A[k], S[k], cmp, dif, R[k], MU[k], v
        }
    }' "$OUT/status.txt" > "$DEST/rig_bitid.tsv"

# every product with differences, in every arm (the controls' and the faults' included)
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
} > "$DEST/rig_diffs.tsv"

# perf: rig_c805.sh's us_run.tsv (arm, pass, type, m, k, n, us/run) as one line per pass and token count, one column
# per arm, in the order the arms ran
ARMS=$(awk -F'\t' '!s[$1]++ { printf "%s ", $1 }' "$OUT/perf/us_run.tsv")
ORDER="off x1 x2 q3off q3x1 q3x2"
COLS=""
for a in $ORDER; do [[ " $ARMS " == *" $a "* ]] && COLS="$COLS$a "; done
for a in $ARMS; do [[ " $COLS " == *" $a "* ]] || COLS="$COLS$a "; done
awk -F'\t' -v cols="$COLS" '
    BEGIN { OFS = "\t"; nc = split(cols, C, " "); h = "pass\ttype\tm\tk\tn"; for (i = 1; i <= nc; i++) h = h "\t" C[i] " us/run"; print h }
    { key = $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6; if (!(key in seen)) { seen[key] = 1; K[++n] = key }; V[key, $1] = $7 }
    END {
        for (j = 1; j <= n; j++) { line = K[j]; for (i = 1; i <= nc; i++) line = line "\t" ((K[j], C[i]) in V ? V[K[j], C[i]] : "-"); print line }
    }' <(sort -t$'\t' -k2,2n -k6,6n "$OUT/perf/us_run.tsv") > "$DEST/rig_perf.tsv"

# each arm against its reference in the same pass and token count: x1, x2 against off; q3x1, q3x2 against q3off;
# q3off against off. Difference in us/run and in % of the reference (negative: faster)
awk -F'\t' '
    BEGIN { OFS = "\t"; print "pass", "n", "arm", "us/run", "reference", "reference us/run", "difference us", "difference %"
            ref["x1"] = "off"; ref["x2"] = "off"; ref["q3x1"] = "q3off"; ref["q3x2"] = "q3off"; ref["q3off"] = "off" }
    { V[$2, $6, $1] = $7; if (!(($2, $6) in seen)) { seen[$2, $6] = 1; P[++np] = $2 SUBSEP $6 } }
    END {
        na = split("x1 x2 q3off q3x1 q3x2", A, " ")
        for (j = 1; j <= np; j++) {
            split(P[j], pn, SUBSEP)
            for (i = 1; i <= na; i++) {
                a = A[i]; r = ref[a]
                if (!((pn[1], pn[2], a) in V) || !((pn[1], pn[2], r) in V)) continue
                v = V[pn[1], pn[2], a]; rv = V[pn[1], pn[2], r]
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%+.2f\t%+.2f\n", pn[1], pn[2], a, v, r, rv, v - rv, 100 * (v - rv) / rv
            }
        }
    }' <(sort -t$'\t' -k2,2n -k6,6n "$OUT/perf/us_run.tsv") > "$DEST/rig_perf_vs_off.tsv"

{
    awk -F'\t' 'NR > 1 { n[$8]++ } END { printf "bitid: %d arms, %d ok, %d WRONG\n", NR - 1, n["ok"], n["WRONG"] }' "$DEST/rig_bitid.tsv"
    awk -F'\t' 'NR > 1 { printf "bitid %s: %s %s, %s of %s with differences (exit %s, must %s)\n", $8, $1, $2, $5, $4, $6, $7 }' "$DEST/rig_bitid.tsv"
    echo "perf: $(($(wc -l < "$OUT/perf/us_run.tsv"))) us/run lines; arms $COLS"
    awk -F'\t' 'NR > 1 && $2 >= 6 { printf "perf pass %s n=%s %s %s us/run against %s %s: %s us (%s %%)\n", $1, $2, $3, $4, $5, $6, $7, $8 }' "$DEST/rig_perf_vs_off.tsv"
} > "$DEST/rig_summary.txt"
cat "$DEST/rig_summary.txt"
