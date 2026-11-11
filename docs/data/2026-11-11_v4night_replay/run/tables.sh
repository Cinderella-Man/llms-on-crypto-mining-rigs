#!/usr/bin/env bash
# The tables of the v4night replay (Q1321, R89 step 2) from scripts/replay_checks.sh's output folder.
# Env: OUT (replay_checks.sh's OUT: results.tsv and one <run>.tsv per run), SET (the replay set: manifest.tsv, runs.tsv),
#   DEST (where the tables go)
# Writes: DEST/results.tsv (replay_checks.sh's, unchanged), DEST/verdicts_by_run.tsv (per run: its expectation, its
#   verdict and its cases by their per-case verdict), DEST/big_cases.tsv (each large-activation product of the set,
#   which the CPU backend does not take: in how many runs it replayed and with what verdict), DEST/failing_cases.tsv
#   (every case not OK in a run expected to pass: run, switches, case id, case, verdict), DEST/cases.tsv.gz (every
#   replayed case of every run: run, case id, case, verdict, errors), DEST/coverage.tsv (per op: the set's cases, how
#   many a run expected to pass replayed and passed), DEST/unreplayed.tsv (the set's cases no run expected to pass
#   replayed)
set -euo pipefail
OUT=${OUT:?OUT=<replay_checks.sh output folder>}
SET=${SET:?SET=<the replay set folder>}
DEST=${DEST:?DEST=<table folder>}
mkdir -p "$DEST"
cp "$OUT/results.tsv" "$DEST/results.tsv"
# results.tsv's run, switches, expect and verdict, split on a non-blank separator: read with a tab IFS would merge the
# empty fields (memcheck errors, known failures) and shift the verdict
runs() { tail -n +2 "$OUT/results.tsv" | awk -F'\t' -v OFS=$'\x1f' '{ print $1, $2, $3, $8 }'; }

# per run: expect and verdict from results.tsv, per-case verdict counts from <run>.tsv
{
    printf 'run\texpect\tverdict\tcases\tOK\tFAIL\tother\tother_verdicts\n'
    runs | while IFS=$'\x1f' read -r run sw expect v; do
        f=$OUT/$run.tsv
        if [ -f "$f" ]; then
            awk -F'\t' -v run="$run" -v expect="$expect" -v v="$v" '
                { n++; if ($3 == "OK") ok++; else if ($3 == "FAIL") fail++; else { oth++; ov[$3]++ } }
                END { s = ""; for (k in ov) s = s (s == "" ? "" : ",") k "=" ov[k]
                      printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n", run, expect, v, n, ok, fail, oth, (s == "" ? "-" : s) }' "$f"
        else
            printf '%s\t%s\t%s\t0\t0\t0\t0\tno per-case table\n' "$run" "$expect" "$v"
        fi
    done
} > "$DEST/verdicts_by_run.tsv"

# the large-activation products: every case of the manifest whose string has big=, against every run's table
awk -F'\t' '$2 ~ /big=/ { print $1 "\t" $2 }' "$SET/manifest.tsv" | sort > "$DEST/.big"
{
    printf 'id\tcase\truns\tOK\tFAIL\tother\truns_not_OK\n'
    for f in "$OUT"/*.tsv; do
        [ "$(basename "$f")" = results.tsv ] && continue
        r=$(basename "$f" .tsv)
        awk -F'\t' -v r="$r" '{ print $1 "\t" r "\t" $3 }' "$f"
    done | awk -F'\t' '
        NR == FNR { big[$1] = $2; order[++nb] = $1; next }
        ($1 in big) { n[$1]++; if ($3 == "OK") ok[$1]++; else if ($3 == "FAIL") { fail[$1]++; bad[$1] = bad[$1] (bad[$1] == "" ? "" : ",") $2 } else { oth[$1]++; bad[$1] = bad[$1] (bad[$1] == "" ? "" : ",") $2 "=" $3 } }
        END { for (i = 1; i <= nb; i++) { id = order[i]
                  printf "%s\t%s\t%d\t%d\t%d\t%d\t%s\n", id, big[id], n[id], ok[id], fail[id], oth[id], (bad[id] == "" ? "-" : bad[id]) } }' \
        "$DEST/.big" -
} > "$DEST/big_cases.tsv"
rm -f "$DEST/.big"

# every case not OK in a run expected to pass
{
    printf 'run\tswitches\tid\tcase\tverdict\n'
    runs | while IFS=$'\x1f' read -r run sw expect v; do
        [ "$expect" = pass ] && [ -f "$OUT/$run.tsv" ] || continue
        awk -F'\t' -v run="$run" -v sw="$sw" '$3 != "OK" { printf "%s\t%s\t%s\t%s\t%s\n", run, sw, $1, $2, $3 }' "$OUT/$run.tsv"
    done
} > "$DEST/failing_cases.tsv"

# every replayed case of every run, in the run list's order
{
    printf 'run\tid\tcase\tverdict\terrors\n'
    runs | while IFS=$'\x1f' read -r run sw expect v; do
        if [ -f "$OUT/$run.tsv" ]; then
            awk -F'\t' -v run="$run" '{ printf "%s\t%s\t%s\t%s\t%s\n", run, $1, $2, $3, $4 }' "$OUT/$run.tsv"
        fi
    done
} | gzip -9 -n > "$DEST/cases.tsv.gz"

# the set's cases against the runs: per op, cases in the set, replayed in a run expected to pass (and OK there), in
# any run; and every case of the set that no run expected to pass replayed
gzip -dc "$DEST/cases.tsv.gz" | tail -n +2 | awk -F'\t' -v OFS='\t' '
    NR == FNR { if ($3 == "pass") pass[$1] = 1; next }
    { any[$2] = 1; if ($1 in pass) { inpass[$2] = 1; if ($4 == "OK") okpass[$2] = 1 } }
    END { for (id in any) print id, (id in inpass) ? 1 : 0, (id in okpass) ? 1 : 0 }' \
    <(tail -n +2 "$OUT/results.tsv") - | sort > "$DEST/.seen"
awk -F'\t' -v OFS='\t' '
    NR == FNR { inpass[$1] = $2; okpass[$1] = $3; next }
    { op = $2; sub(/\(.*/, "", op); n[op]++
      if ($1 in inpass) { any[op]++; if (inpass[$1]) p[op]++; if (okpass[$1]) ok[op]++ } }
    END { print "op", "cases", "in_pass_runs", "OK_in_pass_runs", "in_any_run"
          for (o in n) print o, n[o], p[o] + 0, ok[o] + 0, any[o] + 0 }' "$DEST/.seen" "$SET/manifest.tsv" \
    | { IFS= read -r h; echo "$h"; sort; } > "$DEST/coverage.tsv"
{
    printf 'id\tcase\tin_any_run\n'
    awk -F'\t' -v OFS='\t' 'NR == FNR { inpass[$1] = $2; next } !($1 in inpass) || !inpass[$1] { print $1, $2, ($1 in inpass) ? 1 : 0 }' \
        "$DEST/.seen" "$SET/manifest.tsv"
} > "$DEST/unreplayed.tsv"
rm -f "$DEST/.seen"
echo "tables in $DEST: $(($(wc -l < "$DEST/verdicts_by_run.tsv") - 1)) runs, $(($(wc -l < "$DEST/big_cases.tsv") - 1)) big= cases, $(($(wc -l < "$DEST/failing_cases.tsv") - 1)) cases not OK in pass runs, $(($(wc -l < "$DEST/unreplayed.tsv") - 1)) set cases in no pass run"
