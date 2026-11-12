#!/usr/bin/env bash
# R112 (Q1359): the v4day replay per suite, from the per-run table that
# docs/data/2026-11-11_v4night_replay/run/tables.sh writes (verdicts_by_run.tsv) and each run's log.
# A run's suite is its name up to the first '-' (dmt, q5kmt, q3kmt, bf16x, f32x, glumt, q5k1r).
# Env: DEST (the folder holding verdicts_by_run.tsv; suites.tsv is written there), OUT (scripts/replay_checks.sh's output
#   folder: <run>.log), NAME (default suites.tsv: the file written in DEST)
# Writes: DEST/NAME: per suite, its runs; the runs that must pass, how many ended as expected, the cases each replayed
#   (every distinct count, with how many runs replayed it), and their case replays and OKs; the cases each run's filter
#   selected that the set holds no record of (the replay counts each as a failure); the fault runs, how many ended as
#   expected, and per fault run the cases that failed out of those replayed
set -euo pipefail
DEST=${DEST:?DEST=<folder with verdicts_by_run.tsv>}
OUT=${OUT:?OUT=<replay_checks.sh output folder>}
NAME=${NAME:-suites.tsv}
# per run: the replay's own count of selected cases with no record ("N selected by the filter have no record")
for f in "$OUT"/*.log; do
    n=$(grep -a -o '[0-9]* selected by the filter have no record' "$f" | tail -n 1 | grep -o '^[0-9]*' || true)
    printf '%s\t%s\n' "$(basename "$f" .log)" "${n:-0}"
done > "$DEST/.norec"
awk -F'\t' -v OFS='\t' '
    NR == FNR { norec[$1] = $2; next }
    FNR == 1 { next }
    { s = $1; sub(/-.*/, "", s)
      if (!(s in seen)) { seen[s] = 1; order[++ns] = s }
      runs[s]++
      nr = ($1 in norec) ? norec[$1] : 0
      k2 = s SUBSEP nr; if (!(k2 in nn)) { nl[s] = nl[s] (nl[s] == "" ? "" : ",") nr }; nn[k2]++
      if ($2 == "pass") {
          pr[s]++; if ($3 == "ok") pok[s]++
          key = s SUBSEP $4; if (!(key in cn)) { cl[s] = cl[s] (cl[s] == "" ? "" : ",") $4 }; cn[key]++
          cr[s] += $4; ok[s] += $5
      } else {
          fr[s]++; if ($3 == "ok") fok[s]++
          fl[s] = fl[s] (fl[s] == "" ? "" : ",") $1 ":" $6 "/" $4
      } }
    END {
        print "suite", "runs", "pass_runs", "pass_runs_ok", "cases_a_pass_run", "case_replays_in_pass_runs", \
              "OK_in_pass_runs", "unrecorded_a_run", "fault_runs", "fault_runs_ok", "fault_runs_failed_of_replayed"
        for (i = 1; i <= ns; i++) { s = order[i]
            n = split(cl[s], c, ","); cs = ""
            for (j = 1; j <= n; j++) cs = cs (cs == "" ? "" : ",") c[j] "x" cn[s SUBSEP c[j]]
            n = split(nl[s], c, ","); us = ""
            for (j = 1; j <= n; j++) us = us (us == "" ? "" : ",") c[j] "x" nn[s SUBSEP c[j]]
            print s, runs[s], pr[s] + 0, pok[s] + 0, (cs == "" ? "-" : cs), cr[s] + 0, ok[s] + 0, us, fr[s] + 0, \
                  fok[s] + 0, (fl[s] == "" ? "-" : fl[s]) }
    }' "$DEST/.norec" "$DEST/verdicts_by_run.tsv" > "$DEST/$NAME"
rm -f "$DEST/.norec"
cat "$DEST/$NAME"
