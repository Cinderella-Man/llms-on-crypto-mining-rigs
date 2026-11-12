#!/usr/bin/env bash
# R112 (Q1359): the tables of the v4day replay on a P106. The whole set with the -kern build, then the dmt suite again
# with the -day build that recorded the set, each through docs/data/2026-11-11_v4night_replay/run/tables.sh and
# run/suites.sh; the consoles with the rig's folders written as $W (the run's folder), $SET and $B (the build folder).
# Env: REPO (this repository's root), OUT (the -kern replay's output folder), OUT_DAY (the -day dmt rerun's), SET (the
#   set's folder: manifest.tsv, runs.tsv), DEST (the data folder), RIG_W, RIG_SET, RIG_B (the rig's run folder, set
#   folder and builds folder as the consoles print them)
# Writes in DEST: results.tsv, verdicts_by_run.tsv, big_cases.tsv, failing_cases.tsv, cases.tsv.gz, coverage.tsv,
#   unreplayed.tsv, suites.tsv, console.txt (the -kern run), results_day.tsv, verdicts_by_run_day.tsv, suites_day.tsv,
#   console_day.txt (the -day dmt rerun)
set -euo pipefail
REPO=${REPO:?}; OUT=${OUT:?}; OUT_DAY=${OUT_DAY:?}; SET=${SET:?}; DEST=${DEST:?}
RIG_W=${RIG_W:?}; RIG_SET=${RIG_SET:?}; RIG_B=${RIG_B:?}
T=$REPO/docs/data/2026-11-11_v4night_replay/run/tables.sh
S=$REPO/docs/data/2026-11-12_r112_v4day_replay/run/suites.sh
mkdir -p "$DEST"
anon() { sed -e "s|$RIG_B|\$B|g" -e "s|$RIG_SET|\$SET|g" -e "s|$RIG_W|\$W|g" "$1"; }
OUT=$OUT SET=$SET DEST=$DEST bash "$T"
OUT=$OUT DEST=$DEST bash "$S" > /dev/null
anon "$OUT/console.txt" > "$DEST/console.txt"
tmp=$(mktemp -d)
OUT=$OUT_DAY SET=$SET DEST=$tmp bash "$T"
OUT=$OUT_DAY DEST=$tmp bash "$S" > /dev/null
cp "$tmp/results.tsv" "$DEST/results_day.tsv"
cp "$tmp/verdicts_by_run.tsv" "$DEST/verdicts_by_run_day.tsv"
cp "$tmp/suites.tsv" "$DEST/suites_day.tsv"
anon "$OUT_DAY/console.txt" > "$DEST/console_day.txt"
rm -rf "$tmp"
echo "== -kern"; cat "$DEST/suites.tsv"; echo "== -day, dmt"; cat "$DEST/suites_day.tsv"
