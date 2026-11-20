#!/bin/bash
# tables.sh: one half's tables (R226) from the two folders half_check.sh left on the rig, copied off it: WORK (the
# kit's work folder: out/apply_steps.csv, wd.csv) and OUT (model_check.sh's folder: model.log). Files go to
# <this folder>/<half>/; halves.md is rebuilt from every half folder and R224's published model_rounds.tsv (the stock
# rounds every half is compared with).
#
#   bash run/tables.sh <half: a, b ...> <WORK copy> <OUT copy> <this folder>
#
# R209's model_rounds.py and R203's flash_cards.py, unchanged (the card and watchdog tables drop their temperature
# column, as R224's did), R217's greedy_diff.py on any differing round, and halves.py. Standard library Python only.
set -euo pipefail
H=${1:?half} WORK=${2:?WORK copy} OUT=${3:?OUT copy} D=${4:?this folder}
HERE=$(cd "$(dirname "$0")/../.." && pwd)    # docs/data
MR=$HERE/2026-11-19_r209_oc_v5/run/model_rounds.py
FC=$HERE/2026-11-18_r203_ornith_offsets/run/flash_cards.py
mkdir -p "$D/$H"
cp "$WORK/out/apply_steps.csv" "$D/$H/apply_steps.csv"
python3 -I "$FC" apply "$D/$H/apply_steps.csv" > "$D/$H/apply.md"
python3 -I "$(dirname "$0")/halves.py" load "$OUT/model.log" > "$D/$H/load.tsv"
python3 -I "$MR" rounds "$OUT/model.log" > "$D/$H/model_rounds.tsv"
python3 -I "$MR" cards "$OUT/model.log" "$WORK/wd.csv" |
  awk -F'|' 'BEGIN { OFS = "|" } NF > 3 { $4 = ""; sub(/\|\|/, "|") } { print }' > "$D/$H/model_cards.md"
python3 -I "$MR" whole "$WORK/wd.csv" |
  awk -F'|' 'BEGIN { OFS = "|" } NF > 3 { $3 = ""; sub(/\|\|/, "|") } { print }' > "$D/$H/watchdog.md"
# a differing round against stock's round 1, prompt by prompt (R217's greedy_diff.py reads a climb's card folders, so
# each round goes in as card 0 and the card column is dropped)
T=$(mktemp -d); mkdir -p "$T/out/card0" "$T/stock/card0"
cp "$OUT/greedy-stock1.json" "$T/stock/card0/greedy-stock.json"
n=0
for j in "$OUT"/greedy-profile*.json; do
  [ -e "$j" ] || continue
  t=$(basename "$j" .json)
  [ "$(cat "$OUT/$t.sha")" = "$(cat "$OUT/greedy-stock1.sha")" ] && continue    # the stock hash: skipped
  cp "$j" "$T/out/card0/$t.json"; echo DIFFERS > "$T/out/card0/$t.out"; n=$((n + 1))
done
if [ $n -gt 0 ]; then
  python3 -I "$HERE/2026-11-19_r217_climb_rig1/run/greedy_diff.py" "$T/out" "$T/diff.csv" "$T/stock" "r226$H-ornith"
  cut -d, -f1,3- "$T/diff.csv" > "$D/$H/ornith_differs.csv"
fi
rm -rf "$T"
args=()
for f in "$D"/*/load.tsv; do args+=("$(basename "$(dirname "$f")"):$(dirname "$f")"); done
python3 -I "$(dirname "$0")/halves.py" table "$HERE/2026-11-20_r224_apply_rig1/model_rounds.tsv" "${args[@]}" > "$D/halves.md"
