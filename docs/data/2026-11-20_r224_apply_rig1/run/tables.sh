#!/bin/bash
# tables.sh: R224's tables (step 5 of the first rig's profile) from the two folders apply_check.sh left on the rig,
# copied off it: WORK (the kit's work folder: out/apply_steps.csv, wd.csv) and OUT (model_check.sh's folder: model.log).
#
#   bash run/tables.sh <WORK copy> <OUT copy> <this folder>
#
# R209's model_rounds.py and R203's flash_cards.py, unchanged; the card and watchdog tables drop their temperature
# column. Standard library Python only.
set -euo pipefail
WORK=${1:?WORK copy} OUT=${2:?OUT copy} D=${3:?this folder}
HERE=$(cd "$(dirname "$0")/../.." && pwd)    # docs/data
MR=$HERE/2026-11-19_r209_oc_v5/run/model_rounds.py
FC=$HERE/2026-11-18_r203_ornith_offsets/run/flash_cards.py
cp "$WORK/out/apply_steps.csv" "$D/apply_steps.csv"
python3 -I "$FC" apply "$D/apply_steps.csv" > "$D/apply.md"
python3 -I "$MR" rounds "$OUT/model.log" > "$D/model_rounds.tsv"
python3 -I "$MR" summary "$OUT/model.log" > "$D/model_check.md"
# the third column of `cards` and the second of `whole` are the highest temperature: left out
python3 -I "$MR" cards "$OUT/model.log" "$WORK/wd.csv" |
  awk -F'|' 'BEGIN { OFS = "|" } NF > 3 { $4 = ""; sub(/\|\|/, "|") } { print }' > "$D/model_cards.md"
python3 -I "$MR" whole "$WORK/wd.csv" |
  awk -F'|' 'BEGIN { OFS = "|" } NF > 3 { $3 = ""; sub(/\|\|/, "|") } { print }' > "$D/watchdog.md"
# each differing round against stock's round 1, prompt by prompt, by R217's greedy_diff.py: it reads a climb's card
# folders, so each round goes in as card 0 and the card column (meaningless for a model on all nine cards) is dropped
T=$(mktemp -d); mkdir -p "$T/out/card0" "$T/stock/card0"
cp "$OUT/greedy-stock1.json" "$T/stock/card0/greedy-stock.json"
for j in "$OUT"/greedy-profile*.json; do
  t=$(basename "$j" .json)
  [ "$(cat "$OUT/$t.sha")" = "$(cat "$OUT/greedy-stock1.sha")" ] && continue    # the stock hash: skipped
  cp "$j" "$T/out/card0/$t.json"; echo DIFFERS > "$T/out/card0/$t.out"
done
python3 -I "$HERE/2026-11-19_r217_climb_rig1/run/greedy_diff.py" "$T/out" "$T/diff.csv" "$T/stock" r224-ornith
cut -d, -f1,3- "$T/diff.csv" > "$D/ornith_differs.csv"
rm -rf "$T"
