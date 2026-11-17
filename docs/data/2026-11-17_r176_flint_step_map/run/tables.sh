#!/usr/bin/env bash
# R176's tables from the deep lane's results folder (r176.sh's OUT): flint_map.py's map (every request, the one-token
# step of the preset, a one-token pass by part, what grows with depth), width_curve.py's passes of k tokens step by
# step, and C832's card log
# (scripts/card_table.py: each card's clocks, power, temperature and throttle reasons) over the run and per load.
# CPU only, standard-library Python 3; one profile piece in memory at a time.
# Env: S (this repository's root), OUT (the deep lane's results: requests.csv, requests-<load>.csv, hp-*.csv,
# gp-*.csv, st-*.jsonl, loads.tsv, cards.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r176_flint_step_map/run
N="nice -n 19 python3 -I"
mkdir -p "$RES/cards"
rm -f "$RES/DONE"
$N "$D/flint_map.py" --out-dir "$OUT" --md "$RES/map.md" --tsv "$RES/map.tsv" 2> "$RES/map_notes.txt" || exit 1
# the passes of several tokens, step by step (the speculative arms' step traces paired with their host profiles)
$N "$D/width_curve.py" --out-dir "$OUT" --loads s8,s4,s2 --md "$RES/widths.md" --tsv "$RES/widths.tsv" || exit 1
$N "$S/scripts/card_table.py" "$OUT/cards.csv" --out "$RES/cards/all.md" || exit 1
# each load from its start to the next load's start (loads.tsv's started column)
mapfile -t L < <(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "started") c = i; next } { print $1 "\t" $c }' "$OUT/loads.tsv")
for i in "${!L[@]}"; do
  IFS=$'\t' read -r load since <<< "${L[i]}"
  until=''
  [ $((i + 1)) -lt ${#L[@]} ] && until=$(cut -f2 <<< "${L[i + 1]}")
  $N "$S/scripts/card_table.py" "$OUT/cards.csv" --since "$since" ${until:+--until "$until"} --out "$RES/cards/$load.md"
done
touch "$RES/DONE"
