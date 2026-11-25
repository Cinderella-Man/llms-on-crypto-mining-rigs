#!/usr/bin/env bash
# C1003's tables from the deep lane's results folder (c1003.sh's OUT): each host profile cut into one file per
# generation request (R121's split_trace.py), S37's step split per request (step_split.py), then c1003_tables.py
# (the step by value per load, the replies, the split's medians, the queued counts, the profile's count lines).
# CPU only, standard-library Python 3. Env: S (this repository's root), OUT (the deep lane's results), RES (the tables).
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r121_flash_step_map/run
SPL=$S/docs/data/2026-11-10_served_step_split/run/step_split.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d0s d8192p1 d8192p2 d8192s"
mkdir -p "$RES/pieces" "$RES/split"
for f in "$OUT"/hp-*.csv; do
  a=$(basename "$f" .csv); a=${a#hp-}
  l=${a}1
  $N "$D/split_trace.py" "$f" --out "$RES/pieces/$l" --names "$(echo $NAMES | tr ' ' ,)" > "$RES/pieces/$l.txt" \
    || echo "$l: split_trace.py: not every request found"
  first=1
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    $N "$SPL" "$RES/pieces/$l/$n.csv" --load "$l" --requests "$n" --tsv "$RES/split/.$l-$n.tsv" > "$RES/split/$l-$n.md"
    if [ $first = 1 ]; then cat "$RES/split/.$l-$n.tsv"; first=0; else tail -n +2 "$RES/split/.$l-$n.tsv"; fi
    rm -f "$RES/split/.$l-$n.tsv"
  done > "$RES/split/$l.tsv"
done
$N "$S/docs/data/2026-11-25_c1003_host_nowait_flash/run/c1003_tables.py" "$OUT" "$RES"
