#!/usr/bin/env bash
# R125's tables from the deep lane's results folder (r125.sh's OUT), R121's way (../../2026-11-17_r121_flash_step_map/
# run/tables.sh): every profiler file cut into one file per generation request (R121's split_trace.py), then per
# request S37's step split of the host profiles (2026-11-10_served_step_split/run/step_split.py) and R49's verify pass
# by part of the group profiles (2026-11-10_served_verify_parts/run/verify_parts.py, R41's parts); then the two files
# compared part by part and product by product (products.py) and the deep lane's own tables (deep_table.py).
# CPU only, standard-library Python 3, one request in memory at a time.
# Env: S (this repository's root), OUT (the deep lane's results), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-25_r125_gmix_step/run
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
SPL=$S/docs/data/2026-11-10_served_step_split/run/step_split.py
VP=$S/docs/data/2026-11-10_served_verify_parts/run/verify_parts.py
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
PS=$S/scripts/prof_summary.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d0s d8192p1 d8192p2 d8192s"
NAMES_C=${NAMES// /,}
mkdir -p "$RES/pieces" "$RES/split" "$RES/verify"
rm -f "$RES/DONE"
log() { echo "$(date -Is) $*" | tee -a "$RES/status.txt"; }

for f in "$OUT"/hp-*.csv; do
  [ -e "$f" ] || continue
  l=$(basename "$f" .csv); l=${l#hp-}
  log "host profile $l"
  $N "$R121/split_trace.py" "$f" --out "$RES/pieces/$l" --names "$NAMES_C" > "$RES/pieces/$l.txt" || log "$l: split_trace.py: not every request found"
  first=1
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    $N "$SPL" "$RES/pieces/$l/$n.csv" --load "$l" --requests "$n" --tsv "$RES/split/.$l-$n.tsv" > "$RES/split/$l-$n.md"
    if [ $first = 1 ]; then cat "$RES/split/.$l-$n.tsv"; first=0; else tail -n +2 "$RES/split/.$l-$n.tsv"; fi
    rm -f "$RES/split/.$l-$n.tsv"
  done > "$RES/split/$l.tsv"
  rm -rf "$RES/pieces/$l"
done

for f in "$OUT"/gp-*.csv; do
  [ -e "$f" ] || continue
  l=$(basename "$f" .csv); l=${l#gp-}
  log "group profile $l"
  $N "$R121/split_trace.py" "$f" --out "$RES/pieces/$l" --names "$NAMES_C" > "$RES/pieces/$l.txt" || log "$l: split_trace.py: not every request found"
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    mkdir -p "$RES/verify/$l/$n"
    $N "$VP" --kpass-parts "$KP" --prof-summary "$PS" --event-us 7.6 --out "$RES/verify/$l/$n" \
      "$RES/pieces/$l/$n.csv" > "$RES/verify/$l/$n/report.md" 2> "$RES/verify/$l/$n/stderr.txt"
    rm -f "$RES/verify/$l/$n/verify-r1.csv"  # the kept rows: a copy of the piece's
  done
  rm -rf "$RES/pieces/$l"
done

log "products"
[ -d "$RES/verify/pg" ] && [ -d "$RES/verify/gg" ] &&
  $N "$D/products.py" --base pmix=$RES/verify/pg --other gmix=$RES/verify/gg --requests d0p2,d8192p2,d0p1,d8192p1 \
    --tsv "$RES/products.tsv" > "$RES/products.md"
log "deep lane tables"
$N "$S/bench/deep_lane/deep_table.py" "$OUT" --control pu --out "$RES/table.md" --tsv "$RES/lane_classes.tsv" > /dev/null
log "done"
touch "$RES/DONE"
