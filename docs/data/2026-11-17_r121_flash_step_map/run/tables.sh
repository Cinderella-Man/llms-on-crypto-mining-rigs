#!/usr/bin/env bash
# R121's tables from the deep lane's results folder (r121.sh's OUT): every profiler file cut into one file per
# generation request (split_trace.py), then per request S37's step split of the host profiles
# (2026-11-10_served_step_split/run/step_split.py) and R49's verify pass by part of the group profiles
# (2026-11-10_served_verify_parts/run/verify_parts.py, R41's parts), the first request against its repeat from the
# step traces (first_request.py, with R86's align_t0.py for the host profile's clock and C405's step_gap.py), and the
# map (step_map.py). CPU only, standard-library Python 3; one request in memory at a time, so a 4 GB host runs it.
# Env: S (this repository's root), OUT (the deep lane's results: requests.csv, requests-<load>.csv, hp-*.csv,
# st-*.jsonl, gp-*.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r121_flash_step_map/run
SPL=$S/docs/data/2026-11-10_served_step_split/run/step_split.py
VP=$S/docs/data/2026-11-10_served_verify_parts/run/verify_parts.py
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
PS=$S/scripts/prof_summary.py
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
GAP=$S/docs/data/2026-11-11_c405_server_gap/run/step_gap.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d0s d4096p1 d4096p2 d4096s d8192p1 d8192p2 d8192s"
mkdir -p "$RES/pieces" "$RES/split" "$RES/first" "$RES/verify"
rm -f "$RES/DONE"
log() { echo "$(date -Is) $*" | tee -a "$RES/status.txt"; }
SPLITS='' VERIFS=''

for f in "$OUT"/hp-*.csv; do
  l=$(basename "$f" .csv); l=${l#hp-}
  log "host profile $l"
  $N "$D/split_trace.py" "$f" --out "$RES/pieces/$l" > "$RES/pieces/$l.txt" || log "$l: split_trace.py: not every request found"
  first=1
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    $N "$SPL" "$RES/pieces/$l/$n.csv" --load "$l" --requests "$n" --tsv "$RES/split/.$l-$n.tsv" > "$RES/split/$l-$n.md"
    if [ $first = 1 ]; then cat "$RES/split/.$l-$n.tsv"; first=0; else tail -n +2 "$RES/split/.$l-$n.tsv"; fi
    rm -f "$RES/split/.$l-$n.tsv"
  done > "$RES/split/$l.tsv"
  SPLITS="$SPLITS --split $l=$RES/split/$l.tsv"
  # the host profile's 0 ms on the step trace's clock, then the first request against its repeat
  $N "$AL" "$OUT/st-$l.jsonl" "$f" > "$RES/first/align-$l.md"
  t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/first/align-$l.md" | tail -n 1)
  log "$l: profile t0 $t0"
  $N "$D/first_request.py" --trace "$OUT/st-$l.jsonl" --requests "$OUT/requests-$l.csv" --profile "$f" \
    --profile-t0 "$t0" --tsv "$RES/first/steps-$l.tsv" --pairs-tsv "$RES/first/pairs-$l.tsv" > "$RES/first/first-$l.md"
  $N "$GAP" "$OUT/st-$l.jsonl" --profile "$f" --profile-t0 "$t0" > "$RES/first/gap-$l.md" 2>&1
  rm -rf "$RES/pieces/$l"
done

for f in "$OUT"/gp-*.csv; do
  l=$(basename "$f" .csv); l=${l#gp-}
  log "group profile $l"
  $N "$D/split_trace.py" "$f" --out "$RES/pieces/$l" > "$RES/pieces/$l.txt" || log "$l: split_trace.py: not every request found"
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    mkdir -p "$RES/verify/$l/$n"
    $N "$VP" --kpass-parts "$KP" --prof-summary "$PS" --event-us 7.6 --out "$RES/verify/$l/$n" \
      "$RES/pieces/$l/$n.csv" > "$RES/verify/$l/$n/report.md" 2> "$RES/verify/$l/$n/stderr.txt"
    rm -f "$RES/verify/$l/$n/verify-r1.csv"  # the kept rows: a copy of the piece's
  done
  # the first request at each depth against its repeat, launch group by launch group
  for dd in 0 4096 8192; do
    [ -f "$RES/pieces/$l/d${dd}p1.csv" ] && [ -f "$RES/pieces/$l/d${dd}p2.csv" ] || continue
    $N "$D/first_passes.py" "$RES/pieces/$l/d${dd}p1.csv" "$RES/pieces/$l/d${dd}p2.csv" \
      --tsv "$RES/first/passes-$l-d$dd.tsv" > "$RES/first/passes-$l-d$dd.md"
  done
  rm -rf "$RES/pieces/$l"
  VERIFS="$VERIFS --verify $l=$RES/verify/$l"
done

log "map"
# shellcheck disable=SC2086
$N "$D/step_map.py" --requests "$OUT/requests.csv" $SPLITS $VERIFS --tsv "$RES/map.tsv" > "$RES/map.md"
# shellcheck disable=SC2086
$N "$D/step_map.py" --requests "$OUT/requests.csv" $SPLITS $VERIFS --request p1 --tsv "$RES/map_p1.tsv" > "$RES/map_p1.md"
log "done"
touch "$RES/DONE"
