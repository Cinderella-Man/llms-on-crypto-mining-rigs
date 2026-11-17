#!/usr/bin/env bash
# R176's tables from the deep lane's results folder (r176.sh's OUT), R121's chain
# (docs/data/2026-11-17_r121_flash_step_map/run/tables.sh) for Ornith: every profiler file cut into one file per
# generation request (R121's split_trace.py), then per request S37's step split of the host profiles
# (2026-11-10_served_step_split/run/step_split.py) and R49's verify pass by part of the group profiles
# (2026-11-10_served_verify_parts/run/verify_parts.py, R41's parts; Ornith's MTP draft is the graph with the MTP
# layer's eh_proj product), the one-token step's host part (host_split.py, beside step_split.py's GPU time), the
# drafted steps stage by stage (C405's step_gap.py, with R86's align_t0.py for the host profile's clock), and the map
# (step_map.py, R121's widened). CPU only, standard-library Python 3; one request in memory at a time.
# Env: S (this repository's root), OUT (the deep lane's results: requests.csv, hp-*.csv, st-*.jsonl, gp-*.csv),
# RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r176_ornith_step_map/run
SPLIT=$S/docs/data/2026-11-17_r121_flash_step_map/run/split_trace.py
SPL=$S/docs/data/2026-11-10_served_step_split/run/step_split.py
VP=$S/docs/data/2026-11-10_served_verify_parts/run/verify_parts.py
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
PS=$S/scripts/prof_summary.py
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
GAP=$S/docs/data/2026-11-11_c405_server_gap/run/step_gap.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d8192p1 d8192p2"
mkdir -p "$RES/pieces" "$RES/split" "$RES/host" "$RES/gap" "$RES/verify"
rm -f "$RES/DONE"
log() { echo "$(date -Is) $*" | tee -a "$RES/status.txt"; }
SPLITS='' VERIFS='' HOSTS=''

for f in "$OUT"/hp-*.csv; do
  l=$(basename "$f" .csv); l=${l#hp-}
  log "host profile $l"
  $N "$SPLIT" "$f" --out "$RES/pieces/$l" --names "${NAMES// /,}" > "$RES/pieces/$l.txt" \
    || log "$l: split_trace.py: not every request found"
  first=1
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    $N "$SPL" "$RES/pieces/$l/$n.csv" --load "$l" --requests "$n" --tsv "$RES/split/.$l-$n.tsv" > "$RES/split/$l-$n.md"
    if [ $first = 1 ]; then cat "$RES/split/.$l-$n.tsv"; first=0; else tail -n +2 "$RES/split/.$l-$n.tsv"; fi
    rm -f "$RES/split/.$l-$n.tsv"
  done > "$RES/split/$l.tsv"
  SPLITS="$SPLITS --split $l=$RES/split/$l.tsv"
  case $l in
    w1*)  # no draft: the step's host part
      $N "$D/host_split.py" "$OUT/st-$l.jsonl" --split "$RES/split/$l.tsv" --load "$l" --names "${NAMES// /,}" \
        --tsv "$RES/host/$l.tsv" > "$RES/host/$l.md"
      HOSTS="$HOSTS --host $l=$RES/host/$l.tsv" ;;
    *)  # the MTP draft: the step stage by stage on the trace's clock
      $N "$AL" "$OUT/st-$l.jsonl" "$f" > "$RES/gap/align-$l.md"
      t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/gap/align-$l.md" | tail -n 1)
      log "$l: profile t0 $t0"
      $N "$GAP" "$OUT/st-$l.jsonl" --profile "$f" --profile-t0 "$t0" --tsv "$RES/gap/steps-$l.tsv" \
        > "$RES/gap/gap-$l.md" 2>&1 ;;
  esac
  rm -rf "$RES/pieces/$l"
done

for f in "$OUT"/gp-*.csv; do
  l=$(basename "$f" .csv); l=${l#gp-}
  log "group profile $l"
  $N "$SPLIT" "$f" --out "$RES/pieces/$l" --names "${NAMES// /,}" > "$RES/pieces/$l.txt" \
    || log "$l: split_trace.py: not every request found"
  for n in $NAMES; do
    [ -f "$RES/pieces/$l/$n.csv" ] || continue
    mkdir -p "$RES/verify/$l/$n"
    $N "$VP" --kpass-parts "$KP" --prof-summary "$PS" --event-us 7.6 --other-regex eh_proj --out "$RES/verify/$l/$n" \
      "$RES/pieces/$l/$n.csv" > "$RES/verify/$l/$n/report.md" 2> "$RES/verify/$l/$n/stderr.txt"
    rm -f "$RES/verify/$l/$n/verify-r1.csv"  # the kept rows: a copy of the piece's
  done
  rm -rf "$RES/pieces/$l"
  VERIFS="$VERIFS --verify $l=$RES/verify/$l"
done

log "map"
# shellcheck disable=SC2086
$N "$D/step_map.py" --requests "$OUT/requests.csv" $SPLITS $VERIFS $HOSTS --tsv "$RES/map.tsv" > "$RES/map.md"
# shellcheck disable=SC2086
$N "$D/step_map.py" --requests "$OUT/requests.csv" $SPLITS $VERIFS $HOSTS --request p1 --tsv "$RES/map_p1.tsv" > "$RES/map_p1.md"
log "done"
touch "$RES/DONE"
