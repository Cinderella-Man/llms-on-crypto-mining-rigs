#!/usr/bin/env bash
# R197's tables from the deep lane's results folder (r197.sh's OUT), R176's chain
# (docs/data/2026-11-17_r176_ornith_step_map/run/tables.sh): every profiler file cut into one file per generation
# request (R121's split_trace.py); the one-token load's host profile through S37's step split and R176's host_split.py;
# each group profile through R49's verify_parts.py (R41's parts; Ornith's MTP draft is the graph with the MTP layer's
# eh_proj product); then R176's step_map.py twice, once over arm base's loads and once over arm ro's (each arm's loads
# named w<k>g<n> for it, the name it reads a group-profile load's width from), and compare.py over the two maps.
# CPU only, standard-library Python 3; one request in memory at a time.
# Env: S (this repository's root), OUT (the deep lane's results: requests.csv, hp-*.csv, st-*.jsonl, gp-*.csv,
# <load>.jsonl), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-19_r197_ornith_passes/run
M=$S/docs/data/2026-11-17_r176_ornith_step_map/run
SPLIT=$S/docs/data/2026-11-17_r121_flash_step_map/run/split_trace.py
SPL=$S/docs/data/2026-11-10_served_step_split/run/step_split.py
VP=$S/docs/data/2026-11-10_served_verify_parts/run/verify_parts.py
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
PS=$S/scripts/prof_summary.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d8192p1 d8192p2"
mkdir -p "$RES/pieces" "$RES/split" "$RES/host" "$RES/verify"
rm -f "$RES/DONE"
log() { echo "$(date -Is) $*" | tee -a "$RES/status.txt"; }
SPLITS='' HOSTS=''

for f in "$OUT"/hp-*.csv; do  # the one-token load
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
  $N "$M/host_split.py" "$OUT/st-$l.jsonl" --split "$RES/split/$l.tsv" --load "$l" --names "${NAMES// /,}" \
    --tsv "$RES/host/$l.tsv" > "$RES/host/$l.md"
  HOSTS="$HOSTS --host $l=$RES/host/$l.tsv"
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
done

for arm in b r; do  # step_map.py per arm: the arm's loads as w<k>g<n>, the one-token load as it is
  log "map, arm $arm"
  VERIFS=''
  for d in "$RES"/verify/w?"$arm"?; do
    [ -d "$d" ] || continue
    l=$(basename "$d"); VERIFS="$VERIFS --verify ${l:0:2}g${l:3}=$d"
  done
  awk -F, -v a="$arm" 'NR == 1 || $2 == "w1h1" || $2 ~ "^w[0-9]" a "[0-9]$"' "$OUT/requests.csv" \
    | sed -E "s/\bw([0-9])$arm([0-9])\b/w\1g\2/g" > "$RES/requests_$arm.csv"
  for r in p2 p1; do
    # shellcheck disable=SC2086
    $N "$M/step_map.py" --requests "$RES/requests_$arm.csv" $SPLITS $VERIFS $HOSTS --request $r \
      --tsv "$RES/map_${arm}_$r.tsv" > "$RES/map_${arm}_$r.md"
  done
done

log "compare"
for r in p2 p1; do
  $N "$D/compare.py" --base "$RES/map_b_$r.tsv" --ro "$RES/map_r_$r.tsv" --request "$r" \
    --requests "$OUT/requests.csv" --replies "$OUT" \
    --r176 "$S/docs/data/2026-11-17_r176_ornith_step_map/map.tsv.gz" \
    --r176-requests "$S/docs/data/2026-11-17_r176_ornith_step_map/requests.csv" \
    --tsv "$RES/compare_$r.tsv" > "$RES/compare_$r.md"
done
log "done"
touch "$RES/DONE"
