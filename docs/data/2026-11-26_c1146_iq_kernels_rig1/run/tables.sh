#!/usr/bin/env bash
# C1146's tables from the deep lane's results folder (c1146.sh's OUT), R125's way (../../2026-11-25_r125_gmix_step/run/
# tables.sh): every group profile cut into one file per generation request (R121's split_trace.py), then per request
# R49's verify pass by part (2026-11-10_served_verify_parts/run/verify_parts.py, R41's parts); then the routed gate/up
# product per type at each setting against R125's (types.py), the three settings compared product by product (R125's
# products.py: unset against 1, unset against 2, 1 against 2), the kernels beside their instruction counts and bytes
# (limits.py), the step pair over both lanes (steps.py) and the deep lane's own tables (deep_table.py: the first lane
# with u0 the control, the second with u2).
# CPU only, standard-library Python 3, one request in memory at a time.
# Env: S (this repository's root), OUT (the first lane's results), OUT2 (the second lane's: u2b and p0), RES (the
# tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${OUT2:?}" "${RES:?}"
D=$S/docs/data/2026-11-26_c1146_iq_kernels_rig1/run
R125=$S/docs/data/2026-11-25_r125_gmix_step
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
VP=$S/docs/data/2026-11-10_served_verify_parts/run/verify_parts.py
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
PS=$S/scripts/prof_summary.py
N="nice -n 19 python3 -I"
NAMES="d0p1 d0p2 d0s d8192p1 d8192p2 d8192s"
NAMES_C=${NAMES// /,}
mkdir -p "$RES/pieces" "$RES/verify"
rm -f "$RES/DONE"
log() { echo "$(date -Is) $*" | tee -a "$RES/status.txt"; }

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

log "types"
$N "$D/types.py" --arm unset=$RES/verify/g0 --arm c=$RES/verify/g1 --arm ptx=$RES/verify/g2 --ref r125=$R125/verify/gg \
  --requests d0p2,d8192p2,d0p1,d8192p1 --tsv "$RES/types.tsv" > "$RES/types.md"
log "products"
for pair in "unset=g0 c=g1" "unset=g0 ptx=g2" "c=g1 ptx=g2"; do
  set -- $pair
  b=${1#*=}; o=${2#*=}
  $N "$R125/run/products.py" --base "${1%%=*}=$RES/verify/$b" --other "${2%%=*}=$RES/verify/$o" \
    --requests d0p2,d8192p2 --tsv "$RES/products-$b-$o.tsv" > "$RES/products-$b-$o.md"
done
log "limits"
$N "$D/limits.py" "$RES/types.tsv" --request d0p2 --tsv "$RES/limits.tsv" > "$RES/limits.md"
$N "$D/limits.py" "$RES/types.tsv" --request d8192p2 > "$RES/limits_d8192.md"
log "steps"
$N "$D/steps.py" --control u0 --other p0 "$OUT/requests.csv" "$OUT2/requests.csv" --tsv "$RES/steps.tsv" > "$RES/steps.md"
log "deep lane tables"
$N "$S/bench/deep_lane/deep_table.py" "$OUT" --control u0 --out "$RES/table.md" --tsv "$RES/lane_classes.tsv" > /dev/null
$N "$S/bench/deep_lane/deep_table.py" "$OUT2" --control u2 --out "$RES/table2.md" --tsv "$RES/lane_classes2.tsv" > /dev/null
log "done"
touch "$RES/DONE"
