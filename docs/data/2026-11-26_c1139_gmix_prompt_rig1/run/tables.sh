#!/usr/bin/env bash
# C1139's tables from the deep lane's results folders (c1139.sh's OUT): the prompt reads of both files by part,
# product, weight type and card (prompt_parts.py over the group-profile, host-profile and plain loads), every prime in
# one table (summary.py), the replies side by side (replies.py), and the deep lane's own tables (deep_table.py).
# The first lane (OUT): loads pg gg ph gh pu1 gu1 at depths 0 8192 9216; the second (OUT2): gg2 pg2 gu2 pu2 at depths
# 0 8192 8704; the third (OUT3): gh3 ph3 (host profile) at depths 0 8192 8704.
# CPU only, standard-library Python 3.
# Env: S (this repository's root), OUT, OUT2 and OUT3 (the three lanes' results), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${OUT2:?}" "${OUT3:?}" "${RES:?}"
D=$S/docs/data/2026-11-26_c1139_gmix_prompt_rig1/run
KP=$S/docs/data/2026-11-04_kpass_profile/run/kpass_parts.py
N="nice -n 19 python3 -I"
mkdir -p "$RES"
rm -f "$RES/DONE"
z() { [ -e "$1.gz" ] && echo "$1.gz" || echo "$1"; }
# --gap-ms 3500: requests are at least 4 s apart (PAUSE=4); one host-profiled prime idles 3.1 s between its calls
$N "$D/prompt_parts.py" --kpass-parts "$KP" --min-ms 10 --gap-ms 3500 \
  --group "pmix=$(z "$OUT/gp-pg.csv"),$OUT/requests-pg.csv" --group "gmix=$(z "$OUT/gp-gg.csv"),$OUT/requests-gg.csv" \
  --host "pmix=$(z "$OUT/hp-ph.csv"),$OUT/requests-ph.csv" --host "gmix=$(z "$OUT/hp-gh.csv"),$OUT/requests-gh.csv" \
  --plain "pmix=$OUT/requests-pu1.csv" --plain "gmix=$OUT/requests-gu1.csv" \
  --base pmix --other gmix --tsv "$RES/prompt_parts.tsv" > "$RES/prompt_parts.md" || exit 1
$N "$D/prompt_parts.py" --kpass-parts "$KP" --min-ms 10 --gap-ms 3500 \
  --group "pmix=$(z "$OUT2/gp-pg.csv"),$OUT2/requests-pg2.csv" --group "gmix=$(z "$OUT2/gp-gg.csv"),$OUT2/requests-gg2.csv" \
  --host "pmix=$(z "$OUT3/hp-ph.csv"),$OUT3/requests-ph3.csv" --host "gmix=$(z "$OUT3/hp-gh.csv"),$OUT3/requests-gh3.csv" \
  --plain "pmix=$OUT2/requests-pu2.csv" --plain "gmix=$OUT2/requests-gu2.csv" \
  --base pmix --other gmix --tsv "$RES/prompt_parts2.tsv" > "$RES/prompt_parts2.md" || exit 1
$N "$D/summary.py" --base pmix --other gmix 1="$RES/prompt_parts.tsv" 2="$RES/prompt_parts2.tsv" > "$RES/summary.md"
$N "$D/replies.py" pg="$OUT/requests-pg.csv" ph="$OUT/requests-ph.csv" pu1="$OUT/requests-pu1.csv" \
  > "$RES/replies_pmix.md"
$N "$D/replies.py" gg="$OUT/requests-gg.csv" gh="$OUT/requests-gh.csv" gu1="$OUT/requests-gu1.csv" \
  > "$RES/replies_gmix.md"
$N "$D/replies.py" pg2="$OUT2/requests-pg2.csv" pu2="$OUT2/requests-pu2.csv" ph3="$OUT3/requests-ph3.csv" > "$RES/replies_pmix2.md"
$N "$D/replies.py" gg2="$OUT2/requests-gg2.csv" gu2="$OUT2/requests-gu2.csv" gh3="$OUT3/requests-gh3.csv" > "$RES/replies_gmix2.md"
$N "$S/bench/deep_lane/deep_table.py" "$OUT" --control pu --out "$RES/table.md" --tsv "$RES/lane_classes.tsv" > /dev/null
$N "$S/bench/deep_lane/deep_table.py" "$OUT2" --control pu --out "$RES/table2.md" --tsv "$RES/lane_classes2.tsv" > /dev/null
touch "$RES/DONE"
