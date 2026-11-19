#!/usr/bin/env bash
# FN115's tables from a collected run: OUT (the lane's out/ folder: requests.csv, loads.tsv, requests-<load>.csv,
# st-<load>.jsonl, server-<load>.log), RES (the tables). Runs on any machine with Python 3, standard library only
# (here the operator's machine, on data already copied back from the rig).
#   OUT=<collected out> RES=<tables> bash docs/data/2026-11-19_fn115_flash_draft_head/run/tables.sh
set -euo pipefail
: "${OUT:?}" "${RES:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DT=$HERE/../../../../bench/deep_lane/deep_table.py
MAP=own0=own,own1=own,own2=own,dist1=dist,dist2=dist
mkdir -p "$RES/rel"
python3 -I "$HERE/relabel.py" "$OUT" "$RES/rel" "$MAP"
python3 -I "$DT" "$RES/rel" --control own --out "$RES/table.md" --tsv "$RES/classes.tsv" > /dev/null
python3 -I "$DT" "$RES/rel" --control own --class-on pass1 --out "$RES/table_pass1.md" --tsv "$RES/classes_pass1.tsv" > /dev/null
python3 -I "$DT" "$RES/rel" --control own --class-on pass2 --out "$RES/table_pass2.md" --tsv "$RES/classes_pass2.tsv" > /dev/null
python3 -I "$HERE/trace_table.py" "$OUT" --arm-map "$MAP" --out "$RES/trace.md" --tsv "$RES/trace.tsv" > /dev/null
# the draft's shortlist and cap reports, and the share line, per load (the server's own lines at each load's end)
for f in "$OUT"/server-*.log; do
  l=$(basename "$f" .log); l=${l#server-}
  grep -h -E 'short_report|cap_report|last_row_rep|reads the target|short_init|LLAMA_DRAFT_SHARE' "$f" | sed "s/^[0-9.]* [A-Z] /$l: /"
done > "$RES/draft_reports.txt" || true
echo "tables in $RES"
