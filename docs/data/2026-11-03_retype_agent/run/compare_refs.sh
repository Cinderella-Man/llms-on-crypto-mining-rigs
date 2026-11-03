#!/usr/bin/env bash
# compare_refs.sh CMP ARM=DIR [ARM=DIR ...]: one agent-check report over arms that ran in separate checks. Each DIR is
# a check's results folder (release_check.sh's CLIENT_OUT, on the client); ARM=DIR takes that arm's scored attempts
# (DIR/res/ARM) and its proxy rows (the lines of DIR/proxy.jsonl tagged ARM/...) into CMP, then writes CMP/compare.md
# (report.py compare, the first arm the reference; per request rows in CMP/res/turns.csv) and CMP/table.md. Runs on
# the client, where the folders are, with Python 3; BENCH is the agent_check folder there (release_check.sh copies it
# to CLIENT_DIR/agent_check).
set -euo pipefail
CMP=${1:?usage: compare_refs.sh CMP ARM=DIR ...}
shift
R=${BENCH:?BENCH=the agent_check folder on the client}/report.py
rm -rf "$CMP"
mkdir -p "$CMP/res"
arms=()
for a in "$@"; do
  arm=${a%%=*} dir=${a#*=}
  cp -r "$dir/res/$arm" "$CMP/res/$arm"
  grep -F "\"tag\": \"$arm/" "$dir/proxy.jsonl" >> "$CMP/proxy.jsonl"
  arms+=("$arm")
done
list=$(IFS=,; echo "${arms[*]}")
python3 -I "$R" compare "$CMP/res" --log "$CMP/proxy.jsonl" --arms "$list" > "$CMP/compare.md"
python3 -I "$R" table "$CMP/res" --log "$CMP/proxy.jsonl" --csv "$CMP/res/attempts.csv" > "$CMP/table.md"
cat "$CMP/compare.md"
