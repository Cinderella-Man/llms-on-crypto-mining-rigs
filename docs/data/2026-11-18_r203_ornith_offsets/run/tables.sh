#!/usr/bin/env bash
# R203 (Q1592), Ornith's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_r203_ornith_offsets/run/tables.sh TABLES ARM...
# ARM... are this row's arms: o4oc0 (stock), and o4oc1 (the use profile's offsets) once it has run. TABLES holds
#   res/turns.csv, res/attempts.csv   R166's rows as C854's tables used them (v3ka, o4sna, o4snb) plus each ARM's
#                                     turns.csv rows and its attempts.csv rows (file = the arm)
#   rig/server-<arm>.log              the server logs of o4sna, o4snb and each ARM
#   rig_<ARM>/{meminfo.log,cards.csv} each ARM's lines of the rig's meminfo.log (from its server's start) and card log
#   rig_<ARM>/window                  one line "FROM|TO" ("YYYY-MM-DD HH:MM:SS"): the arm's first request (the driver's
#                                     "client reaches" line) and its last task's end, in the rig's local time (the
#                                     driver's log; its clock and the rig's agreed to the second)
#   rig_o4sna/, rig_o4snb/            R166a's and R166b's lines of meminfo.log and gpu.csv (R166's README)
# and writes its tables there. The per-request tables of each ARM come from this folder's agent/<arm>/turns.csv.
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-18_r203_ornith_offsets; C854=docs/data/2026-11-18_c854b_p106_check
R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4sna o4snb "${ARMS[@]}" > "$T/agent_tables.out" 2>&1  # agent.tsv
logs=(o4sna="$T/rig/server-o4sna.log" o4snb="$T/rig/server-o4snb.log")
clocks=(o4sna="$T/rig_o4sna/gpu.csv" "2026-11-16 14:00:33" "2026-11-16 14:29:45"
        o4snb="$T/rig_o4snb/gpu.csv" "2026-11-16 14:52:05" "2026-11-16 15:29:03")
work=(o4sna=$R166/agent/o4sna/turns.csv o4snb=$R166/agent/o4snb/turns.csv)
: > "$T/host.tsv"
for a in "${ARMS[@]}"; do
  logs+=("$a=$T/rig/server-$a.log")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
  work+=("$a=$D/agent/$a/turns.csv")
  py $C854/run/host_cards.py "$T/rig_$a" "R203-$a" "$a" | { [ -s "$T/host.tsv" ] && tail -n 1 || cat; } >> "$T/host.tsv"
  py scripts/realloc_counts.py --requests "$a=$T/rig/server-$a.log" > "$T/${a}_requests_reallocs.tsv"
done
py $R166/run/host.py "$T/rig_o4sna" R166a o4sna | tail -n 1 >> "$T/host.tsv"
py $R166/run/host.py "$T/rig_o4snb" R166b o4snb | tail -n 1 >> "$T/host.tsv"
py scripts/realloc_counts.py "${logs[@]}" > "$T/reallocs.tsv"
py $C854/run/busy_clocks.py "${clocks[@]}" > "$T/clocks.tsv"
py bench/agent_check/report.py work "${work[@]}" --tsv "$T/work.tsv" > "$T/work.md"
py $C854/run/same_prompts.py "${work[@]}" > "$T/same_prompts.md"
