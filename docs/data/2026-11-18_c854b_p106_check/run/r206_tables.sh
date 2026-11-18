#!/usr/bin/env bash
# R206 (Q1597), the same-rig reference's tables (part (7)), standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_c854b_p106_check/run/r206_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv    ornith_tables.sh's rows (R166b's arms and o4c2) plus o4v5a's: its turns.csv
#                                      rows, and only the attempts.csv rows whose tag starts with o4v5a/
#   rig/server-{o4c2,o4v5a}.log        the two runs' server logs
#   rig_o4v5a/{meminfo.log,cards.csv}  o4v5a's lines of the rig's meminfo.log (from 15:55:02) and its card log
#   rig_o4c2/cards.csv                 o4c2's card log (ornith_tables.sh's)
# and writes its tables there.
set -u
T=${1:?}; D=docs/data/2026-11-18_c854b_p106_check; R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 o4v5a \
  > "$T/agent_tables.out" 2>&1                                                       # agent.tsv
py scripts/realloc_counts.py o4v5a="$T/rig/server-o4v5a.log" o4c2="$T/rig/server-o4c2.log" > "$T/r206_reallocs.tsv"
py $D/run/host_cards.py "$T/rig_o4v5a" R206 o4v5a > "$T/r206_host.tsv"
py $D/run/busy_clocks.py o4v5a="$T/rig_o4v5a/cards.csv" "2026-11-18 16:06:27" "2026-11-18 16:31:32" \
  o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" > "$T/r206_clocks.tsv"
# from the published per-request tables alone
py bench/agent_check/report.py work o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv \
  o4sna=$R166/agent/o4sna/turns.csv o4snb=$R166/agent/o4snb/turns.csv --tsv "$T/r206_work.tsv" > "$T/r206_work.md"
py $D/run/same_prompts.py o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv > "$T/r206_same_prompts.md"
py $D/run/same_prompts.py o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv \
  o4sna=$R166/agent/o4sna/turns.csv o4snb=$R166/agent/o4snb/turns.csv > "$T/r206_same_prompts4.md"
py $D/run/paired_prompts.py o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv > "$T/r206_paired.md"
py $D/run/paired_prompts.py o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv \
  o4sna=$R166/agent/o4sna/turns.csv o4snb=$R166/agent/o4snb/turns.csv > "$T/r206_paired4.md"
# the spread: R166's two runs back to back on one rig, on the same 21 requests
py $D/run/paired_prompts.py o4sna=$R166/agent/o4sna/turns.csv o4snb=$R166/agent/o4snb/turns.csv \
  o4v5a=$D/agent/o4v5a/turns.csv o4c2=$D/agent/o4c2/turns.csv > "$T/r206_paired_r166.md"
