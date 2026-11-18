#!/usr/bin/env bash
# C854 step 4 (Q1525), Ornith's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_c854b_p106_check/run/ornith_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv   R166b's table rows (v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb) plus o4c2's:
#                                     its turns.csv rows, and only the attempts.csv rows whose tag starts with o4c2/
#   rig/server-{o4sna,o4snb,o4c2}.log the three runs' server logs
#   rig_o4c2/{meminfo.log,cards.csv}  o4c2's lines of the rig's meminfo.log (from 13:59:53) and its card log
#   rig_o4sna/, rig_o4snb/            R166a's and R166b's lines of meminfo.log and gpu.csv (R166's README)
# and writes its tables there.
set -u
T=${1:?}; D=docs/data/2026-11-18_c854b_p106_check; R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 \
  > "$T/agent_tables.out" 2>&1                                                       # agent.tsv
py scripts/realloc_counts.py --requests o4c2="$T/rig/server-o4c2.log" > "$T/o4c2_requests_reallocs.tsv"
py scripts/realloc_counts.py o4sna="$T/rig/server-o4sna.log" o4snb="$T/rig/server-o4snb.log" \
  o4c2="$T/rig/server-o4c2.log" > "$T/ornith_reallocs.tsv"
{ py $D/run/host_cards.py "$T/rig_o4c2" C854-ornith o4c2
  py $R166/run/host.py "$T/rig_o4sna" R166a o4sna | tail -n 1
  py $R166/run/host.py "$T/rig_o4snb" R166b o4snb | tail -n 1; } > "$T/ornith_host.tsv"
py $D/run/busy_clocks.py o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" \
  o4sna="$T/rig_o4sna/gpu.csv" "2026-11-16 14:00:33" "2026-11-16 14:29:45" \
  o4snb="$T/rig_o4snb/gpu.csv" "2026-11-16 14:52:05" "2026-11-16 15:29:03" > "$T/ornith_clocks.tsv"
# from the published per-request tables alone
py bench/agent_check/report.py work o4c2=$D/agent/o4c2/turns.csv o4sna=$R166/agent/o4sna/turns.csv \
  o4snb=$R166/agent/o4snb/turns.csv --tsv "$T/ornith_work.tsv" > "$T/ornith_work.md"
py $D/run/same_prompts.py o4c2=$D/agent/o4c2/turns.csv o4sna=$R166/agent/o4sna/turns.csv \
  o4snb=$R166/agent/o4snb/turns.csv > "$T/ornith_same_prompts.md"
py $D/run/same_prompts.py rig1-valid1=docs/data/2026-11-05_release_regression/agent/ornith-valid1/turns.csv \
  rig2-ka=docs/data/2026-11-06_ornith_mtp_rig2/agent/ornith-ka/turns.csv \
  rig2-o3a=docs/data/2026-11-14_r160_ornith_release_agent/agent/o3a/turns.csv \
  rig2-o3b=docs/data/2026-11-14_r160_ornith_release_agent/agent/o3b/turns.csv > "$T/ornith_same_prompts_v3.md"
