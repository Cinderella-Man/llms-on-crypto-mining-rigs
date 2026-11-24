#!/usr/bin/env bash
# R232 (b) (Q1706), the tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/tables.sh TABLES ARM...
# ARM... are the runs of this folder so far, in the order they ran: o5f16a, o5q8a, o5f16b, o5q8b (o5f16 the v5 preset
# with its F16 cache, o5q8 the same with a Q8_0 cache and GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4).
# Beside them, as context only: R17's two runs of the same preset (o5t1a, o5t1b: the same rig and kind of clock
# profile, an older v5 engine, another boot; docs/data/2026-11-19_r17_ornith_textreuse/). TABLES holds
#   res/turns.csv, res/attempts.csv   the rows of v3ka (the equal-work reference), o5t1a, o5t1b and each ARM
#   rig/server-<arm>.log              each ARM's server log
#   rig_<arm>/{meminfo.log,cards.csv} each ARM's lines of the rig's meminfo.log (from its server's start) and card log
#   rig_<arm>/window                  one line "FROM|TO" ("YYYY-MM-DD HH:MM:SS"): the arm's first request (the driver's
#                                     "client reaches" line) and its last task's end, in the rig's local time
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-24_r232b_ornith_q8_agent; R17=docs/data/2026-11-19_r17_ornith_textreuse
C854=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
CTX=(o5t1a o5t1b)
py bench/agent_check/agent_tables.py "$T" v3ka "${CTX[@]}" "${ARMS[@]}" > "$T/agent_tables.out" 2>&1   # agent.tsv
turns=(); logs=(); clocks=(); cards=()
for a in "${CTX[@]}"; do turns+=("$a=$R17/agent/$a/turns.csv"); done
for a in "${ARMS[@]}"; do
  turns+=("$a=$D/agent/$a/turns.csv")
  logs+=("$a=$T/rig/server-$a.log")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
  cards+=("$a=$T/rig_$a/cards.csv=$from=$to")
  py scripts/realloc_counts.py --requests "$a=$T/rig/server-$a.log" > "$T/${a}_requests_reallocs.tsv"
done
: > "$T/host.tsv"
for a in "${ARMS[@]}"; do
  py $C854/run/host_cards.py "$T/rig_$a" "R232b-$a" "$a" | { [ -s "$T/host.tsv" ] && tail -n 1 || cat; } >> "$T/host.tsv"
done
py scripts/realloc_counts.py "${logs[@]}" > "$T/reallocs.tsv"
py $C854/run/busy_clocks.py "${clocks[@]}" > "$T/clocks.tsv"
# the work (C852): prompt tokens read, reply tokens read again and their prompt seconds, the longest first-token wait
py bench/agent_check/report.py work "${turns[@]}" --tsv "$T/work.tsv" > "$T/work.md"
# the requests read on the same prompt in every run here
py $C854/run/same_prompts.py "${turns[@]}" > "$T/same_prompts.md"
# generation by the context it ran at, and at the runs' shared mix
py $C854/run/decode_by_context.py "${turns[@]}" > "$T/decode.md"
# each card's highest power, samples above 80 W and busy memory clock over each run's tasks
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards "${cards[@]}" > "$T/cards.md"
