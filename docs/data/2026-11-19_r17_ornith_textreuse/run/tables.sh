#!/usr/bin/env bash
# R17's Ornith arm (Q260), the tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-19_r17_ornith_textreuse/run/tables.sh TABLES ARM...
# ARM... are the runs of this folder so far, in the order they ran: o5t0a, o5t1a, o5t0b, o5t1b (o5t0 the v5 preset,
# o5t1 the same plus LLAMA_SERVER_TEXT_REUSE=1). Beside them, as context: R209e's o5oc1 (the same rig, profile, engine
# and preset, on the morning's boot; docs/data/2026-11-19_r209_oc_v5/). TABLES holds
#   res/turns.csv, res/attempts.csv   the rows of v3ka (the equal-work reference), o5oc1 and each ARM
#   rig/server-<arm>.log              the server logs of o5oc1 and each ARM
#   rig_<arm>/{meminfo.log,cards.csv} each run's lines of the rig's meminfo.log (from its server's start) and card log
#   rig_<arm>/window                  one line "FROM|TO" ("YYYY-MM-DD HH:MM:SS"): the arm's first request (the driver's
#                                     "client reaches" line) and its last task's end, in the rig's local time
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-19_r17_ornith_textreuse; R209=docs/data/2026-11-19_r209_oc_v5
C854=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
ALL=(o5oc1 "${ARMS[@]}")
py bench/agent_check/agent_tables.py "$T" v3ka "${ALL[@]}" > "$T/agent_tables.out" 2>&1   # agent.tsv, with the work
turns=(o5oc1=$R209/agent/o5oc1/turns.csv); logs=(); clocks=()
for a in "${ALL[@]}"; do
  [ "$a" = o5oc1 ] || turns+=("$a=$D/agent/$a/turns.csv")
  logs+=("$a=$T/rig/server-$a.log")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
done
for a in "${ARMS[@]}"; do
  py scripts/realloc_counts.py --requests "$a=$T/rig/server-$a.log" > "$T/${a}_requests_reallocs.tsv"
done
: > "$T/host.tsv"
for a in "${ALL[@]}"; do
  py $C854/run/host_cards.py "$T/rig_$a" "R17-$a" "$a" | { [ -s "$T/host.tsv" ] && tail -n 1 || cat; } >> "$T/host.tsv"
done
py scripts/realloc_counts.py "${logs[@]}" > "$T/reallocs.tsv"
py $C854/run/busy_clocks.py "${clocks[@]}" > "$T/clocks.tsv"
# the work (C852): prompt tokens read, reply tokens read again and their prompt seconds, the longest first-token wait
py bench/agent_check/report.py work "${turns[@]}" --tsv "$T/work.tsv" > "$T/work.md"
# the requests read on the same prompt in every run here
py $C854/run/same_prompts.py "${turns[@]}" > "$T/same_prompts.md"
# generation by the context it ran at, and at the runs' shared mix
py $C854/run/decode_by_context.py "${turns[@]}" > "$T/decode.md"
# each card's highest temperature and power, samples above 80 W and busy memory clock over each run's tasks
cards=()
for a in "${ALL[@]}"; do IFS='|' read -r from to < "$T/rig_$a/window"; cards+=("$a=$T/rig_$a/cards.csv=$from=$to"); done
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards "${cards[@]}" > "$T/cards.md"
