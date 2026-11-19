#!/usr/bin/env bash
# R209 (Q1603), Ornith's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-19_r209_oc_v5/run/tables.sh TABLES ARM...
# ARM... are this row's Ornith arms: o5oc0 (stock), and o5oc1 (the full profile's offsets) once it has run. Beside them,
# as context only: R203a's o4oc0 (the same rig at stock on v4.0's engine and preset, 2026-11-18) and R204's o4g2a and
# o4g2b (this engine and preset on the first rig, 2026-11-18). TABLES holds
#   res/turns.csv, res/attempts.csv   the rows of v3ka (the equal-work reference), o4oc0, o4g2a, o4g2b and each ARM
#                                     (attempts.csv: the rows whose file is the arm, bytes kept)
#   rig/server-<arm>.log              the server logs of o4oc0, o4g2a, o4g2b and each ARM
#   rig_<arm>/{meminfo.log,cards.csv} each run's lines of the rig's meminfo.log (from its server's start) and card log
#   rig_<ARM>/window                  one line "FROM|TO" ("YYYY-MM-DD HH:MM:SS"): the arm's first request (the driver's
#                                     "client reaches" line) and its last task's end, in the rig's local time
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-19_r209_oc_v5; C854=docs/data/2026-11-18_c854b_p106_check
R203=docs/data/2026-11-18_r203_ornith_offsets; R204=docs/data/2026-11-18_r204_ornith_glu_agent
py() { nice -n 19 python3 -I "$@"; }
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4oc0 o4g2a o4g2b "${ARMS[@]}" \
  > "$T/agent_tables.out" 2>&1                                                        # agent.tsv
turns=(o4oc0=$R203/agent/o4oc0/turns.csv o4g2a=$R204/agent/o4g2a/turns.csv o4g2b=$R204/agent/o4g2b/turns.csv)
logs=(o4oc0="$T/rig/server-o4oc0.log" o4g2a="$T/rig/server-o4g2a.log" o4g2b="$T/rig/server-o4g2b.log")
clocks=(o4oc0="$T/rig_o4oc0/cards.csv" "2026-11-18 15:19:19" "2026-11-18 15:51:25"
        o4g2a="$T/rig_o4g2a/cards.csv" "2026-11-18 18:51:07" "2026-11-18 19:13:52"
        o4g2b="$T/rig_o4g2b/cards.csv" "2026-11-18 19:45:09" "2026-11-18 20:11:57")
mine=()
for a in "${ARMS[@]}"; do
  turns+=("$a=$D/agent/$a/turns.csv"); mine+=("$a=$D/agent/$a/turns.csv")
  logs+=("$a=$T/rig/server-$a.log")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
  py scripts/realloc_counts.py --requests "$a=$T/rig/server-$a.log" > "$T/${a}_requests_reallocs.tsv"
done
: > "$T/host.tsv"
for a in "${ARMS[@]}" o4oc0 o4g2a o4g2b; do
  py $C854/run/host_cards.py "$T/rig_$a" "R209-$a" "$a" | { [ -s "$T/host.tsv" ] && tail -n 1 || cat; } >> "$T/host.tsv"
done
py scripts/realloc_counts.py "${logs[@]}" > "$T/reallocs.tsv"
py $C854/run/busy_clocks.py "${clocks[@]}" > "$T/clocks.tsv"
py bench/agent_check/report.py work "${turns[@]}" --tsv "$T/work.tsv" > "$T/work.md"
# the requests read on the same prompt in every run here, exactly and with the cached tokens within one
py $C854/run/same_prompts.py "${turns[@]}" > "$T/same_prompts.md"
CACHE_TOL=1 py $C854/run/same_prompts.py "${turns[@]}" > "$T/same_prompts_tol1.md"
# generation by context: every run here at its own shared mix, and at R204's mix (its four runs on the first rig,
# r204b_decode.md), so this row's numbers sit beside R204's without moving them
py $C854/run/decode_by_context.py "${turns[@]}" > "$T/decode.md"
py $D/run/decode_at_mix.py o4c2=$C854/agent/o4c2/turns.csv o4c3=$C854/agent/o4c3/turns.csv \
  o4g2a=$R204/agent/o4g2a/turns.csv o4g2b=$R204/agent/o4g2b/turns.csv -- o4oc0=$R203/agent/o4oc0/turns.csv "${mine[@]}" \
  > "$T/decode_r204_mix.md"
