#!/usr/bin/env bash
# R209 (Q1603), Flash-Next's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-19_r209_oc_v5/run/flash_tables.sh TABLES ARM...
# ARM... are this row's Flash-Next arms: f5oc0 (stock), and f5oc1 (the full profile's offsets) once it has run. The
# harness's arm is v4r in both; this folder's agent/<arm>/ holds its turns.csv and attempts.csv with the arm renamed.
# Beside them, as context only: R203c's fnoc0 (the same rig at stock on v4.0's engine, 2026-11-18, R203's agent/fnoc0/)
# and C854's v4rc2 (this engine on the first rig, 2026-11-18, C854's agent/v4rc2/). R203's run/flash_tables.sh with
# those two context runs. TABLES (a folder of its own, not run/tables.sh's) holds
#   rig/server-r90v3.log, rig/server-<RUN>.log   R90's v3 (the equal-work reference) and each run's server log, RUN =
#                                                fnoc0, v4rc2 and each ARM (the draft's acceptance, reallocations)
#   rig_<RUN>/{meminfo.log,cards.csv,window}     each run's lines of the rig's meminfo.log from its server's start, its
#                                                card log (scripts/card_log.py), and "FROM|TO" of its tasks
# and writes flash_* tables there; res/ is rebuilt from the agent/ folders on each call.
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-19_r209_oc_v5; R203=docs/data/2026-11-18_r203_ornith_offsets
C854=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
CTX=(fnoc0 v4rc2); CTXDIR=("$R203/agent/fnoc0" "$C854/agent/v4rc2")
mkdir -p "$T/res"
# agent_tables.py's folder: R90's v3, the two context runs and each ARM in one res/turns.csv and res/attempts.csv
py - "$T/res" "$C854/agent/r90v3" "${CTXDIR[@]}" "${ARMS[@]/#/$D/agent/}" <<'EOF'
import csv, sys
out, srcs = sys.argv[1], sys.argv[2:]
for f in ("turns.csv", "attempts.csv"):
    rows = [r for s in srcs for r in csv.DictReader(open(f"{s}/{f}"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{out}/{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
EOF
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 "${CTX[@]}" "${ARMS[@]}" \
  > "$T/agent_tables.out" 2>&1                                                                            # agent.tsv
cp "$T/agent.tsv" "$T/flash_agent.tsv"
turns=(); clocks=(); logs=(); steps=(); : > "$T/flash_host.tsv"
i=0
for a in "${CTX[@]}" "${ARMS[@]}"; do
  if [ $i -lt ${#CTX[@]} ]; then tc=${CTXDIR[$i]}/turns.csv; else tc=$D/agent/$a/turns.csv; fi
  i=$((i + 1))
  turns+=("$a=$tc")
  steps+=("$a=$T/rig/server-$a.log=$tc")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
  logs+=("$a=$T/rig/server-$a.log")
  py $C854/run/host_cards.py "$T/rig_$a" "$a" "$a" | { [ -s "$T/flash_host.tsv" ] && tail -n 1 || cat; } \
    >> "$T/flash_host.tsv"
done
py $C854/run/same_prompts.py "${turns[@]}" > "$T/flash_same_prompts.md"
py bench/agent_check/report.py work "${turns[@]}" --tsv "$T/flash_work.tsv" > "$T/flash_work.md"
py $R203/run/flash_steps.py "${steps[@]}" > "$T/flash_steps.md"   # ms a decode step by context
py $C854/run/busy_clocks.py "${clocks[@]}" > "$T/flash_clocks.tsv"
py scripts/realloc_counts.py "${logs[@]}" > "$T/flash_reallocs.tsv"
