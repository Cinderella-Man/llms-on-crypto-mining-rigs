#!/usr/bin/env bash
# R203 (Q1592), Flash-Next's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_tables.sh TABLES ARM...
# ARM... are this row's Flash-Next arms: fnoc0 (stock), and fnoc1 (the use profile's offsets) once it has run. The
# harness's arm is v4r in both; this folder's agent/<arm>/ holds its turns.csv and attempts.csv with the arm renamed.
# TABLES (a folder of its own, not run/tables.sh's) holds
#   rig/server-r90v3.log, rig/server-<ARM>.log  R90's v3 (the equal-work reference) and each ARM's server log (the
#                                               draft's acceptance)
#   rig_<ARM>/{meminfo.log,cards.csv,window}    as in run/tables.sh: each ARM's lines of the rig's meminfo.log from
#                                               its server's start, its card log, and "FROM|TO" of its tasks
#   rig_r156a/{meminfo.log,gpu.csv}             R156a's lines, cut to its run (C854's part (6) used them)
# and writes flash_* tables there; res/ is rebuilt from agent/ on each call. R156a's per-request table is C854's
# agent/r156a_v4r/turns.csv (rebuilt from its server log; cached counts within one token, so CACHE_TOL=1).
set -u
T=${1:?}; shift; ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || { echo "no arm"; exit 2; }
D=docs/data/2026-11-18_r203_ornith_offsets; C854=docs/data/2026-11-18_c854b_p106_check
R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
mkdir -p "$T/res"
# agent_tables.py's folder: R90's v3 and each ARM in one res/turns.csv and res/attempts.csv
py - "$T/res" "$C854/agent/r90v3" "${ARMS[@]/#/$D/agent/}" <<'EOF'
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
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 "${ARMS[@]}" > "$T/agent_tables.out" 2>&1  # agent.tsv
cp "$T/agent.tsv" "$T/flash_agent.tsv"
turns=(); clocks=(); : > "$T/flash_host.tsv"; logs=()
for a in "${ARMS[@]}"; do
  turns+=("$a=$D/agent/$a/turns.csv")
  IFS='|' read -r from to < "$T/rig_$a/window"
  clocks+=("$a=$T/rig_$a/cards.csv" "$from" "$to")
  logs+=("$a=$T/rig/server-$a.log")
  py $C854/run/host_cards.py "$T/rig_$a" "R203-$a" "$a" | { [ -s "$T/flash_host.tsv" ] && tail -n 1 || cat; } \
    >> "$T/flash_host.tsv"
done
py $R166/run/host.py "$T/rig_r156a" R156a v4r | tail -n 1 >> "$T/flash_host.tsv"
CACHE_TOL=1 py $C854/run/same_prompts.py "${turns[@]}" r156a-v4r=$C854/agent/r156a_v4r/turns.csv \
  > "$T/flash_same_prompts.md"
py bench/agent_check/report.py work "${turns[@]}" r156a-v4r=$C854/agent/r156a_v4r/turns.csv \
  --tsv "$T/flash_work.tsv" > "$T/flash_work.md"
py $C854/run/busy_clocks.py "${clocks[@]}" \
  r156a-v4r="$T/rig_r156a/gpu.csv" "2026-11-13 18:19:17" "2026-11-13 18:52:50" > "$T/flash_clocks.tsv"
py scripts/realloc_counts.py "${logs[@]}" > "$T/flash_reallocs.tsv"
