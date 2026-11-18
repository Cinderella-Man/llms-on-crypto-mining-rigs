#!/usr/bin/env bash
# C854 step 4 (Q1525), Flash-Next's tables, standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_c854b_p106_check/run/flash_tables.sh TABLES
# TABLES is a folder holding
#   rig/server-r90v3.log, rig/server-v4rc2.log  R90's v3 and this run's server logs (the draft's acceptance)
#   rig_v4rc2/{meminfo.log,cards.csv}           this run's lines of the rig's meminfo.log and its card log
#   rig_r156a/, rig_r156b/                      R156a's and R156b's meminfo.log and gpu.csv, cut to their runs
# and writes its tables there. The per-request tables come from this folder's agent/: v4rc2 (this run, the harness's
# arm v4r renamed), r90v3 (R90's v3, the reference of R156's equal work) and R156a's and R156b's, rebuilt from their
# server logs with run/server_turns.py (README, "Commands").
set -u
T=${1:?}; D=docs/data/2026-11-18_c854b_p106_check; R156=docs/data/2026-11-13_r156_agent_v4r
py() { nice -n 19 python3 -I "$@"; }
NOCX=bugfix-javascript-complex-numbers
mkdir -p "$T/res"
# agent_tables.py's folder: R90's v3 and v4rc2 in one res/turns.csv and res/attempts.csv
py - "$D/agent" "$T/res" <<'EOF'
import csv, sys
src, out = sys.argv[1:3]
for f in ("turns.csv", "attempts.csv"):
    rows = [r for arm in ("r90v3", "v4rc2") for r in csv.DictReader(open(f"{src}/{arm}/{f}"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{out}/{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
EOF
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 v4rc2 > "$T/agent_tables.out" 2>&1        # agent.tsv
SKIP_TASKS=$NOCX OUT_TAG=_nocx py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 v4rc2 \
  > "$T/agent_tables_nocx.out" 2>&1                                                                       # agent_nocx.tsv
# from the published per-request tables alone; R156's cache counts are within one token (server_turns.py)
CACHE_TOL=1 py $D/run/same_prompts.py v4rc2=$D/agent/v4rc2/turns.csv r156a-v4r=$D/agent/r156a_v4r/turns.csv \
  r156b-v4r2=$D/agent/r156b_v4r2/turns.csv > "$T/flash_same_prompts.md"
py bench/agent_check/report.py work v4rc2=$D/agent/v4rc2/turns.csv r156a-v4r=$D/agent/r156a_v4r/turns.csv \
  r156b-v4r2=$D/agent/r156b_v4r2/turns.csv --tsv "$T/flash_work.tsv" > "$T/flash_work.md"
py - $D/agent "$T/flash_work_nocx.tsv" $NOCX <<'EOF'
import csv, sys
src, out, skip = sys.argv[1:4]
with open(out, "w") as fh:
    fh.write("run\ttasks\trequests\tprompt_read\tprompt_read_per_task\tgenerated\tgenerated_per_task\n")
    for run in ("v4rc2", "r156a_v4r", "r156b_v4r2"):
        rows = [r for r in csv.DictReader(open(f"{src}/{run}/turns.csv")) if r["task_id"] != skip]
        tasks = len({r["task_id"] for r in rows})
        pn = sum(int(r["prompt_n"]) for r in rows); gn = sum(int(r["predicted_n"]) for r in rows)
        fh.write(f"{run}\t{tasks}\t{len(rows)}\t{pn}\t{pn / tasks:.0f}\t{gn}\t{gn / tasks:.0f}\n")
EOF
py $D/run/busy_clocks.py v4rc2="$T/rig_v4rc2/cards.csv" "2026-11-18 15:00:11" "2026-11-18 15:32:58" \
  r156b-v4r2="$T/rig_r156b/gpu.csv" "2026-11-13 19:02:57" "2026-11-13 19:34:02" \
  r156a-v4r="$T/rig_r156a/gpu.csv" "2026-11-13 18:19:17" "2026-11-13 18:52:50" > "$T/flash_clocks.tsv"
{ py $D/run/host_cards.py "$T/rig_v4rc2" C854-flash v4rc2
  py docs/data/2026-11-16_r166_ornith_stack_no_taper/run/host.py "$T/rig_r156b" R156b v4r2 | tail -n 1
  py docs/data/2026-11-16_r166_ornith_stack_no_taper/run/host.py "$T/rig_r156a" R156a v4r | tail -n 1; } \
  > "$T/flash_host.tsv"
py scripts/realloc_counts.py v4rc2="$T/rig/server-v4rc2.log" > "$T/flash_reallocs.tsv"
