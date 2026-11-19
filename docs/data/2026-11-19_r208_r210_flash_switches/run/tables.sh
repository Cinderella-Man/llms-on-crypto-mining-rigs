#!/usr/bin/env bash
# R208 and R210 (Q1602, Q1604): Flash-Next's tables for the runs done so far, standard library only (python3 -I), from
# the repository root:
#   bash docs/data/2026-11-19_r208_r210_flash_switches/run/tables.sh TABLES RUN...
# RUN names the runs in the order they ran (f5c1 f5s1 f5t1 f5c2 f5s2 f5t2); this folder's agent/<run>/ holds each
# run's turns.csv and attempts.csv (the client's res/ files with the harness's arm v4r renamed to the run). TABLES holds,
# per run:
#   rig/server-<run>.log                              the rig's llama-server log
#   rig_<run>/meminfo.log, rig_<run>/cards.csv        the rig's meminfo.log and card log (scripts/card_log.py's)
#   proxy_<run>/proxy.jsonl, proxy.jsonl.bodies.gz    the relay proxy's request rows and recorded bodies
#   window_<run>                                      the first task's start and the last task's end, the rig's local
#                                                     time "YYYY-MM-DD HH:MM:SS", one a line
# and, for the context rows, rig/server-r90v3.log and rig/server-v4rc2.log (R90's v3 and C854's run, for the draft's
# acceptance). The context run is C854's v4rc2 (docs/data/2026-11-18_c854b_p106_check/, part (6): the same rig, tree,
# preset and client, 2026-11-18); "a task at equal work" prices R90 v3's work, as there. Writes this folder's tables.
set -u
T=${1:?}; shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs named"; exit 1; }
D=docs/data/2026-11-19_r208_r210_flash_switches; C=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
mkdir -p "$T/res"
dirs=("$C/agent/r90v3" "$C/agent/v4rc2"); named=("v4rc2=$C/agent/v4rc2/turns.csv")
for r in "${RUNS[@]}"; do dirs+=("$D/agent/$r"); named+=("$r=$D/agent/$r/turns.csv"); done
# agent_tables.py's folder: R90's v3, C854's v4rc2 and the runs in one res/turns.csv and res/attempts.csv
py - "$T/res" "${dirs[@]}" <<'EOF'
import csv, sys
out, dirs = sys.argv[1], sys.argv[2:]
for f in ("turns.csv", "attempts.csv"):
    rows = [r for d in dirs for r in csv.DictReader(open(f"{d}/{f}"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{out}/{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
EOF
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 v4rc2 "${RUNS[@]}" > "$T/agent_tables.out" 2>&1
cp "$T/agent.tsv" "$D/agent.tsv"
# the same without JavaScript complex-numbers, the task this model has run to the 12-minute cap on the v4 line
NOCX=bugfix-javascript-complex-numbers
SKIP_TASKS=$NOCX OUT_TAG=_nocx py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 v4rc2 "${RUNS[@]}" \
  > "$T/agent_tables_nocx.out" 2>&1
cp "$T/agent_nocx.tsv" "$D/agent_nocx.tsv"
py - "$NOCX" "${named[@]}" > "$D/work_nocx.tsv" <<'EOF'
import csv, sys
skip = sys.argv[1]
print("run\ttasks\trequests\tprompt_read\tprompt_read_per_task\tgenerated\tgenerated_per_task")
for a in sys.argv[2:]:
    name, path = a.split("=", 1)
    rows = [r for r in csv.DictReader(open(path)) if r["task_id"] != skip and r["prompt_n"]]
    tasks = len({r["task_id"] for r in rows})
    pn = sum(int(r["prompt_n"]) for r in rows); gn = sum(int(r["predicted_n"]) for r in rows)
    print(f"{name}\t{tasks}\t{len(rows)}\t{pn}\t{pn / tasks:.0f}\t{gn}\t{gn / tasks:.0f}")
EOF
# the requests that read the same prompt on the same cached prefix in every run, and the work read and generated
py $C/run/same_prompts.py "${named[@]}" > "$D/same_prompts.md"
py bench/agent_check/report.py work "${named[@]}" --tsv "$D/work.tsv" > "$D/work.md"
# the longest first-token wait of each run, and the 90th percentile of every turn after a task's first
py - "${named[@]}" > "$D/first_token.tsv" <<'EOF'
import csv, sys
print("run\tturns\tlongest_s\ttask\tturn\tprompt_n\tcache_n\tprompt_ms\tlater_turns_median_s\tlater_turns_p90_s")
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    rows = [r for r in csv.DictReader(open(path)) if r["attempt"] == "1" and r["ttft_s"]]
    top = max(rows, key=lambda r: float(r["ttft_s"]))
    later = sorted(float(r["ttft_s"]) for r in rows if r["turn"] != "1")
    pick = lambda q: later[min(len(later) - 1, int(q * len(later)))]
    print(f"{name}\t{len(rows)}\t{float(top['ttft_s']):.2f}\t{top['task_id']}\t{top['turn']}\t{top['prompt_n']}\t"
          f"{top['cache_n']}\t{float(top['prompt_ms']):.0f}\t{pick(0.5):.2f}\t{pick(0.9):.2f}")
EOF
# reply re-reads: turns whose cached prefix stops inside the previous reply, their tokens and prompt seconds
rr=()
for r in "${RUNS[@]}"; do rr+=("$r:v4r:$T/proxy_$r/proxy.jsonl:$T/proxy_$r/proxy.jsonl.bodies.gz:$T/rig/server-$r.log"); done
py $C/run/reply_rereads.py "$D/rereads" "${rr[@]}" > "$T/rereads.out" 2>&1
# where text reuse kept the cache: each `same text` line of the server logs on its turn (none in runs without it)
py $D/run/text_reuse_points.py "$D/text_reuse_points.tsv" "${rr[@]}" > "$T/text_reuse.out" 2>&1
# busy SM clocks, host memory and swap, graph reallocations
cl=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cl+=("$r=$T/rig_$r/cards.csv" "${w[0]}" "${w[1]}"); done
py $C/run/busy_clocks.py "${cl[@]}" > "$D/clocks.tsv"
{ first=1; for r in "${RUNS[@]}"; do
    if [ $first = 1 ]; then py $C/run/host_cards.py "$T/rig_$r" "$r" "$r"; first=0
    else py $C/run/host_cards.py "$T/rig_$r" "$r" "$r" | tail -n 1; fi
  done; } > "$D/host.tsv"
ra=(); for r in "${RUNS[@]}"; do ra+=("$r=$T/rig/server-$r.log"); done
py scripts/realloc_counts.py "${ra[@]}" > "$D/reallocs.tsv"
# text reuse's own cost: the time from choosing the slot to processing each request, per run
sg=(); for r in "${RUNS[@]}"; do sg+=("$T/rig/server-$r.log"); done
py $D/run/slot_gap.py "${sg[@]}" > "$D/slot_gap.tsv"
