#!/usr/bin/env bash
# R220 (Q1635): Flash-Next's tables for the runs done so far, standard library only (python3 -I), from the repository
# root:
#   bash docs/data/2026-11-20_r220_flash_ratio075/run/tables.sh TABLES RUN...
# RUN names the runs in the order they ran (c10a r075a c10b r075b); this folder's agent/<run>/ holds each run's
# turns.csv and attempts.csv (the client's res/ files with the harness's arm v4r renamed to the run by
# docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py). TABLES holds, per run:
#   rig/server-<run>.log                              the rig's llama-server log
#   rig_<run>/meminfo.log, rig_<run>/cards.csv        the rig's meminfo.log and card log (scripts/card_log.py's)
#   proxy_<run>/proxy.jsonl, proxy.jsonl.bodies.gz    the relay proxy's request rows and recorded bodies
#   window_<run>                                      the first task's start and the last task's end, the rig's local
#                                                     time "YYYY-MM-DD HH:MM:SS", one a line
# and, for the context rows, rig/server-r90v3.log (R90's v3: "a task at equal work" prices its work, as in C854's and
# R208's tables) and rig/server-f5oc1.log (R209d, docs/data/2026-11-19_r209_oc_v5/agent/f5oc1/: the same rig, tree and
# clock profile on 2026-11-19, another boot, v4.0's preset, that is without text reuse). This is R208/R210's
# run/tables.sh with those context rows, the decode steps by context (R203's flash_steps.py, and run/steps_ratio.py's
# buckets cut where the ratio acts), the prompt seconds of every task and each card's heat and power (R203's
# flash_cards.py). Writes this folder's tables.
set -u
T=${1:?}; shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs named"; exit 1; }
D=docs/data/2026-11-20_r220_flash_ratio075; C=docs/data/2026-11-18_c854b_p106_check
S=docs/data/2026-11-19_r208_r210_flash_switches; O=docs/data/2026-11-19_r209_oc_v5/agent/f5oc1
py() { nice -n 19 python3 -I "$@"; }
mkdir -p "$T/res"
dirs=("$C/agent/r90v3" "$O"); named=("f5oc1=$O/turns.csv"); steps=("f5oc1=$T/rig/server-f5oc1.log=$O/turns.csv")
for r in "${RUNS[@]}"; do
  dirs+=("$D/agent/$r"); named+=("$r=$D/agent/$r/turns.csv"); steps+=("$r=$T/rig/server-$r.log=$D/agent/$r/turns.csv")
done
# agent_tables.py's folder: R90's v3, R209d's f5oc1 and the runs in one res/turns.csv and res/attempts.csv
py - "$T/res" "${dirs[@]}" <<'PY'
import csv, sys
out, dirs = sys.argv[1], sys.argv[2:]
for f in ("turns.csv", "attempts.csv"):
    rows = [r for d in dirs for r in csv.DictReader(open(f"{d}/{f}"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{out}/{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
PY
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 f5oc1 "${RUNS[@]}" > "$T/agent_tables.out" 2>&1
cp "$T/agent.tsv" "$D/agent.tsv"
# the same without JavaScript complex-numbers, the task this model has run to the 12-minute cap on the v4 line
NOCX=bugfix-javascript-complex-numbers
SKIP_TASKS=$NOCX OUT_TAG=_nocx py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 f5oc1 "${RUNS[@]}" \
  > "$T/agent_tables_nocx.out" 2>&1
cp "$T/agent_nocx.tsv" "$D/agent_nocx.tsv"
# the requests that read the same prompt on the same cached prefix in every run, and the work read and generated
py $C/run/same_prompts.py "${named[@]}" > "$D/same_prompts.md"
py bench/agent_check/report.py work "${named[@]}" --tsv "$D/work.tsv" > "$D/work.md"
# prompt seconds of every task (its turns' prompt_ms summed), per run, and their mean a task
py - "${named[@]}" > "$D/prompt_tasks.tsv" <<'PY'
import csv, sys
runs = [a.split("=", 1) for a in sys.argv[1:]]
per = {}
for name, path in runs:
    for r in csv.DictReader(open(path)):
        if r["attempt"] == "1" and r.get("prompt_ms"):
            per.setdefault(r["task_id"], {}).setdefault(name, [0.0, 0, 0])
            v = per[r["task_id"]][name]
            v[0] += float(r["prompt_ms"]) / 1000; v[1] += int(r["prompt_n"]); v[2] += 1
names = [n for n, _ in runs]
print("task\t" + "\t".join(f"{n}_prompt_s\t{n}_prompt_tokens\t{n}_turns" for n in names))
tot = {n: [0.0, 0, 0, 0] for n in names}
for task in sorted(per):
    cells = []
    for n in names:
        v = per[task].get(n)
        if v:
            cells += [f"{v[0]:.2f}", str(v[1]), str(v[2])]
            t = tot[n]; t[0] += v[0]; t[1] += v[1]; t[2] += v[2]; t[3] += 1
        else:
            cells += ["", "", ""]
    print(task + "\t" + "\t".join(cells))
print("mean a task\t" + "\t".join(f"{t[0] / t[3]:.2f}\t{t[1] / t[3]:.0f}\t{t[2] / t[3]:.2f}" for t in tot.values()))
PY
# the longest first-token wait of each run, and the 90th percentile of every turn after a task's first
py - "${named[@]}" > "$D/first_token.tsv" <<'PY'
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
PY
# ms a decode step by context: R203's buckets, and the buckets cut where the ratio moves a four-token verify pass
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py "${steps[@]}" > "$D/steps.md"
py $D/run/steps_ratio.py "${steps[@]}" > "$D/steps_ratio.md"
# reply re-reads: turns whose cached prefix stops inside the previous reply, their tokens and prompt seconds
rr=()
for r in "${RUNS[@]}"; do rr+=("$r:v4r:$T/proxy_$r/proxy.jsonl:$T/proxy_$r/proxy.jsonl.bodies.gz:$T/rig/server-$r.log"); done
py $C/run/reply_rereads.py "$D/rereads" "${rr[@]}" > "$T/rereads.out" 2>&1
# where text reuse kept the cache: each `same text` line of the server logs on its turn
py $S/run/text_reuse_points.py "$D/text_reuse_points.tsv" "${rr[@]}" > "$T/text_reuse.out" 2>&1
# busy SM clocks, host memory and swap, graph reallocations
cl=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cl+=("$r=$T/rig_$r/cards.csv" "${w[0]}" "${w[1]}"); done
py $C/run/busy_clocks.py "${cl[@]}" > "$D/clocks.tsv"
# each card's highest temperature and power, samples above 80 W and busy memory clock over the tasks (R203's script)
cc=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cc+=("$r=$T/rig_$r/cards.csv=${w[0]}=${w[1]}"); done
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards "${cc[@]}" > "$D/cards.md"
{ first=1; for r in "${RUNS[@]}"; do
    if [ $first = 1 ]; then py $C/run/host_cards.py "$T/rig_$r" "$r" "$r"; first=0
    else py $C/run/host_cards.py "$T/rig_$r" "$r" "$r" | tail -n 1; fi
  done; } > "$D/host.tsv"
ra=(); for r in "${RUNS[@]}"; do ra+=("$r=$T/rig/server-$r.log"); done
py scripts/realloc_counts.py "${ra[@]}" > "$D/reallocs.tsv"
# text reuse's own cost: the time from choosing the slot to processing each request, per run
sg=(); for r in "${RUNS[@]}"; do sg+=("$T/rig/server-$r.log"); done
py $S/run/slot_gap.py "${sg[@]}" > "$D/slot_gap.tsv"
