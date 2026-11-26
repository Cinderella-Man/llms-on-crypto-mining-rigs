#!/usr/bin/env bash
# R248 (Q1938): the agent-check tables of one run of Flash-Next's v5 preset as it stands (arm pre of
# bench/agent_check/preset_agent.sh), standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-26_r248_preset_smoke/run/tables_agent.sh TABLES pre
# R243's run/tables_agent.sh (docs/data/2026-11-26_r243_flash_131k_agent_rig1/) with this folder and the harness's arm
# named pre (preset_agent.sh names the harness's arm after the run, so no renaming); with one run the class sections
# (four runs) are skipped. TABLES holds the same inputs as R243's (its header lists them): rig/server-<run>.log,
# rig_<run>/meminfo.log and cards.csv, proxy_<run>/proxy.jsonl and .bodies.gz, window_<run>, rig/server-r90v3.log.
set -u
T=${1:?}; shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs named"; exit 1; }
D=docs/data/2026-11-26_r248_preset_smoke; N=docs/data/2026-11-26_c1175_nsm10_rig2; C=docs/data/2026-11-18_c854b_p106_check
S=docs/data/2026-11-19_r208_r210_flash_switches; Q=docs/data/2026-11-20_r227_flash_taper1400_agent/agent
REFS=()
py() { nice -n 19 python3 -I "$@"; }
mkdir -p "$T/res"
dirs=("$C/agent/r90v3"); named=(); steps=()
for r in "${REFS[@]}"; do
  dirs+=("$Q/$r"); named+=("$r=$Q/$r/turns.csv"); steps+=("$r=$T/rig/server-$r.log=$Q/$r/turns.csv")
done
for r in "${RUNS[@]}"; do
  dirs+=("$D/agent/$r"); named+=("$r=$D/agent/$r/turns.csv"); steps+=("$r=$T/rig/server-$r.log=$D/agent/$r/turns.csv")
done
# agent_tables.py's folder: R90's v3 and the runs in one res/turns.csv and res/attempts.csv
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
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 "${REFS[@]}" "${RUNS[@]}" > "$T/agent_tables.out" 2>&1
cp "$T/agent.tsv" "$D/agent.tsv"
# the tasks that ran to a cap (the 12-minute cap, runaway_kind time, or the turn cap, runaway_kind turns) in any named
# run, and the table without them
py - "${RUNS[@]/#/$D/agent/}" > "$D/capped.tsv" <<'PY'
import csv, sys
print("run\ttask\tcap\twall_s\tturns\tpassed")
for d in sys.argv[1:]:
    for a in csv.DictReader(open(f"{d}/attempts.csv")):
        if a["runaway_kind"] in ("time", "turns"):
            print(f"{a['file']}\t{a['task_id']}\t{a['runaway_kind']}\t{a['wall_s']}\t{a['turns']}\t{a['pass']}")
PY
CAPPED=$(tail -n +2 "$D/capped.tsv" | cut -f 2 | sort -u | paste -sd,)
if [ -n "$CAPPED" ]; then
  SKIP_TASKS=$CAPPED OUT_TAG=_nocap py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" r90v3 "${REFS[@]}" "${RUNS[@]}" \
    > "$T/agent_tables_nocap.out" 2>&1
  cp "$T/agent_nocap.tsv" "$D/agent_nocap.tsv"
else
  rm -f "$D/agent_nocap.tsv"
fi
# the requests that read the same prompt on the same cached prefix in every run, and the work read and generated
py $C/run/same_prompts.py "${named[@]}" > "$D/same_prompts.md"
# the same requests summed by size (the empty-cache first request, under 600, 600 to 1,400 and over 1,400 new tokens),
# classed (same_sizes.tsv)
[ ${#RUNS[@]} -eq 4 ] && py $D/run/same_sizes.py "$D/same_prompts.md" "${RUNS[0]}" "${RUNS[1]}" "${RUNS[2]}" "${RUNS[3]}" > "$D/same_sizes.tsv"
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
# prompt seconds by the turn's new tokens, cut where the micro-batch changes (up to 1,400 the taper's 64 in both arms;
# above it 512 or 256): turns, tokens, seconds, ms a token; then all calls over 1,400 tokens, and those of them that
# read on a cached prefix (every call but each run's first, which reads its 2,780 tokens on an empty cache); all tasks
# and without the capped ones
CAPPED=$CAPPED py - "${named[@]}" > "$D/prompt_buckets.tsv" <<'PY'
import csv, os, sys
skip = set(filter(None, os.environ.get("CAPPED", "").split(",")))
edges = [(0, 599, "under 600"), (600, 1400, "600-1,400"), (1401, 1999, "1,401-1,999"), (2000, 10**9, "2,000+")]
print("run\ttasks\tbucket\tturns\tnew_tokens\tprompt_s\tms_a_token")
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    rows = [r for r in csv.DictReader(open(path)) if r["attempt"] == "1" and r.get("prompt_ms")]
    for label, keep in (("all", rows), ("without capped", [r for r in rows if r["task_id"] not in skip])):
        if label != "all" and not skip:
            continue
        tot = [0, 0, 0.0]
        for lo, hi, b in edges:
            sel = [r for r in keep if lo <= int(r["prompt_n"]) <= hi]
            n, tok, s = len(sel), sum(int(r["prompt_n"]) for r in sel), sum(float(r["prompt_ms"]) for r in sel) / 1000
            tot[0] += n; tot[1] += tok; tot[2] += s
            print(f"{name}\t{label}\t{b}\t{n}\t{tok}\t{s:.2f}\t{(s * 1000 / tok if tok else 0):.3f}")
        sel = [r for r in keep if int(r["prompt_n"]) > 1400]
        n, tok, sec = len(sel), sum(int(r["prompt_n"]) for r in sel), sum(float(r["prompt_ms"]) for r in sel) / 1000
        print(f"{name}\t{label}\tover 1,400\t{n}\t{tok}\t{sec:.2f}\t{(sec * 1000 / tok if tok else 0):.3f}")
        sel = [r for r in sel if int(r["cache_n"]) > 0]
        n, tok, sec = len(sel), sum(int(r["prompt_n"]) for r in sel), sum(float(r["prompt_ms"]) for r in sel) / 1000
        print(f"{name}\t{label}\tover 1,400 on a cached prefix\t{n}\t{tok}\t{sec:.2f}\t{(sec * 1000 / tok if tok else 0):.3f}")
        print(f"{name}\t{label}\tall turns\t{tot[0]}\t{tot[1]}\t{tot[2]:.2f}\t{(tot[2] * 1000 / tot[1] if tot[1] else 0):.3f}")
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
# the taper each server read at its load
for r in "${REFS[@]}" "${RUNS[@]}"; do
  printf '%s\t%s\n' "$r" "$(grep -m1 -o 'micro-batch taper of prompt calls: .*' "$T/rig/server-$r.log")"
done > "$D/taper_lines.tsv"
# ms a decode step by context (R203's buckets)
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py "${steps[@]}" > "$D/steps.md"
# the same in C1175's buckets: under 4,102 cells (a four-token verify on the dense kernels), 4,102 to 6,153, 6,153 to
# 8,204 and 8,204 to 16,000 (on the per-token sparse kernel, in both arms here)
py $N/run/steps_cells.py "${steps[@]}" > "$D/steps_cells.md"
# reply re-reads: turns whose cached prefix stops inside the previous reply, their tokens and prompt seconds
rr=()
for r in "${RUNS[@]}"; do rr+=("$r:$r:$T/proxy_$r/proxy.jsonl:$T/proxy_$r/proxy.jsonl.bodies.gz:$T/rig/server-$r.log"); done
py $C/run/reply_rereads.py "$D/rereads" "${rr[@]}" > "$T/rereads.out" 2>&1
# where text reuse kept the cache: each `same text` line of the server logs on its turn
py $S/run/text_reuse_points.py "$D/text_reuse_points.tsv" "${rr[@]}" > "$T/text_reuse.out" 2>&1
# busy SM clocks, host memory and swap, graph reallocations
cl=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cl+=("$r=$T/rig_$r/cards.csv" "${w[0]}" "${w[1]}"); done
py $C/run/busy_clocks.py "${cl[@]}" > "$D/clocks.tsv"
# each card's highest temperature and power over the tasks (R203's script), kept beside the inputs, not published
cc=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cc+=("$r=$T/rig_$r/cards.csv=${w[0]}=${w[1]}"); done
py docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards "${cc[@]}" > "$T/cards.md"
{ first=1; for r in "${RUNS[@]}"; do
    if [ $first = 1 ]; then py $C/run/host_cards.py "$T/rig_$r" "$r" "$r"; first=0
    else py $C/run/host_cards.py "$T/rig_$r" "$r" "$r" | tail -n 1; fi
  done; } > "$D/host.tsv"
ra=(); for r in "${RUNS[@]}"; do ra+=("$r=$T/rig/server-$r.log"); done
py scripts/realloc_counts.py "${ra[@]}" > "$D/reallocs.tsv"
# text reuse's own cost: the time from choosing the slot to processing each request, per run
sg=(); for r in "${RUNS[@]}"; do sg+=("$T/rig/server-$r.log"); done
py $S/run/slot_gap.py "${sg[@]}" > "$D/slot_gap.tsv"
# the tokens each server sampled while the tool-call grammar applied (C1080, Q1813): the server log's sampling lines
gr=(); for r in "${REFS[@]}" "${RUNS[@]}"; do gr+=("$r=$T/rig/server-$r.log"); done
py - "${gr[@]}" > "$D/grammar.tsv" <<'PY'
import re, sys
print("run\trequests\ttokens_sampled\twhile_grammar_applied\trequests_without_grammar\trejected")
pat = re.compile(r"sampling =\s+(\d+) tokens, \d+ drawn on the device, (\d+) while the grammar applied, (\d+) rejected")
for arg in sys.argv[1:]:
    name, log = arg.split("=", 1)
    n = tok = gram = none = rej = 0
    for ln in open(log, errors="replace"):
        m = pat.search(ln)
        if m:
            a, g, j = map(int, m.groups())
            n += 1; tok += a; gram += g; none += g == 0; rej += j
    print(f"{name}\t{n}\t{tok}\t{gram}\t{none}\t{rej}")
PY
# the class: the arm against the preset, each arm's two runs, by the release check's rule (a gain or a loss only beyond
# the larger of the two arms' spreads)
if [ ${#RUNS[@]} -eq 4 ]; then
  ARMN=${RUNS[1]%[ab]}
  py bench/agent_check/class_arms.py "$D/agent.tsv" "$ARMN=${RUNS[1]},${RUNS[3]}" "c=${RUNS[0]},${RUNS[2]}" > "$D/class.tsv"
  if [ -n "$CAPPED" ]; then
    py bench/agent_check/class_arms.py "$D/agent_nocap.tsv" "$ARMN=${RUNS[1]},${RUNS[3]}" "c=${RUNS[0]},${RUNS[2]}" \
      > "$D/class_nocap.tsv"
  else
    rm -f "$D/class_nocap.tsv"
  fi
fi
# the decode step per cell bucket and the first-token wait of first and later turns, classed by the same rule
# (class_cells.tsv)
if [ ${#RUNS[@]} -eq 4 ]; then
  py $N/run/class_cells.py "$D/steps_cells.md" "$D/first_token.tsv" "$D/agent" "${RUNS[0]}" "${RUNS[1]}" "${RUNS[2]}" \
    "${RUNS[3]}" > "$D/class_cells.tsv"
fi
# prompt ms a token per bucket of new tokens, classed by the same rule (class_prompt.tsv)
if [ ${#RUNS[@]} -eq 4 ]; then
  py $D/run/class_prompt.py "$D/prompt_buckets.tsv" "${RUNS[0]}" "${RUNS[1]}" "${RUNS[2]}" "${RUNS[3]}" > "$D/class_prompt.tsv"
fi
# each card's memory on nvidia-smi's scale (the driver's reserve, the card log's first reading, taken off): after
# loading (the last sample before the first task) and at its highest over the tasks, and what that leaves of the
# 6,068 MiB a card offers (card_memory.tsv)
cm=(); for r in "${RUNS[@]}"; do mapfile -t w < "$T/window_$r"; cm+=("$r=$T/rig_$r/cards.csv=${w[0]}=${w[1]}"); done
py $D/run/card_memory.py "${CARD_TOTAL_MIB:-6068}" "${cm[@]}" > "$D/card_memory.tsv"
