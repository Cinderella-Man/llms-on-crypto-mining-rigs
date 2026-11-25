#!/usr/bin/env bash
# R229 (Q1642): Flash-Next's tables for the runs done so far, standard library only (python3 -I), from the repository
# root:
#   bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/tables.sh TABLES RUN...
# RUN names the runs in the order they ran (s2a s1a s2b s1b); this folder's agent/<run>/ holds each run's turns.csv and
# attempts.csv (the client's res/ files with the harness's arm v4r renamed to the run by
# docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py). TABLES holds, per run:
#   rig/server-<run>.log                              the rig's llama-server log
#   rig_<run>/meminfo.log, rig_<run>/cards.csv        the rig's meminfo.log and card log (scripts/card_log.py's)
#   proxy_<run>/proxy.jsonl, proxy.jsonl.bodies.gz    the relay proxy's request rows and recorded bodies
#   window_<run>                                      the first task's start and the last task's end, the rig's local
#                                                     time "YYYY-MM-DD HH:MM:SS", one a line
# and rig/server-r90v3.log (R90's v3: "a task at equal work" prices its work, as in C854's, R208's, R220's, R227's and
# C912's tables). This is C912's run/tables.sh (docs/data/2026-11-24_c912_flash_agent/) without its context rows and
# its two-run class, plus: the draft shortlist's own counts from each server log (shortlist.tsv: the full-head rounds
# after a miss its set-up line names, and its drafts with the shortlist and with the full head, the target tokens
# outside the list and the ids in it at the server's end), and with all four runs named the class of the one-round
# arm (s1a, s1b) against the preset (s2a, s2b) by bench/agent_check/class_arms.py (class.tsv; class_nocap.tsv without
# the tasks that ran to a cap in any run). Writes this folder's tables.
set -u
T=${1:?}; shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs named"; exit 1; }
D=docs/data/2026-11-25_r229_flash_shortlist_agent; C=docs/data/2026-11-18_c854b_p106_check
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
# prompt seconds by the turn's new tokens, in the taper's buckets: turns, tokens, seconds, ms a token; all tasks and
# without the capped ones
CAPPED=$CAPPED py - "${named[@]}" > "$D/prompt_buckets.tsv" <<'PY'
import csv, os, sys
skip = set(filter(None, os.environ.get("CAPPED", "").split(",")))
edges = [(0, 599, "under 600"), (600, 1023, "600-1,023"), (1024, 1999, "1,024-1,999"), (2000, 10**9, "2,000+")]
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
# reply re-reads: turns whose cached prefix stops inside the previous reply, their tokens and prompt seconds
rr=()
for r in "${RUNS[@]}"; do rr+=("$r:v4r:$T/proxy_$r/proxy.jsonl:$T/proxy_$r/proxy.jsonl.bodies.gz:$T/rig/server-$r.log"); done
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
# the draft shortlist's counts, from each server log: the rounds its set-up line names and the last count line
sl=(); for r in "${RUNS[@]}"; do sl+=("$r=$T/rig/server-$r.log"); done
py - "${sl[@]}" > "$D/shortlist.tsv" <<'PY2'
import re, sys
print("run\tfull_head_rounds_after_a_miss\tdrafts_with_shortlist\tdrafts_with_full_head\ttarget_tokens_outside\tids_in_list_at_end")
for arg in sys.argv[1:]:
    name, log = arg.split("=", 1)
    rounds, rep = "none", ("none",) * 4
    for ln in open(log, errors="replace"):
        m = re.search(r"full head for (\d+) drafts after a target token outside it", ln)
        if m:
            rounds = m.group(1)
        m = re.search(r"draft head shortlist: (\d+) drafts with it, (\d+) with the full head, (\d+) target "
                      r"tokens outside it; (\d+) ids in it", ln)
        if m:
            rep = m.groups()
    print("\t".join((name, rounds) + tuple(rep)))
PY2
# the class: the one-round arm against the preset, each arm's two runs, by the release check's rule (a gain or a loss
# only beyond the larger of the two arms' spreads)
if [ ${#RUNS[@]} -eq 4 ]; then
  py bench/agent_check/class_arms.py "$D/agent.tsv" "s1=${RUNS[1]},${RUNS[3]}" "s2=${RUNS[0]},${RUNS[2]}" > "$D/class.tsv"
  if [ -n "$CAPPED" ]; then
    py bench/agent_check/class_arms.py "$D/agent_nocap.tsv" "s1=${RUNS[1]},${RUNS[3]}" "s2=${RUNS[0]},${RUNS[2]}" \
      > "$D/class_nocap.tsv"
  else
    rm -f "$D/class_nocap.tsv"
  fi
fi
# the draft's own measures, classed by the same rule (class_draft.tsv): the draft's acceptance and tokens a step
# (agent.tsv), the decode step over all requests (steps.md), and from shortlist.tsv the full-head drafting rounds a run,
# those rounds a miss, and the misses a 1,000 generated tokens (steps.md's tokens)
if [ ${#RUNS[@]} -eq 4 ]; then
  py - "$D/agent.tsv" "$D/steps.md" "$D/shortlist.tsv" "${RUNS[@]}" > "$D/class_draft.tsv" <<'PY2'
import csv, sys
agent, steps, short, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
a = {r["arm"]: r for r in csv.DictReader(open(agent), delimiter="\t")}
sl = {r["run"]: r for r in csv.DictReader(open(short), delimiter="\t")}
st = {}
for ln in open(steps):
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) == 9 and c[1] == "all":
        st[c[0]] = (int(c[3]), float(c[8]))
val = {}
for r in runs:
    full, miss = int(sl[r]["drafts_with_full_head"]), int(sl[r]["target_tokens_outside"])
    val[r] = {"acceptance": float(a[r]["acceptance"]), "tokens_per_step": float(a[r]["tokens_per_step"]),
              "ms_per_step_all": st[r][1], "full_head_rounds": full, "full_head_rounds_per_miss": full / miss,
              "misses_per_1000_tokens": miss * 1000 / st[r][0]}
measures = [("acceptance", "the draft's acceptance", "higher"), ("tokens_per_step", "tokens a step", "higher"),
            ("ms_per_step_all", "ms a decode step, all requests", "lower"),
            ("full_head_rounds", "full-head drafting rounds a run", "lower"),
            ("full_head_rounds_per_miss", "full-head drafting rounds a miss", "lower"),
            ("misses_per_1000_tokens", "target tokens outside the list a 1,000 generated", "lower")]
arm, base = [runs[1], runs[3]], [runs[0], runs[2]]
print("measure\tarm_runs\tbase_runs\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for k, label, better in measures:
    av, bv = [val[r][k] for r in arm], [val[r][k] for r in base]
    am, bm = sum(av) / 2, sum(bv) / 2
    asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
    if abs(d) <= max(asp, bsp):
        cls = "negligible"
    else:
        cls = "gain" if (d < 0) == (better == "lower") else "loss"
    f = lambda x: f"{x:.4g}"
    print(f"{label}\t{' '.join(map(f, av))}\t{' '.join(map(f, bv))}\t{f(am)}\t{f(bm)}\t{f(asp)}\t{f(bsp)}\t{d:+.4g}\t"
          f"{d / bm * 100:+.1f} %\t{cls}")
PY2
fi
