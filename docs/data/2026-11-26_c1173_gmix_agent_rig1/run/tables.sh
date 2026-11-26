#!/usr/bin/env bash
# C1173 (Q1906): Flash-Next's tables for the runs done so far, standard library only (python3 -I), from the repository
# root:
#   bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/tables.sh TABLES RUN...
# RUN names the runs in the order they ran (gmixa pmixa gmixb pmixb); this folder's agent/<run>/ holds each run's
# turns.csv and attempts.csv (the client's res/ files with the harness's arm v4r renamed to the run by
# docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py). TABLES holds, per run:
#   rig/server-<run>.log                              the rig's llama-server log
#   rig_<run>/meminfo.log, rig_<run>/cards.csv        the rig's meminfo.log and card log (scripts/card_log.py's)
#   proxy_<run>/proxy.jsonl, proxy.jsonl.bodies.gz    the relay proxy's request rows and recorded bodies
#   window_<run>                                      the first task's start and the last task's end, the rig's local
#                                                     time "YYYY-MM-DD HH:MM:SS", one a line
# and rig/server-r90v3.log (R90's v3: "a task at equal work" prices its work, as in C854's, R208's, R220's, R227's,
# C912's and R229's tables). This is R229's run/tables.sh (docs/data/2026-11-25_r229_flash_shortlist_agent/) without
# its shortlist counts, plus: the model file each server loaded (models.tsv); each task's minutes at equal work split
# into its three parts, R90's time outside the server, R90's prompt tokens at the run's prompt speed and R90's
# generated tokens at the run's generation speed, with each part's share, and each run's own session split the same
# way (shares.tsv; shares_nocap.tsv without the tasks that ran to a cap in any run); and with all four runs named the
# class of gmix-q2 (gmixa, gmixb) against pmix-q2-fast (pmixa, pmixb) by bench/agent_check/class_arms.py (class.tsv;
# class_nocap.tsv without the capped tasks) and of the parts by the same rule (class_shares.tsv). Writes this folder's
# tables.
set -u
T=${1:?}; shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs named"; exit 1; }
D=docs/data/2026-11-26_c1173_gmix_agent_rig1; C=docs/data/2026-11-18_c854b_p106_check
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
# the model file each server loaded (its load_model line names the file; the base name only)
ml=(); for r in "${RUNS[@]}"; do ml+=("$r=$T/rig/server-$r.log"); done
py - "${ml[@]}" > "$D/models.tsv" <<'PY2'
import os, re, sys
print("run\tmodel_file")
for arg in sys.argv[1:]:
    name, log = arg.split("=", 1)
    f = "none"
    for ln in open(log, errors="replace"):
        m = re.search(r"load_model: loading model '([^']+)'", ln)
        if m and f == "none":
            f = os.path.basename(m.group(1))
    print(f"{name}\t{f}")
PY2
# each task's minutes at equal work in its three parts (agent_tables.py's formula: R90's time outside the server, R90's
# prompt tokens at the run's prompt speed, R90's generated tokens at the run's generation speed, over R90's tasks), and
# each run's own session in the same three parts (its wall time outside the server, its prompt seconds, its decode
# seconds, a task); all tasks, and without the capped ones
for tag in "" _nocap; do
  [ "$tag" = _nocap ] && [ -z "$CAPPED" ] && { rm -f "$D/shares_nocap.tsv"; continue; }
  skip=; [ "$tag" = _nocap ] && skip=$CAPPED
  SKIP_TASKS=$skip py - "$T/res" r90v3 "${RUNS[@]}" > "$D/shares$tag.tsv" <<'PY2'
import csv, os, sys
res, ref, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
skip = set(filter(None, os.environ.get("SKIP_TASKS", "").split(",")))
turns = [t for t in csv.DictReader(open(f"{res}/turns.csv")) if t["task_id"] not in skip]
atts = [a for a in csv.DictReader(open(f"{res}/attempts.csv")) if a["task_id"] not in skip]
def st(arm):
    tt = [t for t in turns if t["arm"] == arm]
    aa = [a for a in atts if a["file"] == arm]
    f = lambda k: sum(float(t[k] or 0) for t in tt)
    return dict(n=len(aa), pn=f("prompt_n"), ps=f("prompt_ms") / 1000, gn=f("predicted_n"), gs=f("predicted_ms") / 1000,
                wall=sum(float(a["wall_s"]) for a in aa))
r = st(ref)
r_other = r["wall"] - r["ps"] - r["gs"]
print("run\ttasks\teq_other_min\teq_prompt_min\teq_decode_min\teq_min\teq_prompt_share\teq_decode_share\t"
      "own_other_min\town_prompt_min\town_decode_min\town_min\town_prompt_share\town_decode_share")
for run in runs:
    s = st(run)
    pr, gr = s["pn"] / s["ps"], s["gn"] / s["gs"]
    o, p, d = r_other / r["n"] / 60, r["pn"] / pr / r["n"] / 60, r["gn"] / gr / r["n"] / 60
    eq = o + p + d
    oo, op, od = (s["wall"] - s["ps"] - s["gs"]) / s["n"] / 60, s["ps"] / s["n"] / 60, s["gs"] / s["n"] / 60
    own = oo + op + od
    print(f"{run}\t{s['n']}\t{o:.3f}\t{p:.3f}\t{d:.3f}\t{eq:.3f}\t{p / eq * 100:.1f} %\t{d / eq * 100:.1f} %\t"
          f"{oo:.3f}\t{op:.3f}\t{od:.3f}\t{own:.3f}\t{op / own * 100:.1f} %\t{od / own * 100:.1f} %")
PY2
done
# the class: gmix-q2 against pmix-q2-fast, each file's two runs, by the release check's rule (a gain or a loss only
# beyond the larger of the two arms' spreads)
if [ ${#RUNS[@]} -eq 4 ]; then
  py bench/agent_check/class_arms.py "$D/agent.tsv" "gmix=${RUNS[0]},${RUNS[2]}" "pmix=${RUNS[1]},${RUNS[3]}" > "$D/class.tsv"
  if [ -n "$CAPPED" ]; then
    py bench/agent_check/class_arms.py "$D/agent_nocap.tsv" "gmix=${RUNS[0]},${RUNS[2]}" "pmix=${RUNS[1]},${RUNS[3]}" \
      > "$D/class_nocap.tsv"
  else
    rm -f "$D/class_nocap.tsv"
  fi
  # the parts of a task at equal work, and the decode step over all requests (steps.md), by the same rule
  for tag in "" _nocap; do
    [ -f "$D/shares$tag.tsv" ] || continue
    py - "$D/shares$tag.tsv" "$D/steps.md" "$tag" "${RUNS[@]}" <<'PY2'
import csv, sys
shares, steps, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sh = {r["run"]: r for r in csv.DictReader(open(shares), delimiter="\t")}
st = {}
for ln in open(steps):
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) == 9 and c[1] == "all":
        st[c[0]] = float(c[8])
measures = [("eq_prompt_min", "prompt part of a task at equal work, min"),
            ("eq_decode_min", "decode part of a task at equal work, min"),
            ("eq_min", "a task at equal work, min (the three parts)"),
            ("own_prompt_min", "the run's own prompt minutes a task"),
            ("own_decode_min", "the run's own decode minutes a task")]
if not tag:
    measures.append(("ms_step", "ms a decode step, all requests"))
arm, base = [runs[0], runs[2]], [runs[1], runs[3]]
out = open(shares.replace("shares", "class_shares"), "w")
out.write("measure\tarm_runs\tbase_runs\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass\n")
for k, label in measures:
    v = lambda r: st[r] if k == "ms_step" else float(sh[r][k])
    av, bv = [v(r) for r in arm], [v(r) for r in base]
    am, bm = sum(av) / 2, sum(bv) / 2
    asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
    cls = "negligible" if abs(d) <= max(asp, bsp) else ("gain" if d < 0 else "loss")
    f = lambda x: f"{x:.4g}"
    out.write(f"{label}\t{' '.join(map(f, av))}\t{' '.join(map(f, bv))}\t{f(am)}\t{f(bm)}\t{f(asp)}\t{f(bsp)}\t"
              f"{d:+.4g}\t{d / bm * 100:+.1f} %\t{cls}\n")
PY2
  done
fi
