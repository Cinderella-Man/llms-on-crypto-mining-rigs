#!/usr/bin/env python3
"""c930_cuts.py: C930's (Q1668) two checks of `cuts=1` on one load of rig_taper.sh: did the server read every call as
C905's CPU plan says (plans.tsv's `before` column for the control, `after` for `cuts=1`), and what did each call whose
plan changes cost, pass by pass?

    python3 -I c930_cuts.py --load c930/load-A --plans ../2026-11-21_c905_taper_cuts/plans.tsv \
        --setting short=1400:64 --out tables

The server's `ubatch taper:` lines are matched to requests in order, as c222_tables.py and r225_plans.py do. Every
request counts here, the warm-up pass and the unmeasured prefix requests ("reset") included (column kind); the class
is c222_tables.py's, on the replayed turns only.

Writes, in --out:
  plan_check.tsv     per pass and arm: calls logged, calls whose size is in plans.tsv, of those the ones read as the
                     plan expects for that arm, and the ones that differ (listed as size: logged / expected)
  changed_calls.tsv  every request whose call size has a plan that cuts=1 changes: pass by pass prompt ms, the
                     control passes' mean and largest difference, the cuts=1 passes' mean, the difference
  changed_sum.tsv    per pass: prompt seconds over the changed requests (turns only, and all), and over the others
Standard library only.
"""
import argparse, collections, csv, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True)
ap.add_argument("--plans", required=True, help="C905's plans.tsv")
ap.add_argument("--setting", default="short=1400:64", help="the plans.tsv setting and the control passes' arm")
ap.add_argument("--test", default=None, help="the arm read with plans.tsv's after column (default: SETTING,cuts=1)")
ap.add_argument("--drop-pass", type=int, action="append", default=[0])
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

plan = {}
for r in csv.DictReader(open(a.plans), delimiter="\t"):
    if r["setting"] == a.setting:
        plan[int(r["call_tokens"])] = (r["before"], r["after"])
changed = {n for n, (b, f) in plan.items() if b != f}

reqs = list(csv.DictReader(open(os.path.join(a.load, "requests.tsv")), delimiter="\t"))
ansi = re.compile(r"\x1b\[[0-9;]*m")
pend, k = [], 0
for line in open(os.path.join(a.load, "server.log"), errors="replace"):
    line = ansi.sub("", line)
    m = re.search(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$", line)
    if m:
        pend.append((int(m.group(1)), m.group(5).strip()))
        continue
    if "prompt eval time =" in line:
        if k < len(reqs):
            reqs[k]["calls"] = pend
        pend, k = [], k + 1
if k != len(reqs):
    raise SystemExit(f"{k} timing lines for {len(reqs)} requests")


TEST = a.test or a.setting + ",cuts=1"


def is_cuts(arm):
    return arm == TEST


# plan check
rows = []
for p in sorted({int(r["pass"]) for r in reqs}):
    rs = [r for r in reqs if int(r["pass"]) == p]
    arm = rs[0]["arm"]
    n = known = ok = 0
    bad = []
    for r in rs:
        for tok, got in r.get("calls", []):
            n += 1
            if tok in plan:
                known += 1
                want = plan[tok][1] if is_cuts(arm) else plan[tok][0]
                if got == want:
                    ok += 1
                else:
                    bad.append(f"{tok}: {got} / {want}")
    rows.append([p, arm, n, known, ok, known - ok, "; ".join(bad) or "-"])
with open(os.path.join(a.out, "plan_check.tsv"), "w") as f:
    f.write("pass\tarm\tcalls_logged\tcalls_in_plan\tas_planned\tdiffer\tdifferences\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")

# changed calls, pass by pass
meas = [p for p in sorted({int(r["pass"]) for r in reqs}) if p not in a.drop_pass]
arm_of = {int(r["pass"]): r["arm"] for r in reqs}
ctl = [p for p in meas if arm_of[p] == a.setting]
cut = [p for p in meas if is_cuts(arm_of[p])]
key = lambda r: (r["kind"], r["task"], r["turn"])
by = collections.defaultdict(dict)
size = {}
for r in reqs:
    tok = [t for t, _ in r.get("calls", [])]
    by[key(r)][int(r["pass"])] = float(r["prompt_ms"])
    size[key(r)] = tok
order = []
for r in reqs:
    if int(r["pass"]) == meas[0]:
        order.append(key(r))
with open(os.path.join(a.out, "changed_calls.tsv"), "w") as f:
    f.write("kind\ttask\tturn\tcall_tokens\tplan_before\tplan_after\t" +
            "\t".join(f"p{p}:{arm_of[p]}" for p in meas) +
            "\tcontrol_mean_ms\tcontrol_spread_ms\tcuts_mean_ms\tdiff_ms\n")
    for kk in order:
        toks = size[kk]
        if not any(t in changed for t in toks):
            continue
        t = [x for x in toks if x in changed][0]
        c = [by[kk][p] for p in ctl]
        x = [by[kk][p] for p in cut]
        cm, xm = statistics.mean(c), statistics.mean(x)
        f.write("\t".join([kk[0], kk[1], kk[2], str(t), plan[t][0], plan[t][1]] +
                          [f"{by[kk][p]:.1f}" for p in meas] +
                          [f"{cm:.1f}", f"{max(c) - min(c):.1f}", f"{xm:.1f}", f"{xm - cm:+.1f}"]) + "\n")
with open(os.path.join(a.out, "changed_sum.tsv"), "w") as f:
    f.write("pass\tarm\tchanged_requests\tchanged_turns\tchanged_turns_s\tchanged_all_s\tother_turns_s\n")
    for p in meas:
        rs = [r for r in reqs if int(r["pass"]) == p]
        ch = [r for r in rs if any(t in changed for t, _ in r.get("calls", []))]
        cht = [r for r in ch if r["kind"] == "turn"]
        oth = [r for r in rs if r not in ch and r["kind"] == "turn"]
        s = lambda xs: sum(float(r["prompt_ms"]) for r in xs) / 1000
        f.write(f"{p}\t{arm_of[p]}\t{len(ch)}\t{len(cht)}\t{s(cht):.2f}\t{s(ch):.2f}\t{s(oth):.2f}\n")
print(open(os.path.join(a.out, "plan_check.tsv")).read())
print(open(os.path.join(a.out, "changed_sum.tsv")).read())
