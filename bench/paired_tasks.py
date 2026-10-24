"""Paired comparison of quantized files on per-attempt task results (Q41's analysis).

Reads the per_attempt.csv files of bench/bfcl_flint (single- and multi-turn) and bench/livecodebench_flint, one
row per task, file and attempt. For every lane and every file against the reference file:
- pass@1 per task is the mean over that file's attempts; the paired difference is taken per task over the attempts
  both files have for it (so runs still in progress compare like with like), then averaged (Miller, arXiv
  2411.00640: average the K samples first, then the standard error across tasks);
- standard error plain and clustered by the `cluster` column, 90 % and 95 % intervals with the larger of the two,
  and a task-level bootstrap interval;
- the non-inferiority verdict: the lower 90 % bound above -margin;
- the process gate: (malformed or runaway) rate per task, its paired difference and the upper 90 % bound against
  the lane's limit (+0.5 points on the BFCL lanes, +2 points on LiveCodeBench, where it is mostly cut-off samples);
- noise: E[p(1-p)] within tasks and the share of tasks whose first two attempts' verdicts differ.

usage: python paired_tasks.py --ref FILE [--margin LANE=POINTS ...] [--gate LANE=POINTS ...] [--bootstrap N] CSV ...
Needs numpy.
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np

DEFAULT_MARGIN = {"bfcl_single_turn": 1.5, "bfcl_multi_turn": 3.0, "livecodebench": 3.0}
DEFAULT_GATE = {"bfcl_single_turn": 0.5, "bfcl_multi_turn": 0.5, "livecodebench": 2.0}

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--ref", required=True, help="the bar, e.g. the UD-Q4_K_XL file's name in the CSV")
ap.add_argument("--margin", action="append", default=[], help="LANE=POINTS, overrides the defaults")
ap.add_argument("--gate", action="append", default=[], help="LANE=POINTS, overrides the defaults")
ap.add_argument("--bootstrap", type=int, default=10000)
args = ap.parse_args()
margins = dict(DEFAULT_MARGIN, **{k: float(v) for k, v in (m.split("=") for m in args.margin)})
gates = dict(DEFAULT_GATE, **{k: float(v) for k, v in (m.split("=") for m in args.gate)})

# lane -> file -> task -> {attempt: (pass, process_failure)}; lane -> task -> cluster
runs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
clusters = defaultdict(dict)
for path in args.csv:
    for r in csv.DictReader(open(path)):
        runs[r["lane"]][r["file"]][r["task_id"]][int(r["attempt"])] = (
            int(r["pass"]), int(r["malformed"]) | int(r["runaway"]))
        clusters[r["lane"]][r["task_id"]] = r["cluster"] or r["task_id"]


def per_task(tasks, field):
    return {t: sum(v[field] for v in a.values()) / len(a) for t, a in tasks.items()}


def matched(a, b, field):
    """A task's mean in both files over the attempts both have."""
    shared = set(a) & set(b)
    return (sum(a[x][field] for x in shared) / len(shared) - sum(b[x][field] for x in shared) / len(shared),
            len(shared))


def within_var(tasks):
    """E[p(1-p)] from K attempts per task (unbiased), the per-sample noise left after pairing."""
    vals = []
    for a in tasks.values():
        k = len(a)
        if k >= 2:
            p = sum(v[0] for v in a.values()) / k
            vals.append(p * (1 - p) * k / (k - 1))
    return sum(vals) / len(vals) if vals else float("nan")


def self_discordance(tasks):
    pairs = [[a[x][0] for x in sorted(a)[:2]] for a in tasks.values() if len(a) >= 2]
    return sum(x != y for x, y in pairs) / len(pairs) if pairs else float("nan")


def paired(d, cl):
    """Mean, plain and cluster-robust standard errors of per-task differences."""
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    sums = defaultdict(float)
    for x, c in zip(d - mean, cl):
        sums[c] += x
    se_cl = math.sqrt(sum(s * s for s in sums.values()) * n / (n - 1)) / n
    return mean, se, se_cl


rng = np.random.default_rng(0)
for lane in sorted(runs):
    files = runs[lane]
    if args.ref not in files:
        print(f"## {lane}: reference {args.ref} missing\n")
        continue
    m, gate = margins.get(lane, 3.0), gates.get(lane, 0.5)
    print(f"## {lane} (reference {args.ref}, margin {m} points, process gate +{gate} points)\n")
    print("| File | tasks | K | pass@1 | E[p(1-p)] | self-discordance | paired tasks (samples) | Δ pass@1 | "
          "SE (clustered) | 90 % | 95 % | bootstrap 90 % | verdict | Δ malformed+runaway | upper 90 % | gate |")
    print("|---" * 16 + "|")
    ref = files[args.ref]
    for f in [args.ref] + sorted(x for x in files if x != args.ref):
        tasks = files[f]
        common = sorted(t for t in set(tasks) & set(ref) if set(tasks[t]) & set(ref[t]))
        k = int(np.median([len(a) for a in tasks.values()]))
        pf, rf = per_task(tasks, 0), per_task(ref, 0)
        head = (f"| {f} | {len(tasks)} | {k} | {100 * sum(pf.values()) / len(pf):.2f} | {within_var(tasks):.3f} | "
                f"{100 * self_discordance(tasks):.1f} % |")
        if f == args.ref:
            print(head + " — |" * 10)
            continue
        if len(common) < 2:
            print(head + " 0 | no attempts shared with the reference |" + " — |" * 8)
            continue
        cl = [clusters[lane][t] for t in common]
        dm = [matched(tasks[t], ref[t], 0) for t in common]
        d = np.array([x for x, _ in dm]) * 100
        mean, se, se_cl = paired(d, cl)
        s = max(se, se_cl)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.bootstrap)])
        lo90, hi90 = mean - 1.645 * s, mean + 1.645 * s
        ok = lo90 > -m
        g = np.array([matched(tasks[t], ref[t], 1)[0] for t in common]) * 100
        gmean, gse, gse_cl = paired(g, cl)
        gup = gmean + 1.645 * max(gse, gse_cl)
        print(head + f" {len(common)} ({sum(n for _, n in dm)}) | {mean:+.2f} | {se:.2f} ({se_cl:.2f}) | {lo90:+.2f} to {hi90:+.2f} | "
              f"{mean - 1.96 * s:+.2f} to {mean + 1.96 * s:+.2f} | "
              f"{np.percentile(boot, 5):+.2f} to {np.percentile(boot, 95):+.2f} | {'pass' if ok else 'FAIL'} | "
              f"{gmean:+.2f} | {gup:+.2f} | {'pass' if gup <= gate else 'FAIL'} |")
    print()
