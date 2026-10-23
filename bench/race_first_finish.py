"""First-finish race on sampled attempts (Q93): if k samples of one request run side by side and the first to close
its reasoning is kept, how much shorter is the reasoning, and is the kept answer as good?

Reads the per_attempt.csv files of the task harnesses (one row per task, file and attempt). For every lane, file
and task with at least k attempts, every k-subset of its attempts is taken (all of them, not a sample), and in each
the attempt with the least reasoning (ties share the pick); then:
- pass@1: the mean pass over all of the task's attempts (a random attempt);
- pass of the shortest: the mean over the k-subsets of the picked attempt's pass;
- the shortest reasoning over the mean reasoning, as a ratio of sums over tasks (the share of reasoning time left),
  and the same for all output tokens;
- the paired difference (shortest − random) per task, averaged, with a task-level bootstrap 95 % interval;
- CV: the mean over tasks of the standard deviation of reasoning length between attempts over its mean.
Reasoning length is the `reasoning_chars` column (characters inside <think>), or `output_tokens` with
--length output_tokens. On multi-turn tasks a row is a whole conversation, so its length sums all steps; a race
would run per step, so treat that lane's numbers as a rough guide.

usage: python race_first_finish.py [--length reasoning_chars|output_tokens] [--k 2,3,5] [--bootstrap N] CSV ...
Needs numpy.
"""
import argparse
import csv
from collections import defaultdict
from itertools import combinations

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--length", default="reasoning_chars")
ap.add_argument("--k", default="2,3,5")
ap.add_argument("--bootstrap", type=int, default=10000)
args = ap.parse_args()
ks = [int(x) for x in args.k.split(",")]

# lane -> file -> task -> list of (pass, length, output_tokens)
runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for path in args.csv:
    for r in csv.DictReader(open(path)):
        if r[args.length] in ("", None) or r["output_tokens"] in ("", None):
            continue
        runs[r["lane"]][r["file"]][r["task_id"]].append(
            (int(r["pass"]), float(r[args.length]), float(r["output_tokens"])))


def shortest(subset):
    """Pass and output tokens of the attempt with the least reasoning; ties share the pick."""
    lo = min(a[1] for a in subset)
    tied = [a for a in subset if a[1] == lo]
    return sum(a[0] for a in tied) / len(tied), lo, sum(a[2] for a in tied) / len(tied)


rng = np.random.default_rng(0)
for lane in sorted(runs):
    print(f"## {lane} (reasoning length = {args.length})\n")
    print("| File | k | tasks | pass@1 (random) | pass of shortest | Δ (95 % bootstrap) | shortest / mean reasoning "
          "| shortest / mean output tokens | CV of reasoning |")
    print("|---" * 9 + "|")
    for file in sorted(runs[lane]):
        tasks = runs[lane][file]
        for k in ks:
            rows = []
            for atts in tasks.values():
                if len(atts) < k:
                    continue
                picks = [shortest(s) for s in combinations(atts, k)]
                lengths = np.array([a[1] for a in atts])
                rows.append((np.mean([a[0] for a in atts]), np.mean([p[0] for p in picks]), lengths.mean(),
                             np.mean([p[1] for p in picks]), np.mean([a[2] for a in atts]),
                             np.mean([p[2] for p in picks]),
                             lengths.std() / lengths.mean() if lengths.mean() > 0 else 0.0))
            if not rows:
                continue
            m = np.array(rows)
            d = (m[:, 1] - m[:, 0]) * 100
            boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.bootstrap)])
            print(f"| {file} | {k} | {len(m)} | {100 * m[:, 0].mean():.2f} | {100 * m[:, 1].mean():.2f} | "
                  f"{d.mean():+.2f} ({np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}) | "
                  f"{m[:, 3].sum() / m[:, 2].sum():.3f} | {m[:, 5].sum() / m[:, 4].sum():.3f} | {m[:, 6].mean():.2f} |")
    print()
