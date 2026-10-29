"""Paired difference of two heads scored by score.py on the same positions (e.g. a training arm against its control):
per window, the mean of B - A over all scored positions and over the head's positions in the server's order (where
neither the copy lookup nor a 2-token continuation guesses, source_split.py's cont1 and none), with a 95 % interval
from resampling whole sessions of the per-position differences.

usage: python paired_diff.py ROWS.jsonl A.npz B.npz [--windows g9,g9miss] [--metric uniform]
Needs numpy.
"""
import argparse
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np

_spec = importlib.util.spec_from_file_location("source_split", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "source_split.py"))
source_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source_split)


def ci(v, groups, n_boot=2000, seed=0):
    """mean and a 95 % interval from resampling whole sessions"""
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), (n_boot, len(ug)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--windows", default="g9,g9miss")
    ap.add_argument("--metric", default="uniform", choices=("uniform", "coupled", "top", "overlap"))
    args = ap.parse_args()
    za, zb = np.load(args.a), np.load(args.b)
    seq, j = za["seq"], za["j"]
    if not (np.array_equal(seq, zb["seq"]) and np.array_equal(j, zb["j"])):
        raise SystemExit("the two files score different positions")
    by_seq = defaultdict(list)
    for i in range(len(seq)):
        by_seq[int(seq[i])].append(i)
    head = np.zeros(len(seq), bool)
    for line in open(args.rows):
        r = json.loads(line)
        sid = int(r["id"].split("_")[-1])
        if sid not in by_seq:
            continue
        lab = source_split.labels(r)
        for i in by_seq[sid]:
            x = lab.get(int(j[i]))
            head[i] = x is not None and x[0] in ("cont1", "none")
    print(f"B - A, metric {args.metric}; A {args.a}; B {args.b}")
    print("\n| window | positions | A | B | B - A | 95 % |")
    print("|---|---|---|---|---|---|")
    for w in args.windows.split(","):
        a, b = za[f"{w}/{args.metric}"].astype(float), zb[f"{w}/{args.metric}"].astype(float)
        for name, sel in (("all", np.ones(len(seq), bool)), ("head positions", head)):
            m, lo, hi = ci(b[sel] - a[sel], seq[sel])
            print(f"| {w}, {name} | {int(sel.sum())} | {a[sel].mean():.4f} | {b[sel].mean():.4f} | {m:+.4f} | "
                  f"{lo:+.4f} – {hi:+.4f} |")


if __name__ == "__main__":
    main()
