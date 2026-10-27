#!/usr/bin/env python3
"""GGML_CUDA_PROFILE traces -> one CSV of GPU time per launch-group label, plus one CSV of totals.

usage: prof_to_csv.py [--tokens 24] OUT_PREFIX NAME=TRACE.csv...
Writes OUT_PREFIX_by_label.csv (run, label, ms per token, share, calls per token, us per call) and
OUT_PREFIX_totals.csv (run, devices, steady-state evaluations, GPU ms per evaluation summed over the devices, per
device, and wall ms per evaluation from the first device's host timestamps).

Same selection and arithmetic as scripts/cuda_profile_summary.py: the last --tokens evaluations of every device are
the steady state; a label's time is summed over the devices and divided by the number of evaluations. The trace comes
from the fork's CUDA backend (one line per launch group: device, eval, host_ms, group, label, us); each group boundary
is a CUDA event, so profiled runs are a little slower than unprofiled ones.
"""
import argparse
import csv
from collections import defaultdict


def summarize(path, n_tokens):
    rows = list(csv.DictReader(open(path)))
    evals = defaultdict(set)
    for r in rows:
        evals[int(r["device"])].add(int(r["eval"]))
    keep = {d: set(sorted(e)[-n_tokens:]) for d, e in evals.items()}
    n_tok = min(len(k) for k in keep.values())
    per_label = defaultdict(lambda: [0.0, 0])
    per_dev = defaultdict(float)
    host = defaultdict(dict)
    for r in rows:
        d, e = int(r["device"]), int(r["eval"])
        if e not in keep[d]:
            continue
        us = float(r["us"])
        per_label[r["label"]][0] += us
        per_label[r["label"]][1] += 1
        per_dev[d] += us
        host[d][e] = float(r["host_ms"])
    total = sum(v[0] for v in per_label.values())
    d0 = min(host)
    ts = [host[d0][e] for e in sorted(host[d0])]
    wall = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else 0.0
    return n_tok, total, per_dev, wall, per_label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("prefix")
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    with open(a.prefix + "_by_label.csv", "w", newline="") as fl, open(a.prefix + "_totals.csv", "w", newline="") as ft:
        wl = csv.writer(fl, lineterminator="\n")
        wt = csv.writer(ft, lineterminator="\n")
        wl.writerow(["run", "label", "ms_per_eval", "share_pct", "calls_per_eval", "us_per_call"])
        wt.writerow(["run", "devices", "evaluations", "gpu_ms_per_eval", "gpu_ms_per_eval_by_device",
                     "wall_ms_per_eval"])
        for spec in a.traces:
            name, path = spec.split("=", 1)
            n_tok, total, per_dev, wall, per_label = summarize(path, a.tokens)
            wt.writerow([name, len(per_dev), n_tok, f"{total / n_tok / 1000:.3f}",
                         " ".join(f"{per_dev[d] / n_tok / 1000:.3f}" for d in sorted(per_dev)), f"{wall:.3f}"])
            for label, (us, n) in sorted(per_label.items(), key=lambda kv: -kv[1][0]):
                wl.writerow([name, label, f"{us / n_tok / 1000:.4f}", f"{100 * us / total:.2f}", f"{n / n_tok:.1f}",
                             f"{us / n:.2f}"])
            print(f"{name}: {len(per_dev)} devices, {n_tok} evaluations, {total / n_tok / 1000:.2f} ms GPU, "
                  f"{wall:.2f} ms wall, {len(per_label)} labels")


if __name__ == "__main__":
    main()
