#!/usr/bin/env python3
# Per job of a rig_run.sh output folder: when the first token was sampled (seconds after the job's clock started) and the
# trace-based rate (tokens per second from the first to the last sampled token), next to the tok_s in summary.tsv.
# Before fork commit 99c3a7c00 llama-pipespec's clock started before the prompt's tail had run, so its tok_s carried a
# 0.1-3.3 s wait; the trace rate does not. rig_run.sh now reports trace-based rates itself (column basis = trace).
# Usage: trace_rates.py <rig_run.sh output folder with *.trace and summary.tsv> [...]
import sys, glob, os, csv
for d in sys.argv[1:]:
    summ = {}
    p = os.path.join(d, "summary.tsv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter="\t"):
            summ[r.get("name") or r.get("job")] = r
    print(f"== {d}")
    plain = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.trace"))):
        name = os.path.basename(f)[:-6]
        t = [int(l.split()[0]) for l in open(f).readlines()[1:]]
        if len(t) < 3:
            continue
        d_ = [(b - a) for a, b in zip(t, t[1:])]
        rate = 1e6 * len(d_) / (t[-1] - t[0])
        rows.append((name, t[0] / 1e6, rate, summ.get(name, {}).get("tok_s", "-")))
    for name, t0, rate, s in rows:
        print(f"  {name:28s} first sample at {t0:6.2f} s  trace tok/s {rate:7.2f}  summary tok/s {s}")
