#!/usr/bin/env python3
"""trace_census.py: which graphs a group-mode GGML_CUDA_PROFILE trace holds, per card. A group-mode trace names no
graph (columns device,eval,host_ms,group,label,us), so an evaluation's graph is taken from its launch groups: two
evaluations with the same labels in the same order ran the same graph.

  python3 -I trace_census.py <prof.csv> [--out census.tsv] [--gap-ms 150]

Per card and graph, in the order the card first ran them: evaluations, launch groups per evaluation, the median and
the sum of their GPU time, the host time of the first and the last one, and the graph's first labels. Then, per card,
the pauses longer than --gap-ms between consecutive evaluations (the breaks between requests). Standard library
only."""
import argparse, collections, csv, hashlib, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--out", default=""); ap.add_argument("--gap-ms", type=float, default=150.0)
a = ap.parse_args()

ev = collections.OrderedDict()  # (device, eval) -> [host_ms, [labels], us]
with open(a.trace, newline="") as f:
    for r in csv.DictReader(f):
        if int(r["group"]) < 0:
            continue
        e = ev.setdefault((int(r["device"]), int(r["eval"])), [float(r["host_ms"]), [], 0.0])
        e[1].append(r["label"])
        e[2] += float(r["us"])

rows = []
print("card\tgraph\tevals\tgroups\tmedian_ms\tsum_ms\tfirst_host_ms\tlast_host_ms\tfirst_labels")
for d in sorted({k[0] for k in ev}):
    es = sorted((e for k, e in ev.items() if k[0] == d), key=lambda e: e[0])
    by = collections.OrderedDict()
    for e in es:
        by.setdefault(hashlib.md5("\n".join(e[1]).encode()).hexdigest()[:8], []).append(e)
    for g, v in by.items():
        row = (d, g, len(v), len(v[0][1]), statistics.median(e[2] for e in v) / 1000, sum(e[2] for e in v) / 1000,
               v[0][0], v[-1][0], " | ".join(x[:60] for x in v[0][1][:3]))
        rows.append(row)
        print("%d\t%s\t%d\t%d\t%.3f\t%.1f\t%.1f\t%.1f\t%s" % row)
    gaps = [(b[0] - x[0], x[0]) for x, b in zip(es, es[1:]) if b[0] - x[0] > a.gap_ms]
    print(f"# card {d}: {len(es)} evaluations; pauses over {a.gap_ms:g} ms after host_ms: "
          + ", ".join(f"{t:.0f} (+{g:.0f})" for g, t in gaps[:12]) + (" ..." if len(gaps) > 12 else ""), file=sys.stderr)
if a.out:
    with open(a.out, "w") as f:
        f.write("card\tgraph\tevals\tgroups\tmedian_ms\tsum_ms\tfirst_host_ms\tlast_host_ms\tfirst_labels\n")
        for row in rows:
            f.write("%d\t%s\t%d\t%d\t%.3f\t%.1f\t%.1f\t%.1f\t%s\n" % row)
