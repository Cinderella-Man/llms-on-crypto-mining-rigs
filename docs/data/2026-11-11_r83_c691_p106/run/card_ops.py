#!/usr/bin/env python3
"""card_ops.py: R83 (Q1300). The launch groups of the backend samplers' PAD, TOP_K and its gather on the head's card
over a whole profiled load (every graph: the main model's verify and prompt passes and the MTP draft's passes), from
2026-11-10_served_verify_parts/run/verify_prof.sh's group-mode GGML_CUDA_PROFILE trace. verify_parts.py reports only
the verify passes; the switch is read per context, so the draft's sampler chains change too.

  python3 -I card_ops.py --device 8 --event-us 7.6 [--select 'PAD,TOP_K top_k,GET_ROWS top_k_rows'] \
      [--parts PARTS] --out DIR NAME=prof-verify.csv ... > card_ops.md

A launch group is selected when its label is a selector or starts with a selector and a space; groups are counted per
full label (the op and its tensor's name). --event-us is subtracted per group, as verify_parts.py does. With --parts
(verify_parts.py's --out folders, PARTS/NAME/cut.tsv) the load's generation steps are summed over its requests and
the selected time is also given per generation step (the whole load's time over its generation steps: it includes the
prompts' passes, so it is an upper bound per step).
Out: DIR/card_ops.tsv (load, label, groups, sum_us, net_ms), DIR/card_ops_totals.tsv (load, groups, net_ms,
generation_steps, net_ms_per_step); the Markdown report to stdout. Standard library only."""
import argparse, collections, csv, os

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+"); ap.add_argument("--device", type=int, default=8)
ap.add_argument("--event-us", type=float, default=0.0); ap.add_argument("--select", default="PAD,TOP_K top_k,GET_ROWS top_k_rows")
ap.add_argument("--parts", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sel = [s.strip() for s in a.select.split(",") if s.strip()]
rows, tot = [], []
for t in a.traces:
    name, path = t.split("=", 1)
    n, s = collections.Counter(), collections.defaultdict(float)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                d, us, lab = int(r["device"]), float(r["us"]), r["label"] or ""
            except (KeyError, TypeError, ValueError):
                continue  # a line cut at the server's exit
            if d == a.device and any(lab == x or lab.startswith(x + " ") for x in sel):
                n[lab] += 1
                s[lab] += us
    for lab in sorted(n):
        rows.append((name, lab, n[lab], s[lab], (s[lab] - n[lab] * a.event_us) / 1000))
    g = sum(n.values()); net = sum((s[l] - n[l] * a.event_us) / 1000 for l in n)
    steps = ""
    cp = os.path.join(a.parts, name, "cut.tsv") if a.parts else ""
    if cp and os.path.exists(cp):
        with open(cp) as f:
            steps = sum(int(r["generation_steps"]) for r in csv.DictReader(f, delimiter="\t"))
    tot.append((name, g, net, steps, net / steps if steps else float("nan")))
with open(os.path.join(a.out, "card_ops.tsv"), "w") as w:
    w.write("load\tlabel\tgroups\tsum_us\tnet_ms\n")
    for x in rows:
        w.write(f"{x[0]}\t{x[1]}\t{x[2]}\t{x[3]:.1f}\t{x[4]:.3f}\n")
with open(os.path.join(a.out, "card_ops_totals.tsv"), "w") as w:
    w.write("load\tgroups\tnet_ms\tgeneration_steps\tnet_ms_per_step\n")
    for x in tot:
        w.write(f"{x[0]}\t{x[1]}\t{x[2]:.3f}\t{x[3]}\t{x[4]:.3f}\n")
print(f"## Card {a.device}: the selected launch groups over each whole load ({', '.join(sel)}; {a.event_us} µs a group subtracted)\n")
print("| Load | Label | Groups | Sum, µs | Net, ms |")
print("|---|---|---:|---:|---:|")
for x in rows:
    print(f"| {x[0]} | {x[1]} | {x[2]} | {x[3]:.1f} | {x[4]:.3f} |")
print("\n| Load | Groups | Net, ms | Generation steps | Net ms a generation step |")
print("|---|---:|---:|---:|---:|")
for x in tot:
    print(f"| {x[0]} | {x[1]} | {x[2]:.3f} | {x[3]} | {x[4]:.3f} |")
