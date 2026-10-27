#!/usr/bin/env python3
"""Groups prof_to_csv.py's per-label rows into tensor classes, one column of ms per evaluation per run.

usage: profile_classes.py BY_LABEL.csv > classes.csv
A matrix-product label is classed by its first weight's name and type (e.g. "routed experts gate/up, q5_K"); every
other label (norms, attention, routing, adds, copies) goes to "everything else".
"""
import csv
import sys
from collections import defaultdict

ROLE = {"attn_q": "attention q", "attn_k": "attention k, v", "attn_v": "attention k, v",
        "attn_output": "attention output", "ffn_gate_exps": "routed experts gate/up",
        "ffn_up_exps": "routed experts gate/up", "ffn_down_exps": "routed experts down",
        "ffn_gate_shexp": "shared expert gate, up", "ffn_up_shexp": "shared expert gate, up",
        "ffn_down_shexp": "shared expert down", "ffn_gate": "dense FFN", "ffn_up": "dense FFN",
        "ffn_down": "dense FFN", "output": "output head", "ffn_gate_inp": "router"}
ORDER = ["attention q", "attention k, v", "attention output", "routed experts gate/up", "routed experts down",
         "shared expert gate, up", "shared expert down", "dense FFN", "output head", "router", "everything else"]


def cls(label):
    ops, _, rest = label.partition(" ")
    if "MUL_MAT" in ops and rest:
        w = rest.split()[0].split(":")
        if len(w) == 3 and w[0] in ROLE:
            return ROLE[w[0]], w[1]
    return "everything else", ""


def main():
    ms = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(lambda: defaultdict(float))
    runs = []
    for r in csv.DictReader(open(sys.argv[1])):
        if r["run"] not in runs:
            runs.append(r["run"])
        k = cls(r["label"])
        ms[k][r["run"]] += float(r["ms_per_eval"])
        calls[k][r["run"]] += float(r["calls_per_eval"])
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["class", "type", "launches_per_eval"] + [f"ms_{x}" for x in runs])
    keys = sorted(ms, key=lambda k: (ORDER.index(k[0]), -ms[k][runs[0]]))
    for k in keys:
        w.writerow([k[0], k[1], f"{calls[k][runs[0]]:.0f}"] + [f"{ms[k][x]:.2f}" for x in runs])
    w.writerow(["total", "", ""] + [f"{sum(ms[k][x] for k in ms):.2f}" for x in runs])


if __name__ == "__main__":
    main()
