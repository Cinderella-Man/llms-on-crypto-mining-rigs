#!/usr/bin/env python3
"""limits.py: what limits C1146's routed gate/up kernels on a P106 (Q1871).

  python3 -I limits.py types.tsv [--request d0p2] [--tsv limits.tsv] > limits.md

Reads types.py's long table (ms a pass and launch groups a pass per type and setting) and sets each kernel's time a
layer beside two things it could be limited by:
  - issue: the static SASS instructions per 32 weights of one matrix in one pass of the inner loop at one member (the
    PTX notes' summary, docs/research/notes/ptx_kernels.md; at four tokens almost every expert has one member), and
    the time a layer per instruction against Q2_0's kernel (1.00 = Q2_0's issue rate);
  - memory: the bytes a layer read at est. 38.84 distinct experts (C827: 4 tokens x 10 experts of 512 picked at
    random; real picks overlap more, so fewer bytes) x 2 matrices x 2560 x 640 weights x the type's bits a weight, over
    the time, against the card's ~173 GB/s measured (docs/mining_rig.md).
Standard library only.
"""
import argparse
import csv
import sys

# static SASS per 32 weights, one member (ptx_kernels.md, "The four kernels: summary"); Q2_0's kernel has one version
COUNTS = {
    "iq2_xxs": {"unset": 131.44, "c": 76.94, "ptx": 72.50},
    "iq2_s": {"unset": 136.31, "c": 70.25, "ptx": 62.94},
    "iq2_xs": {"unset": 142.69, "c": 81.62, "ptx": 73.00},
    "iq1_m": {"unset": 76.69, "c": 52.50, "ptx": 45.69},
    "q2_0": {"unset": 42.88, "c": 42.88, "ptx": 42.88},
}
BPW = {"iq2_xxs": 2.0625, "iq2_s": 2.5625, "iq2_xs": 2.3125, "iq1_m": 1.75, "q2_0": 2.25}
EXPERTS = 38.84
WEIGHTS = 2 * 2560 * 640
GBS = 173.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("types_tsv")
    ap.add_argument("--request", default="d0p2")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    t = {}
    with open(a.types_tsv, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["table"] == "per_layer" and r["request"] == a.request and r["arm"] in ("unset", "c", "ptx"):
                g = float(r["groups_per_pass"])
                t[(r["key"], r["arm"])] = float(r["ms"]) / g if g else None
    P = print
    P(f"# What limits the routed gate/up kernels (request {a.request})\n")
    P("ms a layer from the group profile; instructions: static SASS per 32 weights at one member; per instruction:"
      " ms a layer / instructions, against Q2_0's (1.00 = Q2_0's rate); GB/s: est. bytes a layer at 38.84 distinct"
      " experts over the time, against the card's ~173 GB/s measured.\n")
    P("| Type | bits a weight | est. MB a layer | setting | ms a layer | instructions | per instruction, Q2_0 = 1 | est. GB/s | of 173 |")
    P("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    out = []
    q = t[("q2_0", "unset")] / COUNTS["q2_0"]["unset"]
    for typ in COUNTS:
        mb = EXPERTS * WEIGHTS * BPW[typ] / 8 / 1e6
        for arm in ("unset", "c", "ptx"):
            if typ == "q2_0" and arm != "unset":
                continue
            ms = t.get((typ, arm))
            if ms is None:
                continue
            n = COUNTS[typ][arm]
            rel = ms / n / q
            gbs = mb / 1e3 / (ms / 1e3)
            name = {"unset": "old (unset)", "c": "CUDA C (1)", "ptx": "PTX (2)"}[arm]
            P(f"| {typ.upper()} | {BPW[typ]} | {mb:.2f} | {name} | {ms:.3f} | {n:.2f} | {rel:.2f} | {gbs:.1f} | {gbs / GBS * 100:.0f} % |")
            out.append((typ, arm, f"{ms:.4f}", f"{n:.2f}", f"{rel:.3f}", f"{mb:.3f}", f"{gbs:.2f}"))
    P("\n## Over Q2_0's time, with the PTX kernels\n")
    P("| Type | layers | ms a layer over Q2_0's | ms a pass over Q2_0's |")
    P("|---|---:|---:|---:|")
    layers = {"iq2_xxs": 11, "iq2_s": 7, "iq2_xs": 4, "iq1_m": 2}
    tot = 0.0
    for typ, n in layers.items():
        d = t[(typ, "ptx")] - t[("q2_0", "unset")]
        tot += d * n
        P(f"| {typ.upper()} | {n} | {d:+.3f} | {d * n:+.3f} |")
    P(f"| **the four types** | 24 | | **{tot:+.3f}** |")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("type\tsetting\tms_per_layer\tinstructions_per_32\tper_instruction_vs_q2_0\test_mb_per_layer\test_gb_s\n")
            for r in out:
                f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
