#!/usr/bin/env python3
"""C899: GGML_CUDA_MMVQ_PASCAL_MT=2's rule from R197's per-product timings, and what it saves (est.).

Reads R197's compare.tsv.gz (docs/data/2026-11-19_r197_ornith_passes/), rows "label": every launch group of Ornith-1.5-122B
Pascal N2P GGUF's verify pass at depth 0, the repeated request, for arm base (the v5 preset) and arm ro (the same with
GGML_CUDA_MMVQ_PASCAL_MT=1, _KCOL_SKIP=q5_0,q8_0, _GLU_MT=1), two loads each, at 2, 3 and 4 tokens. For each plain Q5_0 or
Q8_0 product (MUL_MAT) it writes:
  - the class at each width, R197's rule: "mt" (the read-once kernel faster) when both loads' paired differences
    ro - base are below zero and each is larger than both arms' spreads between their two loads, "unset" (the kernel
    the product runs with the switch unset faster) when both are above zero and larger, "level" otherwise;
  - the kernel each arm ran (from the source, as R197 read it: base runs the k-column kernel for Q8_0 and for Q5_0 wider
    than 4,096 values in, mul_mat_vec_q_pascal_mc for the other Q5_0; ro runs mul_mat_vec_q_pascal_mt);
  - the kernel GGML_CUDA_MMVQ_PASCAL_MT=2 picks (RULES below, a copy of table 2 in the fork's
    ggml/src/ggml-cuda/mmvq-pascal-pick.h) and whether it agrees with the class.
Then the sum of the products per arm and for the rule (est.: each product's time taken from the arm whose kernel the rule
picks, the mean of that arm's two loads), against base.
Usage: python3 -I rule.py <compare.tsv.gz, or compare_p2.tsv from tables.sh> <out dir>   writes rule.md and rule.tsv
"""
import collections
import gzip
import os
import sys

# table 2 of mmvq-pascal-pick.h: (type, kmin, kmax, rmin, rmax, tokens that go to mul_mat_vec_q_pascal_mt)
BIG = 1 << 62
RULES = [
    ("q5_0", 4097, BIG, 1, BIG, {2, 3, 4}),
    ("q5_0", 2049, 4096, 1, 4096, {2, 3}),
    ("q5_0", 2049, 4096, 4097, BIG, {3}),
    ("q5_0", 1, 2048, 1, BIG, {3}),
    ("q8_0", 1, 4096, 1, BIG, {2, 3, 4}),
]


def rule_takes(typ, k, rows, ntok):
    for t, kmin, kmax, rmin, rmax, toks in RULES:
        if t == typ and kmin <= k <= kmax and rmin <= rows <= rmax:
            return ntok in toks
    return False


def unset_kernel(typ, k):
    return "kcol" if typ == "q8_0" or k > 4096 else "mc"


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = collections.defaultdict(list)
    with (gzip.open(src, "rt") if src.endswith(".gz") else open(src)) as f:
        next(f)
        for line in f:
            table, w, _dep, _req, arm, load, name, val = line.rstrip("\n").split("\t")
            if table != "label" or not name.startswith("MUL_MAT "):
                continue
            tensor, typ, shape = name[len("MUL_MAT "):].split(":")
            if typ not in ("q5_0", "q8_0"):
                continue
            k, rows = (int(x) for x in shape.split("x"))
            d[(tensor, typ, k, rows, int(w), arm)].append((load, float(val)))
    prods = sorted({key[:4] for key in d}, key=lambda p: (p[1], -p[2], -p[3], p[0]))
    rows_tsv = ["product\ttype\tin\trows\ttokens\tbase_1\tbase_2\tro_1\tro_2\tdiff_1\tdiff_2\tspread_base\tspread_ro"
                "\tclass\tkernel_unset\tkernel_mt2\tagrees\tchosen_ms"]
    sums = collections.defaultdict(float)
    md = ["| Product (values in × rows) | Type | Tokens | base, ms a pass | ro, ms a pass | ro − base, paired | "
          "Spreads (base, ro) | Faster | Unset runs | `MT=2` runs |",
          "|---|---|---:|---|---|---|---|---|---|---|"]
    bad = 0
    for tensor, typ, k, rows in prods:
        for w in (2, 3, 4):
            b = [v for _, v in sorted(d[(tensor, typ, k, rows, w, "base")])]
            r = [v for _, v in sorted(d[(tensor, typ, k, rows, w, "ro")])]
            assert len(b) == 2 and len(r) == 2, (tensor, w, b, r)
            diff = [r[i] - b[i] for i in range(2)]
            sb, sr = abs(b[0] - b[1]), abs(r[0] - r[1])
            lim = max(sb, sr)
            if all(x < 0 and -x > lim for x in diff):
                cls = "mt"
            elif all(x > 0 and x > lim for x in diff):
                cls = "unset"
            else:
                cls = "level"
            k0 = unset_kernel(typ, k)
            mt2 = rule_takes(typ, k, rows, w)
            k2 = "mt" if mt2 else k0
            agrees = cls == "level" or (cls == "mt") == mt2
            bad += not agrees
            chosen = (r[0] + r[1]) / 2 if mt2 else (b[0] + b[1]) / 2
            sums[(w, "base")] += (b[0] + b[1]) / 2
            sums[(w, "ro")] += (r[0] + r[1]) / 2
            sums[(w, "mt2")] += chosen
            sums[(w, "best")] += min((b[0] + b[1]) / 2, (r[0] + r[1]) / 2)
            rows_tsv.append("\t".join(str(x) for x in (
                tensor, typ, k, rows, w, f"{b[0]:.4f}", f"{b[1]:.4f}", f"{r[0]:.4f}", f"{r[1]:.4f}", f"{diff[0]:+.4f}",
                f"{diff[1]:+.4f}", f"{sb:.4f}", f"{sr:.4f}", cls, k0, k2, "yes" if agrees else "NO", f"{chosen:.4f}")))
            label = {"mt": "read-once", "unset": "unset", "level": "level"}[cls]
            md.append(f"| `{tensor}` ({k:,} × {rows:,}) | {typ.upper()} | {w} | {b[0]:.2f}, {b[1]:.2f} | {r[0]:.2f}, "
                      f"{r[1]:.2f} | {diff[0]:+.2f}, {diff[1]:+.2f} | {sb:.2f}, {sr:.2f} | {label} | {k0} | {k2} |")
    md += ["", "| Tokens | base: the 13 products, ms a pass | ro | `MT=2` (est.) | `MT=2` − base (est.) | "
           "the faster arm per product (est.) |", "|---:|---:|---:|---:|---:|---:|"]
    for w in (2, 3, 4):
        md.append(f"| {w} | {sums[(w, 'base')]:.2f} | {sums[(w, 'ro')]:.2f} | {sums[(w, 'mt2')]:.2f} | "
                  f"{sums[(w, 'mt2')] - sums[(w, 'base')]:+.2f} | {sums[(w, 'best')]:.2f} |")
    md.append("")
    md.append(f"Products and widths where `MT=2` disagrees with R197's class: {bad}.")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rule.tsv"), "w") as f:
        f.write("\n".join(rows_tsv) + "\n")
    with open(os.path.join(out, "rule.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
