#!/usr/bin/env python3
"""shexp_table.py: C757 (Q1333). The shared expert's gate, up and SWIGLU in a served verify pass, from
2026-11-10_served_verify_parts/run/verify_parts.py's labels.tsv of each profiled arm load (rig_c757.sh's served step),
with GGML_CUDA_MMVQ_PASCAL_GLU_MT off (the gate and the up as one-op MUL_MAT launch groups of ffn_gate_shexp and
ffn_up_shexp, the SWIGLU as its own group) and on (MUL_MAT+MUL_MAT+SWIGLU groups naming the two tensors, and whatever
layers it leaves apart).

  python3 -I shexp_table.py PARTS --out DIR > shexp.md

PARTS holds verify_parts.py's --out folder of each arm load, named <k>-<arm> (1-off, 2-on, ...), each with labels.tsv
(trace, ..., label, launch groups a pass, ms a pass) and parts.tsv. Every row of labels.tsv is one label of one trace,
summed over the card groups' rows (cards 0 to 7 and card 8).

Out: DIR/shexp.tsv (per arm load and trace: ms and launch groups a pass of the apart gate and up products, of the
SWIGLU groups, of the fused groups, their sum, and the "shared expert" part and the pass's sum of parts from parts.tsv),
DIR/fused_by_pair.tsv (per arm load and trace and gate+up type pair: the fused groups' ms and groups a pass), and a
Markdown report on stdout. Standard library only."""
import argparse, collections, os, re

ap = argparse.ArgumentParser()
ap.add_argument("parts"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = sorted((d for d in os.listdir(a.parts) if re.fullmatch(r"\d+-\w+", d) and os.path.isdir(os.path.join(a.parts, d))),
              key=lambda d: int(d.split("-")[0]))

APART = re.compile(r"^MUL_MAT ffn_(gate|up)_shexp:([A-Za-z0-9_]+):\d+x\d+$")
GLU = re.compile(r"^SWIGLU ")
FUSED = re.compile(r"^MUL_MAT\+MUL_MAT\+SWIGLU ffn_(gate|up)_shexp:([A-Za-z0-9_]+):\d+x\d+ ffn_(gate|up)_shexp:([A-Za-z0-9_]+):\d+x\d+$")
ms = collections.defaultdict(float)
grp = collections.defaultdict(float)
pair_ms = collections.defaultdict(float)
pair_grp = collections.defaultdict(float)
glu_labels = collections.Counter()
parts = {}
traces = collections.OrderedDict()
for r in runs:
    lp = os.path.join(a.parts, r, "labels.tsv")
    if not os.path.exists(lp):
        continue
    for line in open(lp):
        f = line.rstrip("\n").split("\t")
        if f[0] == "trace" or len(f) < 5:
            continue
        tr, label, g, t = f[0], f[2], float(f[3]), float(f[4])
        traces[(r, tr)] = 1
        if APART.match(label):
            kind = "apart"
        elif GLU.match(label):
            kind = "swiglu"
            glu_labels[label] += 1
        else:
            m = FUSED.match(label)
            if not m:
                continue
            kind = "fused"
            gt, ut = (m.group(2), m.group(4)) if m.group(1) == "gate" else (m.group(4), m.group(2))
            pair_ms[(r, tr, gt + "+" + ut)] += t
            pair_grp[(r, tr, gt + "+" + ut)] += g
        ms[(r, tr, kind)] += t
        grp[(r, tr, kind)] += g
    pp = os.path.join(a.parts, r, "parts.tsv")
    if os.path.exists(pp):
        for line in open(pp):
            f = line.rstrip("\n").split("\t")
            if len(f) == 5 and f[1] == "all cards":
                parts[(r, f[0], f[3])] = float(f[4])
                parts[(r, f[0], "passes")] = f[2]

KINDS = ("apart", "swiglu", "fused")
cols = ["run", "trace", "passes"] + [f"{k}_{x}" for k in KINDS for x in ("ms", "groups")] + \
    ["gate_up_swiglu_ms", "shared_expert_part_ms", "sum_of_parts_ms"]
rows = []
for (r, tr) in traces:
    x = {"run": r, "trace": tr, "passes": parts.get((r, tr, "passes"), "")}
    for k in KINDS:
        x[f"{k}_ms"] = f"{ms[(r, tr, k)]:.3f}"
        x[f"{k}_groups"] = f"{grp[(r, tr, k)]:.2f}"
    x["gate_up_swiglu_ms"] = f"{sum(ms[(r, tr, k)] for k in KINDS):.3f}"
    sp = [v for (rr, tt, p), v in parts.items() if rr == r and tt == tr and p.startswith("shared expert")]
    x["shared_expert_part_ms"] = f"{sp[0]:.3f}" if sp else ""
    x["sum_of_parts_ms"] = f"{parts[(r, tr, 'sum of the parts')]:.3f}" if (r, tr, "sum of the parts") in parts else ""
    rows.append(x)
with open(os.path.join(a.out, "shexp.tsv"), "w") as w:
    w.write("\t".join(cols) + "\n")
    for x in rows:
        w.write("\t".join(str(x[c]) for c in cols) + "\n")
pairs = sorted({p for (_, _, p) in pair_ms}, key=lambda p: -max(pair_grp[(r, tr, p)] for (r, tr) in traces))
with open(os.path.join(a.out, "fused_by_pair.tsv"), "w") as w:
    w.write("run\ttrace\tgate+up\tgroups_per_pass\tms_per_pass\tus_per_group\n")
    for (r, tr) in traces:
        for p in pairs:
            if (r, tr, p) in pair_ms:
                g = pair_grp[(r, tr, p)]
                w.write(f"{r}\t{tr}\t{p}\t{g:.2f}\t{pair_ms[(r, tr, p)]:.3f}\t{1000 * pair_ms[(r, tr, p)] / g if g else 0:.2f}\n")

print("## The shared expert's gate, up and SWIGLU per verify pass, all cards\n")
print("| " + " | ".join(cols) + " |")
print("|---|---|---:" + "|---:" * (len(cols) - 3) + "|")
for x in rows:
    print("| " + " | ".join(str(x[c]) for c in cols) + " |")
print("\n## The fused launch groups by gate+up types, ms a pass (groups a pass)\n")
print("| run | trace | " + " | ".join(pairs) + " |")
print("|---|---" + "|---:" * len(pairs) + "|")
for (r, tr) in traces:
    print(f"| {r} | {tr} | " + " | ".join(f"{pair_ms[(r, tr, p)]:.3f} ({pair_grp[(r, tr, p)]:.2f})" if (r, tr, p) in pair_ms else ""
                                          for p in pairs) + " |")
print("\nThe SWIGLU groups counted (label: traces): " + "; ".join(f"`{l}`: {n}" for l, n in glu_labels.most_common()))
