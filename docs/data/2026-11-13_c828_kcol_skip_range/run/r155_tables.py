#!/usr/bin/env python3
"""r155_tables.py: R155 (Q1470), C828 on a P106. The perf and bitid tables of rig_c755.sh's output on v4-rc2.

Usage:
  python3 -I r155_tables.py --run <rig_c755.sh OUT> --r150 <C755's rig_check/perf.tsv> --out <dir> > r155.md

--run   rig_c755.sh's OUT with STEPS="bitid perf": perf/us_run.tsv (arm, pass, type, m, k, n, us a call), bitid/<arm>.txt
        and status.txt (the exit of each bitid arm)
--r150  R150's perf.tsv (the same steps on a pascal-p106-v4 build without C828), for its kcol and mc columns
--out   writes perf.tsv, perf_pass.tsv and bitid.tsv

perf: per type and token count, each arm's us a call in each pass, mc - kcol, est. ms a pass (16 Q4_0 and 7 Q5_0
shared-expert downs in Flash-Next) and a class by R150's rule (spread = the larger range of the two arms over the
passes: |difference| within it negligible, else gain or loss when the arms do not overlap, else mixed); the kernel each
arm runs by C828's routing; R150's kcol and mc for comparison.
bitid: per arm, input and type, comparisons, differing comparisons, values differing, values compared.
"""
import argparse
import collections
import os
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--r150", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LAYERS = {"q4_0": 16, "q5_0": 7}


def kernel(arm, n):
    if n == 1:
        return "one-token Pascal"
    if arm == "mc" and n <= 4:
        return "mul_mat_vec_q_pascal_mc"
    return "k-column"


def klass(on, off):
    sp = max(max(on) - min(on), max(off) - min(off))
    d = statistics.mean(on) - statistics.mean(off)
    if abs(d) <= sp:
        return "negligible", d, sp
    if max(on) < min(off):
        return "gain", d, sp
    if min(on) > max(off):
        return "loss", d, sp
    return "mixed", d, sp


us = {}  # (arm, pass, type, n) -> us a call
for line in open(os.path.join(a.run, "perf", "us_run.tsv")):
    f = line.rstrip("\n").split("\t")
    if len(f) == 7:
        us[(f[0], int(f[1]), f[2], int(f[5]))] = float(f[6])
r150 = {}  # (type, n) -> {column: value}
with open(a.r150) as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        f = dict(zip(hdr, line.rstrip("\n").split("\t")))
        r150[(f["type"], int(f["n"]))] = f
passes = sorted({p for _, p, _, _ in us})
types = [t for t in ("q4_0", "q5_0") if any(k[2] == t for k in us)]
ns = sorted({n for _, _, _, n in us})

print("## perf: us a call, m = 2,560, k = 640, on v4-rc2\n")
hdr = ["type", "n", "kernel_kcol", "kernel_mc"] + [f"{arm}_p{p}" for p in passes for arm in ("kcol", "mc")] \
    + [f"mc-kcol_p{p}" for p in passes] + [f"est_ms_a_pass_p{p}" for p in passes] \
    + ["class", "mean_diff", "spread"] + [f"r150_{arm}_p{p}" for p in passes for arm in ("kcol", "mc")]
print("| " + " | ".join(h.replace("_", " ") for h in hdr) + " |\n|" + "---|" * 4 + "---:|" * (len(hdr) - 4))
tot = collections.defaultdict(float)  # (n, pass) -> est ms a pass, both types
with open(os.path.join(a.out, "perf.tsv"), "w") as w:
    w.write("\t".join(hdr) + "\n")
    for t in types:
        for n in ns:
            v = [us[(arm, p, t, n)] for p in passes for arm in ("kcol", "mc")]
            d = [us[("mc", p, t, n)] - us[("kcol", p, t, n)] for p in passes]
            e = [x * LAYERS[t] / 1000 for x in d]
            for p, x in zip(passes, e):
                tot[(n, p)] += x
            c, md, sp = klass([us[("mc", p, t, n)] for p in passes], [us[("kcol", p, t, n)] for p in passes])
            old = r150.get((t, n), {})
            row = [t, str(n), kernel("kcol", n), kernel("mc", n)] + [f"{x:.2f}" for x in v] \
                + [f"{x:+.2f}" for x in d] + [f"{x:+.3f}" for x in e] + [c, f"{md:+.3f}", f"{sp:.2f}"] \
                + [old.get(f"{arm}_p{p}", "") for p in passes for arm in ("kcol", "mc")]
            w.write("\t".join(row) + "\n")
            print("| " + " | ".join(row) + " |")
print("\nBoth types, est. ms a pass (mc - kcol; 16 Q4_0 + 7 Q5_0 products), R155 and R150:\n")
print("| n | " + " | ".join(f"R155 pass {p}" for p in passes) + " | "
      + " | ".join(f"R150 pass {p}" for p in passes) + " |\n|---:|" + "---:|" * (2 * len(passes)))
with open(os.path.join(a.out, "perf_pass.tsv"), "w") as w:
    w.write("n\t" + "\t".join(f"est_ms_pass{p}" for p in passes) + "\t"
            + "\t".join(f"r150_est_ms_pass{p}" for p in passes) + "\n")
    for n in ns:
        old = [sum(float(r150[(t, n)][f"est_ms_a_pass_p{p}"]) for t in types if (t, n) in r150) for p in passes]
        row = [f"{tot[(n, p)]:+.3f}" for p in passes] + [f"{x:+.3f}" for x in old]
        w.write(f"{n}\t" + "\t".join(row) + "\n")
        print(f"| {n} | " + " | ".join(row) + " |")

print("\n## bitid: shdown, 2-4 tokens against the one-token products\n")
status = open(os.path.join(a.run, "status.txt")).read()
LINE = re.compile(r"^(MUL_MAT\S*)\s+(q\d_\d)\s+n=(\d+) m=(\d+)\s+k=(\d+)\s+: (bit-identical|DIFFERENT) \((\d+) of (\d+) values")
HOOK = re.compile(r"^(q\d_\d): the k-column hook (\w+) this type")
print("| arm | exit | hook | input | type | comparisons | differing | values differing | values |\n"
      "|---|---:|---|---|---|---:|---:|---:|---:|")
with open(os.path.join(a.out, "bitid.tsv"), "w") as w:
    w.write("arm\texit\thook\tinput\ttype\tcomparisons\tdiffering\tvalues_differing\tvalues\n")
    for arm in ("on", "table", "off", "mc0"):
        m = re.search(rf"bitid {arm} exit (\d+)", status)
        rc = m.group(1) if m else "?"
        hook, agg = {}, collections.OrderedDict()
        for line in open(os.path.join(a.run, "bitid", f"{arm}.txt")):
            h = HOOK.match(line)
            if h:
                hook[h.group(1)] = h.group(2)
            m = LINE.match(line)
            if m:
                k = (m.group(1), m.group(2))
                s = agg.setdefault(k, [0, 0, 0, 0])
                s[0] += 1
                s[1] += m.group(6) == "DIFFERENT"
                s[2] += int(m.group(7))
                s[3] += int(m.group(8))
        for (inp, t), s in agg.items():
            row = [arm, rc, hook.get(t, ""), inp, t] + [str(x) for x in s]
            w.write("\t".join(row) + "\n")
            print("| " + " | ".join(row) + " |")
