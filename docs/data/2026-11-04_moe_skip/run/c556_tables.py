#!/usr/bin/env python3
"""c556_tables.py: the tables of c556_alt.sh's run (Q1028). Per build and k: the loads' T(k) (ms per micro-batch of k
tokens), their mean, standard deviation and range; per pair of consecutive loads (1,2), (3,4), ... the second build's
T(k) minus the first's, and the mean of those differences with its 95 % interval (Student's t over the pairs).

  python3 -I c556_tables.py --requests OUT/requests.tsv --out <dir> [--a kshape] [--b moeskip]

Writes <out>/c556_loads.tsv (k, then T(k) per load), <out>/c556_tk.tsv (k, per build n, mean, sd, min, max; the
difference b - a of the means, the paired mean, its sd and 95 % interval) and prints the Markdown tables. Standard
library only."""
import argparse, csv, math, os, statistics as st

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

ap = argparse.ArgumentParser()
ap.add_argument("--requests", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--a", default="kshape"); ap.add_argument("--b", default="moeskip")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

t = {}  # load -> (build, k -> T(k))
for r in csv.DictReader(open(a.requests), delimiter="\t"):
    n = int(r["load"])
    t.setdefault(n, (r["build"], {}))[1][int(r["k"])] = float(r["ms_per_micro_batch"])
loads = sorted(t)
ks = sorted(set(k for n in loads for k in t[n][1]))

with open(os.path.join(a.out, "c556_loads.tsv"), "w") as f:
    f.write("k\t" + "\t".join(f"L{n}_{t[n][0]}" for n in loads) + "\n")
    for k in ks:
        f.write(f"{k}\t" + "\t".join(f"{t[n][1].get(k, float('nan')):.2f}" for n in loads) + "\n")
print("| k | " + " | ".join(f"{n}: `{t[n][0]}`" for n in loads) + " |")
print("|---:" * (1 + len(loads)) + "|")
for k in ks:
    print(f"| {k} | " + " | ".join(f"{t[n][1][k]:.1f}" for n in loads) + " |")
print()

def summ(xs):
    return len(xs), st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan")), min(xs), max(xs)

rows = []
for k in ks:
    xa = [t[n][1][k] for n in loads if t[n][0] == a.a and k in t[n][1]]
    xb = [t[n][1][k] for n in loads if t[n][0] == a.b and k in t[n][1]]
    d = []  # per pair of consecutive loads: b - a
    for i in range(0, len(loads) - 1, 2):
        p, q = t[loads[i]], t[loads[i + 1]]
        if {p[0], q[0]} == {a.a, a.b} and k in p[1] and k in q[1]:
            vb, va = (p[1][k], q[1][k]) if p[0] == a.b else (q[1][k], p[1][k])
            d.append(vb - va)
    sa, sb = summ(xa), summ(xb)
    dm = st.mean(d)
    dsd = st.stdev(d) if len(d) > 1 else float("nan")
    half = T975.get(len(d) - 1, float("nan")) * dsd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    rows.append((k, sa, sb, sb[1] - sa[1], d, dm, dsd, half))

with open(os.path.join(a.out, "c556_tk.tsv"), "w") as f:
    f.write("k\t" + "\t".join(f"{b}_{c}" for b in (a.a, a.b) for c in ("n", "mean", "sd", "min", "max"))
            + "\tdiff_of_means\tpairs\tpaired_mean\tpaired_sd\tci95_low\tci95_high\n")
    for k, sa, sb, dmeans, d, dm, dsd, half in rows:
        f.write(f"{k}\t" + "\t".join(f"{v:.2f}" if i % 5 else str(v) for s in (sa, sb) for i, v in enumerate(s))
                + f"\t{dmeans:.2f}\t{','.join(f'{x:.2f}' for x in d)}\t{dm:.2f}\t{dsd:.2f}\t{dm - half:.2f}\t{dm + half:.2f}\n")

print(f"| k | `{a.a}` mean ± sd (range), n | `{a.b}` mean ± sd (range), n | `{a.b}` − `{a.a}`, per pair | paired mean (95 % interval) |")
print("|---:|---:|---:|---:|---:|")
for k, sa, sb, dmeans, d, dm, dsd, half in rows:
    print(f"| {k} | {sa[1]:.1f} ± {sa[2]:.1f} ({sa[3]:.1f}–{sa[4]:.1f}), {sa[0]} | {sb[1]:.1f} ± {sb[2]:.1f} "
          f"({sb[3]:.1f}–{sb[4]:.1f}), {sb[0]} | {', '.join(f'{x:+.1f}' for x in d)} | {dm:+.2f} ({dm - half:+.2f} to "
          f"{dm + half:+.2f}) |")
