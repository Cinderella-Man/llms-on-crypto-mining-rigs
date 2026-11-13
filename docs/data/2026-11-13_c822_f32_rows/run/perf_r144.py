#!/usr/bin/env python3
"""perf_r144.py: R144 (Q1448). The perf step of rig_r144.sh as tables: every case's us/run per arm and pass, by value,
per row count, width and token count the class of each comparison, and per width and token count the row counts where
mul_mat_vec_f beats cuBLAS (where GGML_CUDA_MMVF_F32_MT_ROWS's limit belongs). From R132's perf_rows.py
(2026-11-12_c807_f32_router/run/), with C822's arm auto and any width.

  python3 -I perf_r144.py us_run.tsv --out DIR > perf.md

us_run.tsv (no header): arm, pass, type, m, k, n, us/run, one line per arm, pass and test-backend-ops case, as
rig_r144.sh (and rig_c807.sh) writes it. m is the product's rows, k its width, n the tokens. The arms and the kernel
each runs at 4-8 tokens (at 1-3 tokens every arm runs mul_mat_vec_f):
  off    the switch off: cuBLAS
  on     GGML_CUDA_MMVF_F32_MT=1 at its default _ROWS=32: mul_mat_vec_f up to 32 rows, cuBLAS above
  rows0  the switch with _ROWS=0: mul_mat_vec_f at every row count
  auto   the switch with _ROWS=auto (C822): mul_mat_vec_f up to 48 rows at 4 tokens and 64 at 5-8, cuBLAS above

The comparisons, each paired pass by pass (the arms alternate within a pass), at 4-8 tokens:
  rows0 against off   mul_mat_vec_f against cuBLAS at every row count: where the limit belongs (the placement table)
  auto against off    the new value against the switch off
  auto against on     the new value against the old (they differ only where one takes the product and the other not)
  auto against rows0  the new limit itself: the same kernel where auto takes the product
  on against off      the old value against the switch off
A comparison changes a case ("kernel") when the two arms run different kernels there, else "none" (the same kernel:
negligible is expected). The spread of a shape: the largest (max - min) / median over every arm and pass at each token
count where every arm runs the same kernel (1-3 tokens). A case is a gain when the arm is faster than the reference by
more than the shape's spread in every pass, a loss when slower by more than the spread in every pass, negligible when
within the spread in every pass, mixed otherwise.

Out: DIR/wide.tsv, DIR/spread.tsv, DIR/verdict.tsv (per comparison, m, k, n >= 4), DIR/place.tsv (per k and n >= 4:
rows0 against off at each row count, the largest row count up to which every measured row count gains, and the first
that does not). Standard library only."""
import argparse, collections, os, statistics

LIMITS = {"on": {n: 32 for n in range(4, 9)}, "rows0": {n: 0 for n in range(4, 9)},
          "auto": {4: 48, 5: 64, 6: 64, 7: 64, 8: 64}}
ap = argparse.ArgumentParser()
ap.add_argument("us_run"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

us = {}  # (arm, pass, m, k, n) -> us/run
passes = set()
for line in open(a.us_run):
    f = line.rstrip("\n").split("\t")
    if len(f) != 7 or f[2] != "f32":
        continue
    arm, p, m, k, n, v = f[0], int(f[1]), int(f[3]), int(f[4]), int(f[5]), float(f[6])
    us[(arm, p, m, k, n)] = v
    passes.add(p)
passes = sorted(passes)
ARMS = [x for x in ("off", "on", "rows0", "auto") if any(q[0] == x for q in us)]
shapes = sorted({(m, k) for (_, _, m, k, _) in us}, key=lambda s: (s[1], s[0]))
widths = sorted({k for (_, k) in shapes})
ns = sorted({n for (*_, n) in us})


def kernel(arm, m, n):
    if n < 4 or n > 8:
        return "mmvf"
    if arm == "off":
        return "cublas"
    lim = LIMITS[arm][n]
    return "mmvf" if lim <= 0 or m <= lim else "cublas"


CMPS = [(x, r) for (x, r) in (("rows0", "off"), ("auto", "off"), ("auto", "on"), ("auto", "rows0"), ("on", "off"))
        if x in ARMS and r in ARMS]
cols = [f"{x}-{p}" for x in ARMS for p in passes]

with open(os.path.join(a.out, "wide.tsv"), "w") as w:
    w.write("m\tk\tn\t" + "\t".join(cols) + "\n")
    for (m, k) in shapes:
        for n in ns:
            vals = [us.get((x, p, m, k, n)) for x in ARMS for p in passes]
            if all(v is None for v in vals):
                continue
            w.write(f"{m}\t{k}\t{n}\t" + "\t".join("" if v is None else f"{v:.2f}" for v in vals) + "\n")

spread, same_ns = {}, {}
with open(os.path.join(a.out, "spread.tsv"), "w") as w:
    w.write("m\tk\tsame_kernel_n\tspread_by_n\tspread\n")
    for (m, k) in shapes:
        sn = [n for n in ns if len({kernel(x, m, n) for x in ARMS}) == 1]
        by = {}
        for n in sn:
            v = [us[(x, p, m, k, n)] for x in ARMS for p in passes if (x, p, m, k, n) in us]
            if len(v) > 1:
                by[n] = (max(v) - min(v)) / statistics.median(v)
        spread[(m, k)] = max(by.values()) if by else float("nan")
        same_ns[(m, k)] = sn
        w.write(f"{m}\t{k}\t{','.join(map(str, sn))}\t" + ",".join(f"{n}:{by[n]:.4f}" for n in by)
                + f"\t{spread[(m, k)]:.4f}\n")


def klass(ch, sp):
    if all(c < -sp for c in ch):
        return "gain"
    if all(c > sp for c in ch):
        return "loss"
    if all(abs(c) <= sp for c in ch):
        return "negligible"
    return "mixed"


rows = []
for (x, r) in CMPS:
    for (m, k) in shapes:
        for n in ns:
            if n < 4:
                continue
            ref = [us.get((r, p, m, k, n)) for p in passes]
            val = [us.get((x, p, m, k, n)) for p in passes]
            if None in ref or None in val:
                continue
            ch = [(v - q) / q for v, q in zip(val, ref)]
            sw = "none" if kernel(x, m, n) == kernel(r, m, n) else "kernel"
            rows.append(dict(cmp=f"{x}-{r}", arm=x, ref_arm=r, sw=sw, m=m, k=k, n=n, ref=ref, val=val, ch=ch,
                             spread=spread[(m, k)], klass=klass(ch, spread[(m, k)])))
with open(os.path.join(a.out, "verdict.tsv"), "w") as w:
    w.write("comparison\tchanges\tm\tk\tn\t" + "\t".join(f"ref_us_p{p}" for p in passes) + "\t"
            + "\t".join(f"arm_us_p{p}" for p in passes) + "\t" + "\t".join(f"change_p{p}" for p in passes)
            + "\tspread\tclass\n")
    for q in rows:
        w.write(f"{q['cmp']}\t{q['sw']}\t{q['m']}\t{q['k']}\t{q['n']}\t"
                + "\t".join(f"{v:.2f}" for v in q["ref"]) + "\t" + "\t".join(f"{v:.2f}" for v in q["val"]) + "\t"
                + "\t".join(f"{c:+.4f}" for c in q["ch"]) + f"\t{q['spread']:.4f}\t{q['klass']}\n")

# the placement: rows0 against off, per width and token count
place = []
for k in widths:
    for n in [n for n in ns if 4 <= n <= 8]:
        cls = sorted(((q["m"], q["klass"]) for q in rows if q["cmp"] == "rows0-off" and q["k"] == k and q["n"] == n))
        if not cls:
            continue
        upto, first = 0, None
        for m, c in cls:
            if c == "gain" and first is None:
                upto = m
            elif first is None:
                first = (m, c)
        place.append(dict(k=k, n=n, cls=cls, upto=upto, first=first))
with open(os.path.join(a.out, "place.tsv"), "w") as w:
    w.write("k\tn\tclasses\tgains_up_to\tfirst_not_gain\n")
    for q in place:
        w.write(f"{q['k']}\t{q['n']}\t" + " ".join(f"{m}:{c}" for m, c in q["cls"]) + f"\t{q['upto']}\t"
                + (f"{q['first'][0]}:{q['first'][1]}" if q["first"] else "") + "\n")

print(f"# perf: {len(us)} cases, arms {' '.join(ARMS)}, passes {' '.join(map(str, passes))}\n")
print("## Where mul_mat_vec_f beats cuBLAS (rows0 against off)\n")
print("Per width and token count: the largest row count up to which every measured row count gains in both passes, and")
print("the first row count that does not (its class). auto's limit is 48 rows at 4 tokens and 64 at 5-8.\n")
print("| width | tokens | gains up to (rows) | first not a gain | every row count's class |")
print("|---:|---:|---:|---|---|")
for q in place:
    print(f"| {q['k']:,} | {q['n']} | {q['upto']} | " + (f"{q['first'][0]} ({q['first'][1]})" if q["first"] else "none")
          + " | " + ", ".join(f"{m}: {c}" for m, c in q["cls"]) + " |")
print()
print("## The spread per shape\n")
print("| m × k | token counts where every arm runs the same kernel | spread |")
print("|---|---|---:|")
for (m, k) in shapes:
    print(f"| {m:,} × {k:,} | {', '.join(map(str, same_ns[(m, k)]))} | {spread[(m, k)] * 100:.1f} % |")
print()
for (x, r) in CMPS:
    cmp = f"{x}-{r}"
    print(f"## {x} against {r}, us/run per pass\n")
    print(f"| changes | m × k | n | {r} " + " / ".join(f"p{p}" for p in passes) + f" | {x} "
          + " / ".join(f"p{p}" for p in passes) + " | change " + " / ".join(f"p{p}" for p in passes)
          + " | spread | class |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for q in sorted((q for q in rows if q["cmp"] == cmp), key=lambda q: (q["k"], q["m"], q["n"])):
        print(f"| {q['sw']} | {q['m']:,} × {q['k']:,} | {q['n']} | " + " / ".join(f"{v:,.2f}" for v in q["ref"])
              + " | " + " / ".join(f"{v:,.2f}" for v in q["val"]) + " | " + " / ".join(f"{c * 100:+.1f} %" for c in q["ch"])
              + f" | {q['spread'] * 100:.1f} % | {q['klass']} |")
    print()
print("## Classes per comparison (4-8 tokens)\n")
print("| comparison | changes | cases | gain | negligible | loss | mixed |")
print("|---|---|---:|---:|---:|---:|---:|")
for (x, r) in CMPS:
    cmp = f"{x}-{r}"
    for sw in ("kernel", "none"):
        c = collections.Counter(q["klass"] for q in rows if q["cmp"] == cmp and q["sw"] == sw)
        if sum(c.values()):
            print(f"| {x} against {r} | {sw} | {sum(c.values())} | {c['gain']} | {c['negligible']} | {c['loss']} | {c['mixed']} |")
