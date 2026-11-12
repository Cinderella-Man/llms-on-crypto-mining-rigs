#!/usr/bin/env python3
"""perf_rows.py: R132 (Q1413). The perf step of rig_c807.sh as tables: every case's us/run per arm and pass, by value,
and per row count and token count the class of each comparison. From R120b's perf_table.py
(2026-11-12_r120b_kern_speed/run/), for C807's arms and F32 row counts.

  python3 -I perf_rows.py us_run.tsv --out DIR > perf.md

us_run.tsv (no header): arm, pass, type, m, k, n, us/run, one line per arm, pass and test-backend-ops case, as
rig_c807.sh writes it. m is the product's rows, k its width (2,560), n the tokens. The arms: off (cuBLAS at 4-8 tokens),
on (GGML_CUDA_MMVF_F32_MT=1: mul_mat_vec_f at 4-8 tokens up to _ROWS = 32 rows, cuBLAS above), rows0 (the switch with
_ROWS=0: mul_mat_vec_f at 4-8 tokens at every row count). At 1-3 tokens every arm runs mul_mat_vec_f.

The comparisons, each paired pass by pass (the arms alternate within a pass), at 4-8 tokens:
  rows0 against off  mul_mat_vec_f against cuBLAS at every row count: where the limit belongs
  on against off     the switch at its default: mul_mat_vec_f up to 32 rows, the same kernel as off above
  on against rows0   the limit itself: the same kernel up to 32 rows, cuBLAS against mul_mat_vec_f above
The kernel a case runs in each arm decides whether a comparison changes it ("F32_MT" or "limit") or not ("none").
The spread of a shape: the largest (max - min) / median over every arm and pass, at each token count where every arm
runs the same kernel (1-3 tokens). A case is a gain when the arm is faster than the reference by more than the shape's
spread in every pass, a loss when slower by more than the spread in every pass, negligible when within the spread in
every pass, mixed otherwise.

Out: DIR/wide.tsv (per m, k, n: each arm-pass's us/run), DIR/spread.tsv (per m, k: the token counts used and the
spread), DIR/verdict.tsv (per comparison, m, k, n >= 4: what changes, the reference's and the arm's us/run per pass, the
change per pass, the spread, the class). Standard library only."""
import argparse, collections, os, statistics

ROWS = 32  # the default limit
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
ARMS = [x for x in ("off", "on", "rows0") if any(q[0] == x for q in us)]
shapes = sorted({(m, k) for (_, _, m, k, _) in us})
ns = sorted({n for (*_, n) in us})


def kernel(arm, m, n):
    if n < 4 or arm == "rows0" or (arm == "on" and m <= ROWS):
        return "mmvf"
    return "cublas"


def owner(x, r, m, n):
    if kernel(x, m, n) == kernel(r, m, n):
        return "none"
    return "limit" if (x, r) == ("on", "rows0") else "F32_MT"


CMPS = [(x, r) for (x, r) in (("rows0", "off"), ("on", "off"), ("on", "rows0")) if x in ARMS and r in ARMS]
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
            rows.append(dict(cmp=f"{x}-{r}", arm=x, ref_arm=r, sw=owner(x, r, m, n), m=m, k=k, n=n, ref=ref, val=val,
                             ch=ch, spread=spread[(m, k)], klass=klass(ch, spread[(m, k)])))
with open(os.path.join(a.out, "verdict.tsv"), "w") as w:
    w.write("comparison\tchanges\tm\tk\tn\t" + "\t".join(f"ref_us_p{p}" for p in passes) + "\t"
            + "\t".join(f"arm_us_p{p}" for p in passes) + "\t" + "\t".join(f"change_p{p}" for p in passes)
            + "\tspread\tclass\n")
    for q in rows:
        w.write(f"{q['cmp']}\t{q['sw']}\t{q['m']}\t{q['k']}\t{q['n']}\t"
                + "\t".join(f"{v:.2f}" for v in q["ref"]) + "\t" + "\t".join(f"{v:.2f}" for v in q["val"]) + "\t"
                + "\t".join(f"{c:+.4f}" for c in q["ch"]) + f"\t{q['spread']:.4f}\t{q['klass']}\n")

print(f"# perf: {len(us)} cases, arms {' '.join(ARMS)}, passes {' '.join(map(str, passes))}\n")
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
    for q in sorted((q for q in rows if q["cmp"] == cmp), key=lambda q: (q["m"], q["n"])):
        print(f"| {q['sw']} | {q['m']:,} × {q['k']:,} | {q['n']} | " + " / ".join(f"{v:,.2f}" for v in q["ref"])
              + " | " + " / ".join(f"{v:,.2f}" for v in q["val"]) + " | " + " / ".join(f"{c * 100:+.1f} %" for c in q["ch"])
              + f" | {q['spread'] * 100:.1f} % | {q['klass']} |")
    print()
print("## Classes per comparison and row count (4-8 tokens)\n")
print("| comparison | changes | rows | cases | gain | negligible | loss | mixed |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for (x, r) in CMPS:
    cmp = f"{x}-{r}"
    for (sw, m) in sorted({(q["sw"], q["m"]) for q in rows if q["cmp"] == cmp}, key=lambda s: (s[0] == "none", s[1])):
        c = collections.Counter(q["klass"] for q in rows if q["cmp"] == cmp and q["sw"] == sw and q["m"] == m)
        print(f"| {x} against {r} | {sw} | {m:,} | {sum(c.values())} | {c['gain']} | {c['negligible']} | {c['loss']} | {c['mixed']} |")
