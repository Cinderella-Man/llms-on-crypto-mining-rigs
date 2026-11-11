#!/usr/bin/env python3
"""perf_table.py: R79 (Q1277). The perf step of 2026-11-11_c722_short_batch_products/run/rig_c722.sh as tables: every
case's us/run per arm and pass, by value, and each arm's class at 2-4 tokens against the reference arm.

  python3 -I perf_table.py us_run.tsv --out DIR [--ref off] > perf.md

us_run.tsv (no header): arm, pass, type, m, k, n, us/run, one line per arm, pass and test-backend-ops case, as rig_c722.sh
writes it. m is the product's rows, k its width, n the tokens.

The spread of a shape: its 1-token and 5-token cases run the same kernels in every arm (the one-token kernel, and the
k-column kernel), so the largest (max - min) / median over every arm and pass at n = 1 and at n = 5 is how far two runs
of one kernel land apart. An arm's case at 2-4 tokens, paired pass by pass with the reference arm's (the arms
alternate within a pass): a gain when faster than the reference by more than the shape's spread in every pass, a loss
when slower by more than the spread in every pass, negligible when within the spread in every pass, mixed otherwise.

Out: DIR/wide.tsv (per type, m, k, n: each arm-pass's us/run), DIR/verdict.tsv (per arm, type, m, k, n at 2-4: the
reference's and the arm's us/run per pass, the change per pass, the spread, the class, the arm against the shape's
1-token time), DIR/spread.tsv (per type, m, k: the spread at n = 1 and n = 5), DIR/arms4.tsv (per arm: us/run summed
over the 4-token cases per pass, the sum rig_c722.sh's served step ranks the arms by). Standard library only."""
import argparse, collections, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("us_run"); ap.add_argument("--out", required=True); ap.add_argument("--ref", default="off")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

us = {}  # (arm, pass, type, m, k, n) -> us/run
arms, passes = [], []
for line in open(a.us_run):
    f = line.rstrip("\n").split("\t")
    if len(f) != 7:
        continue
    arm, p, t, m, k, n, v = f[0], int(f[1]), f[2], int(f[3]), int(f[4]), int(f[5]), float(f[6])
    us[(arm, p, t, m, k, n)] = v
    if arm not in arms:
        arms.append(arm)
    if p not in passes:
        passes.append(p)
passes.sort()
order = [a.ref] + [x for x in ("on", "nopf", "r1", "r2", "r1nopf", "nw2") if x in arms and x != a.ref]
order += [x for x in arms if x not in order]
shapes = sorted({(t, m, k) for (_, _, t, m, k, _) in us})
ns = sorted({n for (*_, n) in us})
cols = [f"{x}-{p}" for x in order for p in passes]

with open(os.path.join(a.out, "wide.tsv"), "w") as w:
    w.write("type\tm\tk\tn\t" + "\t".join(cols) + "\n")
    for (t, m, k) in shapes:
        for n in ns:
            vals = [us.get((x, p, t, m, k, n)) for x in order for p in passes]
            if all(v is None for v in vals):
                continue
            w.write(f"{t}\t{m}\t{k}\t{n}\t" + "\t".join("" if v is None else f"{v:.2f}" for v in vals) + "\n")

spread = {}
with open(os.path.join(a.out, "spread.tsv"), "w") as w:
    w.write("type\tm\tk\tspread_n1\tspread_n5\tspread\n")
    for (t, m, k) in shapes:
        s = {}
        for n in (1, 5):
            v = [us[(x, p, t, m, k, n)] for x in order for p in passes if (x, p, t, m, k, n) in us]
            s[n] = (max(v) - min(v)) / statistics.median(v) if len(v) > 1 else float("nan")
        spread[(t, m, k)] = max(x for x in s.values() if x == x) if any(x == x for x in s.values()) else float("nan")
        w.write(f"{t}\t{m}\t{k}\t{s[1]:.4f}\t{s[5]:.4f}\t{spread[(t, m, k)]:.4f}\n")


def klass(ch, sp):
    if all(c < -sp for c in ch):
        return "gain"
    if all(c > sp for c in ch):
        return "loss"
    if all(abs(c) <= sp for c in ch):
        return "negligible"
    return "mixed"


rows = []
for x in order:
    if x == a.ref:
        continue
    for (t, m, k) in shapes:
        one = [us[(y, p, t, m, k, 1)] for y in order for p in passes if (y, p, t, m, k, 1) in us]
        one = statistics.median(one) if one else float("nan")
        for n in (2, 3, 4):
            ref = [us.get((a.ref, p, t, m, k, n)) for p in passes]
            arm = [us.get((x, p, t, m, k, n)) for p in passes]
            if None in ref or None in arm:
                continue
            ch = [(v - r) / r for v, r in zip(arm, ref)]
            rows.append(dict(arm=x, type=t, m=m, k=k, n=n, ref=ref, val=arm, ch=ch, spread=spread[(t, m, k)],
                             klass=klass(ch, spread[(t, m, k)]), one=one))
with open(os.path.join(a.out, "verdict.tsv"), "w") as w:
    w.write("arm\ttype\tm\tk\tn\t" + "\t".join(f"{a.ref}_us_p{p}" for p in passes) + "\t"
            + "\t".join(f"arm_us_p{p}" for p in passes) + "\t" + "\t".join(f"change_p{p}" for p in passes)
            + "\tspread\tclass\tone_token_us\tarm_over_one_token_p1\n")
    for r in rows:
        w.write(f"{r['arm']}\t{r['type']}\t{r['m']}\t{r['k']}\t{r['n']}\t"
                + "\t".join(f"{v:.2f}" for v in r["ref"]) + "\t" + "\t".join(f"{v:.2f}" for v in r["val"]) + "\t"
                + "\t".join(f"{c:+.4f}" for c in r["ch"]) + f"\t{r['spread']:.4f}\t{r['klass']}\t{r['one']:.2f}"
                + f"\t{r['val'][0] / r['one']:.3f}\n")

sums = collections.defaultdict(float)
for (x, p, t, m, k, n), v in us.items():
    if n == 4:
        sums[(x, p)] += v
with open(os.path.join(a.out, "arms4.tsv"), "w") as w:
    w.write("arm\t" + "\t".join(f"us_4tok_p{p}" for p in passes) + "\tus_4tok_both\n")
    for x in order:
        w.write(f"{x}\t" + "\t".join(f"{sums[(x, p)]:.2f}" for p in passes) + f"\t{sum(sums[(x, p)] for p in passes):.2f}\n")

# the report
print(f"# perf: {len(us)} cases, arms {' '.join(order)}, passes {' '.join(map(str, passes))}\n")
print("## Each shape at 2-4 tokens, " + a.ref + " against each arm, us/run per pass\n")
for x in order:
    if x == a.ref:
        continue
    print(f"### {x}\n")
    print(f"| type | m × k | n | {a.ref} " + " / ".join(f"p{p}" for p in passes) + f" | {x} " + " / ".join(f"p{p}" for p in passes)
          + " | change " + " / ".join(f"p{p}" for p in passes) + " | spread | class | 1 token | " + x + " p1 / 1 token |")
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in rows:
        if r["arm"] != x:
            continue
        print(f"| {r['type']} | {r['m']:,} × {r['k']:,} | {r['n']} | " + " / ".join(f"{v:,.2f}" for v in r["ref"]) + " | "
              + " / ".join(f"{v:,.2f}" for v in r["val"]) + " | " + " / ".join(f"{c * 100:+.1f} %" for c in r["ch"])
              + f" | {r['spread'] * 100:.1f} % | {r['klass']} | {r['one']:,.2f} | {r['val'][0] / r['one']:.2f} |")
    print()
print("## Classes per arm\n")
print("| arm | gain | negligible | loss | mixed |")
print("|---|---:|---:|---:|---:|")
for x in order:
    if x == a.ref:
        continue
    c = collections.Counter(r["klass"] for r in rows if r["arm"] == x)
    print(f"| {x} | {c['gain']} | {c['negligible']} | {c['loss']} | {c['mixed']} |")
print("\n## The fastest arm per shape at 2-4 tokens (us/run summed over the passes), against r1 and the 1-token time\n")
print("| type | m × k | n | " + a.ref + " " + " / ".join(f"p{p}" for p in passes) + " | fastest | its " + " / ".join(f"p{p}" for p in passes)
      + " | r1 " + " / ".join(f"p{p}" for p in passes) + " | fastest / 1 token |")
print("|---|---|---:|---:|---|---:|---:|---:|")
with open(os.path.join(a.out, "fastest.tsv"), "w") as w:
    w.write("type\tm\tk\tn\tfastest\t" + "\t".join(f"us_p{p}" for p in passes) + "\t"
            + "\t".join(f"r1_us_p{p}" for p in passes) + "\tone_token_us\n")
    for (t, m, k) in shapes:
        one = statistics.median([us[(y, p, t, m, k, 1)] for y in order for p in passes if (y, p, t, m, k, 1) in us])
        for n in (2, 3, 4):
            cand = [(sum(us[(x, p, t, m, k, n)] for p in passes), x) for x in order
                    if all((x, p, t, m, k, n) in us for p in passes)]
            if not cand:
                continue
            best = min(cand)[1]
            b = [us[(best, p, t, m, k, n)] for p in passes]
            r1 = [us.get(("r1", p, t, m, k, n), float("nan")) for p in passes]
            ref = [us[(a.ref, p, t, m, k, n)] for p in passes]
            w.write(f"{t}\t{m}\t{k}\t{n}\t{best}\t" + "\t".join(f"{v:.2f}" for v in b) + "\t"
                    + "\t".join(f"{v:.2f}" for v in r1) + f"\t{one:.2f}\n")
            print(f"| {t} | {m:,} × {k:,} | {n} | " + " / ".join(f"{v:,.2f}" for v in ref) + f" | {best} | "
                  + " / ".join(f"{v:,.2f}" for v in b) + " | " + " / ".join(f"{v:,.2f}" for v in r1)
                  + f" | {sum(b) / len(b) / one:.2f} |")
print("\n## Each arm's 4-token cases summed, us/run\n")
print("| arm | " + " | ".join(f"pass {p}" for p in passes) + " | both |")
print("|---|" + "---:|" * (len(passes) + 1))
for x in order:
    print(f"| {x} | " + " | ".join(f"{sums[(x, p)]:,.2f}" for p in passes) + f" | {sum(sums[(x, p)] for p in passes):,.2f} |")
