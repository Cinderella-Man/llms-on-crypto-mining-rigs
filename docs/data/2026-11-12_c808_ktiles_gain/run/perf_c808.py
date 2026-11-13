#!/usr/bin/env python3
"""perf_c808.py: R133 (C808, Q1414). rig_c808.sh's perf and alone steps as tables: every case's us/run per arm and pass,
by value, each switch's class per shape and token count, the five products at 2,560 x 6,144 the row is about, and the
alone step against perf. From R120b's perf_table.py (2026-11-12_r120b_kern_speed/run/), with C808's kernel choices.

  python3 -I perf_c808.py PERF_US_RUN --alone ALONE_US_RUN [--r120b R120B_WIDE] --out DIR > perf.md

PERF_US_RUN and ALONE_US_RUN (no header): arm, pass, type, m, k, n, us/run, one line per arm, pass and test-backend-ops
case, as rig_c808.sh writes them. m is the product's rows, k its width, n the tokens. R120B_WIDE is R120b's
perf/wide.tsv (its arms off, mt, kern and kernm2 on the base tree), for the side-by-side table.

The arms: off (the preset's k-column switches), mt (+ _DMT=1 _Q5K_MT=1), kern (mt + _Q6K_MT=1 _DMT_KT=1 _RO8=1), kernm1
(kern + _DMT_KT_M=1) and kernm2 (kern + _DMT_KT_M=2). With C808's rule, from kern_pick.cu's table at Flash-Next's widths
(../pick/flashnext_kern.tsv): the tiles take Q4_K at K = 6,144 and 4 tokens only (M = 2 in kern and kernm2, M = 1 in
kernm1); IQ4_XS at K = 6,144 and 4 tokens and Q4_K at 3 keep C722's kernel, as in mt; kernm2 makes kern's choices
exactly.

The comparisons, each paired pass by pass (the arms alternate within a pass), each case tagged with the switch that
changes its kernel:
  kern against mt       Q6K_MT for Q6_K at 2-4 tokens; DMT_KT for Q4_K at K = 6,144 and 4 tokens; RO8 for IQ4_XS, Q4_K
                        and Q5_K at 5-8 tokens except where it declines (IQ4_XS at K = 6,144 and 7-8 tokens, Q4_K and Q5_K
                        at K = 6,144 and 6-8 tokens); "none" where both arms run the same kernel (IQ4_XS at K = 6,144 and
                        4 tokens and Q4_K at 3 among them: there kern must be mt)
  kernm1 against mt     the same, the tile at M = 1
  kernm1 against kern   DMT_KT_M (M = 1 against 2) for the tile case; "none" elsewhere
  kernm2 against kern   "none" everywhere: the same choices with one more variable in the environment
  mt against off        the read-once kernels (DMT for IQ4_XS and Q4_K, Q5K_MT for Q5_K, at 2-4 tokens)
The spread of a shape: the largest (max - min) / median over every arm and pass, at each token count where every arm
runs the same kernel (1 token everywhere, and the cases no switch changes in any comparison). A case is a gain when the
arm is faster than the reference by more than the shape's spread in every pass, a loss when slower by more than the
spread in every pass, negligible when within the spread in every pass, mixed otherwise.

The alone step runs Q6_K 2,560 x 6,144 at 4 tokens, Q4_K at 5 (no tile in any arm) and Q4_K at 4 (the tile) each in a
process of its own. Its table puts each value beside the same arm and pass's perf value, and kernm2 and kernm1 against
kern in both steps, against the perf shape's spread.

Out: DIR/wide.tsv (per type, m, k, n: each arm-pass's us/run), DIR/spread.tsv, DIR/verdict.tsv (per comparison, type,
m, k, n >= 2: the switch, the reference's and the arm's us/run per pass, the change per pass, the spread, the class, the
1-token time), DIR/focus.tsv (the five products, every arm and pass, beside R120b's), DIR/alone.tsv (per case, arm and
pass: alone and perf us/run). Standard library only."""
import argparse, collections, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("us_run"); ap.add_argument("--alone"); ap.add_argument("--r120b"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def read(path):
    d, ps = {}, set()
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) != 7:
            continue
        arm, p, t, m, k, n, v = f[0], int(f[1]), f[2], int(f[3]), int(f[4]), int(f[5]), float(f[6])
        d[(arm, p, t, m, k, n)] = v
        ps.add(p)
    return d, sorted(ps)


us, passes = read(a.us_run)
ARMS = [x for x in ("off", "mt", "kern", "kernm1", "kernm2") if any(k[0] == x for k in us)]
shapes = sorted({(t, m, k) for (_, _, t, m, k, _) in us})
ns = sorted({n for (*_, n) in us})


def tile(t, k, n):
    return t == "q4_K" and k == 6144 and n == 4


def owner(cmp, t, k, n):
    """the switch that changes the case's kernel in a comparison, or 'none'"""
    if n < 2:
        return "none"
    if cmp == "mt-off":
        if n <= 4 and t in ("iq4_xs", "q4_K"):
            return "DMT"
        if n <= 4 and t == "q5_K":
            return "Q5K_MT"
        return "none"
    if cmp == "kernm2-kern":
        return "none"
    if cmp == "kernm1-kern":
        return "DMT_KT_M" if tile(t, k, n) else "none"
    # kern-mt and kernm1-mt
    if t == "q6_K":
        return "Q6K_MT" if n <= 4 else "none"
    if tile(t, k, n):
        return "DMT_KT"
    if n >= 5 and t in ("iq4_xs", "q4_K", "q5_K"):
        if k == 6144 and ((t == "iq4_xs" and n >= 7) or (t in ("q4_K", "q5_K") and n >= 6)):
            return "none"
        return "RO8"
    return "none"


CMPS = [("kern", "mt"), ("kernm1", "mt"), ("kernm1", "kern"), ("kernm2", "kern"), ("mt", "off")]
CMPS = [(x, r) for (x, r) in CMPS if x in ARMS and r in ARMS]
cols = [f"{x}-{p}" for x in ARMS for p in passes]

with open(os.path.join(a.out, "wide.tsv"), "w") as w:
    w.write("type\tm\tk\tn\t" + "\t".join(cols) + "\n")
    for (t, m, k) in shapes:
        for n in ns:
            vals = [us.get((x, p, t, m, k, n)) for x in ARMS for p in passes]
            if all(v is None for v in vals):
                continue
            w.write(f"{t}\t{m}\t{k}\t{n}\t" + "\t".join("" if v is None else f"{v:.2f}" for v in vals) + "\n")

spread, same_ns = {}, {}
with open(os.path.join(a.out, "spread.tsv"), "w") as w:
    w.write("type\tm\tk\tsame_kernel_n\tspread_by_n\tspread\n")
    for (t, m, k) in shapes:
        sn = [n for n in ns if all(owner(f"{x}-{r}", t, k, n) == "none" for (x, r) in CMPS)]
        by = {}
        for n in sn:
            v = [us[(x, p, t, m, k, n)] for x in ARMS for p in passes if (x, p, t, m, k, n) in us]
            if len(v) > 1:
                by[n] = (max(v) - min(v)) / statistics.median(v)
        spread[(t, m, k)] = max(by.values()) if by else float("nan")
        same_ns[(t, m, k)] = sn
        w.write(f"{t}\t{m}\t{k}\t{','.join(map(str, sn))}\t" + ",".join(f"{n}:{by[n]:.4f}" for n in by)
                + f"\t{spread[(t, m, k)]:.4f}\n")


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
    cmp = f"{x}-{r}"
    for (t, m, k) in shapes:
        one = [us[(y, p, t, m, k, 1)] for y in ARMS for p in passes if (y, p, t, m, k, 1) in us]
        one = statistics.median(one) if one else float("nan")
        for n in ns:
            if n < 2:
                continue
            ref = [us.get((r, p, t, m, k, n)) for p in passes]
            val = [us.get((x, p, t, m, k, n)) for p in passes]
            if None in ref or None in val:
                continue
            ch = [(v - q) / q for v, q in zip(val, ref)]
            rows.append(dict(cmp=cmp, arm=x, ref_arm=r, sw=owner(cmp, t, k, n), type=t, m=m, k=k, n=n, ref=ref, val=val,
                             ch=ch, spread=spread[(t, m, k)], klass=klass(ch, spread[(t, m, k)]), one=one))
with open(os.path.join(a.out, "verdict.tsv"), "w") as w:
    w.write("comparison\tswitch\ttype\tm\tk\tn\t" + "\t".join(f"ref_us_p{p}" for p in passes) + "\t"
            + "\t".join(f"arm_us_p{p}" for p in passes) + "\t" + "\t".join(f"change_p{p}" for p in passes)
            + "\tspread\tclass\tone_token_us\n")
    for q in rows:
        w.write(f"{q['cmp']}\t{q['sw']}\t{q['type']}\t{q['m']}\t{q['k']}\t{q['n']}\t"
                + "\t".join(f"{v:.2f}" for v in q["ref"]) + "\t" + "\t".join(f"{v:.2f}" for v in q["val"]) + "\t"
                + "\t".join(f"{c:+.4f}" for c in q["ch"]) + f"\t{q['spread']:.4f}\t{q['klass']}\t{q['one']:.2f}\n")

# the five products at 2,560 x 6,144, beside R120b's
FOCUS = [("iq4_xs", 4), ("q4_K", 3), ("q4_K", 4), ("q4_K", 5), ("q6_K", 4)]
old, old_cols = {}, []
if a.r120b:
    lines = open(a.r120b).read().splitlines()
    old_cols = lines[0].split("\t")[4:]
    for line in lines[1:]:
        f = line.split("\t")
        old[(f[0], int(f[1]), int(f[2]), int(f[3]))] = dict(zip(old_cols, f[4:]))
with open(os.path.join(a.out, "focus.tsv"), "w") as w:
    w.write("type\tm\tk\tn\t" + "\t".join(cols) + "\t" + "\t".join(f"r120b_{c}" for c in old_cols) + "\n")
    for (t, n) in FOCUS:
        vals = [us.get((x, p, t, 2560, 6144, n)) for x in ARMS for p in passes]
        o = old.get((t, 2560, 6144, n), {})
        w.write(f"{t}\t2560\t6144\t{n}\t" + "\t".join("" if v is None else f"{v:.2f}" for v in vals) + "\t"
                + "\t".join(o.get(c, "") for c in old_cols) + "\n")

alone, apasses = (read(a.alone) if a.alone else ({}, []))
AARMS = [x for x in ("kern", "kernm1", "kernm2") if any(k[0] == x for k in alone)]
if alone:
    with open(os.path.join(a.out, "alone.tsv"), "w") as w:
        w.write("type\tm\tk\tn\tarm\tpass\talone_us\tperf_us\n")
        for (t, n) in [("q6_K", 4), ("q4_K", 5), ("q4_K", 4)]:
            for x in AARMS:
                for p in apasses:
                    av, pv = alone.get((x, p, t, 2560, 6144, n)), us.get((x, p, t, 2560, 6144, n))
                    w.write(f"{t}\t2560\t6144\t{n}\t{x}\t{p}\t" + ("" if av is None else f"{av:.2f}") + "\t"
                            + ("" if pv is None else f"{pv:.2f}") + "\n")

# the report
fv = lambda v: "–" if v is None else f"{v:,.2f}"
print(f"# perf: {len(us)} cases, arms {' '.join(ARMS)}, passes {' '.join(map(str, passes))}\n")
print("## The five products at 2,560 × 6,144, µs per product, pass 1 / pass 2\n")
print("| type | n | " + " | ".join(ARMS) + (" | R120b " + " | R120b ".join(dict.fromkeys(c.rsplit('-', 1)[0] for c in old_cols)) if old_cols else "") + " |")
print("|---|---:|" + "---:|" * (len(ARMS) + len(dict.fromkeys(c.rsplit('-', 1)[0] for c in old_cols))))
for (t, n) in FOCUS:
    cells = [" / ".join(fv(us.get((x, p, t, 2560, 6144, n))) for p in passes) for x in ARMS]
    o = old.get((t, 2560, 6144, n), {})
    oc = [" / ".join(o.get(f"{x}-{p}", "–") for p in passes) for x in dict.fromkeys(c.rsplit('-', 1)[0] for c in old_cols)]
    print(f"| {t} | {n} | " + " | ".join(cells + oc) + " |")
print()
print("## The spread per shape\n")
print("| type | m × k | token counts where every arm runs the same kernel | spread |")
print("|---|---|---|---:|")
for (t, m, k) in shapes:
    print(f"| {t} | {m:,} × {k:,} | {', '.join(map(str, same_ns[(t, m, k)]))} | {spread[(t, m, k)] * 100:.1f} % |")
print()
for (x, r) in CMPS:
    cmp = f"{x}-{r}"
    print(f"## {x} against {r}, us/run per pass\n")
    print(f"| switch | type | m × k | n | {r} " + " / ".join(f"p{p}" for p in passes) + f" | {x} "
          + " / ".join(f"p{p}" for p in passes) + " | change " + " / ".join(f"p{p}" for p in passes)
          + " | spread | class | 1 token |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|---:|")
    for q in sorted((q for q in rows if q["cmp"] == cmp), key=lambda q: (q["sw"] == "none", q["sw"], q["type"], q["k"], q["m"], q["n"])):
        print(f"| {q['sw']} | {q['type']} | {q['m']:,} × {q['k']:,} | {q['n']} | " + " / ".join(f"{v:,.2f}" for v in q["ref"])
              + " | " + " / ".join(f"{v:,.2f}" for v in q["val"]) + " | " + " / ".join(f"{c * 100:+.1f} %" for c in q["ch"])
              + f" | {q['spread'] * 100:.1f} % | {q['klass']} | {q['one']:,.2f} |")
    print()
print("## Classes per comparison and switch\n")
print("| comparison | switch | cases | gain | negligible | loss | mixed |")
print("|---|---|---:|---:|---:|---:|---:|")
for (x, r) in CMPS:
    cmp = f"{x}-{r}"
    for sw in sorted({q["sw"] for q in rows if q["cmp"] == cmp}, key=lambda s: (s == "none", s)):
        c = collections.Counter(q["klass"] for q in rows if q["cmp"] == cmp and q["sw"] == sw)
        print(f"| {x} against {r} | {sw} | {sum(c.values())} | {c['gain']} | {c['negligible']} | {c['loss']} | {c['mixed']} |")
print()
if alone:
    print("## alone: each product in a process of its own, beside perf, µs per product\n")
    print("| type | n | arm | alone " + " / ".join(f"p{p}" for p in apasses) + " | perf " + " / ".join(f"p{p}" for p in apasses)
          + " | alone against kern " + " / ".join(f"p{p}" for p in apasses) + " | perf against kern "
          + " / ".join(f"p{p}" for p in apasses) + " | perf spread |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for (t, n) in [("q6_K", 4), ("q4_K", 5), ("q4_K", 4)]:
        sp = spread.get((t, 2560, 6144), float("nan"))
        for x in AARMS:
            av = [alone.get((x, p, t, 2560, 6144, n)) for p in apasses]
            pv = [us.get((x, p, t, 2560, 6144, n)) for p in apasses]
            ak = [alone.get(("kern", p, t, 2560, 6144, n)) for p in apasses]
            pk = [us.get(("kern", p, t, 2560, 6144, n)) for p in apasses]
            rel = lambda v, r: "–" if v is None or r is None else f"{(v - r) / r * 100:+.1f} %"
            print(f"| {t} | {n} | {x} | " + " / ".join(fv(v) for v in av) + " | " + " / ".join(fv(v) for v in pv) + " | "
                  + ("" if x == "kern" else " / ".join(rel(v, r) for v, r in zip(av, ak))) + " | "
                  + ("" if x == "kern" else " / ".join(rel(v, r) for v, r in zip(pv, pk))) + f" | {sp * 100:.1f} % |")
