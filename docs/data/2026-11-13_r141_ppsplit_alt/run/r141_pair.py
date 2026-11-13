#!/usr/bin/env python3
"""r141_pair.py: R141's (Q1437) table, two loads of a reference rung against two loads of a test rung on one build, by
R90's size buckets, for loads run in the order ref, test, test, ref.

    python3 -I r141_pair.py --ref embd1=r141_1_embd/load-A --test pp1_2=r141_2_pp1/load-A \
        --test pp1_3=r141_3_pp1/load-A --ref embd4=r141_4_embd/load-A --out tables

Input per load folder (rig_taper.sh's load-A): requests.tsv (taper_replay.py). Only the replayed turns count (kind
"turn"), the turns every pass of every load holds; a turn is binned by its planned new tokens (n_new) in R90's buckets
(under 512, 512-1,023, 1,024-1,999, 2,000 and more) as in R123a's ladder (r123a_ladder.py).

Writes, in --out:
  pair.tsv   per bucket: turns, tokens, each load's server prompt seconds of each pass by value and its mean, the
             reference's mean and the test's (each the mean of its loads' means), the difference, the reference's
             spread (the two reference loads' means apart), the test's spread, and the class
  order.tsv  per bucket: each test load against the reference load next to it in the order (the first test against
             the first reference, the second test against the second reference), the difference
  decision.md  the classes in words
Class: gain when the test's mean is below the reference's by more than the reference's spread, loss when above by more,
negligible otherwise. Standard library only.
"""
import argparse, csv, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--ref", action="append", required=True, help="name=folder of a reference load, two of them")
ap.add_argument("--test", action="append", required=True, help="name=folder of a test load, two of them")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
assert len(a.ref) == 2 and len(a.test) == 2

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


loads = {}
ref = [s.split("=", 1)[0] for s in a.ref]
test = [s.split("=", 1)[0] for s in a.test]
for spec in a.ref + a.test:
    name, path = spec.split("=", 1)
    per = {}
    for r in csv.DictReader(open(os.path.join(path, "requests.tsv")), delimiter="\t"):
        if r["kind"] != "turn":
            continue
        per.setdefault(int(r["pass"]), {})[(r["task"], r["turn"])] = (int(r["n_new"]), float(r["prompt_ms"]))
    loads[name] = dict(sorted(per.items()))
keys = None
for ts in (t for name in loads for t in loads[name].values()):
    keys = set(ts) if keys is None else keys & set(ts)
keys = sorted(keys)
n_new = {k: next(iter(loads[ref[0]].values()))[k][0] for k in keys}


def secs(name, lo, hi):
    return [sum(ts[k][1] for k in keys if lo <= n_new[k] < hi) / 1000 for ts in loads[name].values()]


def m(name, lo, hi):
    return statistics.mean(secs(name, lo, hi))


cls = {}
with open(f"{a.out}/pair.tsv", "w") as f, open(f"{a.out}/order.tsv", "w") as g:
    f.write("new_tokens\tturns\ttokens\t" + "\t".join(f"{n}_s_by_pass\t{n}_mean_s" for n in ref + test) +
            "\tref_mean_s\ttest_mean_s\tdelta_s\tdelta_pct\tref_spread_s\ttest_spread_s\tclass\n")
    g.write("new_tokens\ttest\tagainst\ttest_mean_s\tref_mean_s\tdelta_s\n")
    for lo, hi in BUCKETS:
        kb = [k for k in keys if lo <= n_new[k] < hi]
        if not kb:
            continue
        rm = statistics.mean(m(n, lo, hi) for n in ref)
        tm = statistics.mean(m(n, lo, hi) for n in test)
        rs, ts_ = abs(m(ref[0], lo, hi) - m(ref[1], lo, hi)), abs(m(test[0], lo, hi) - m(test[1], lo, hi))
        d = tm - rm
        c = "gain" if d < -rs else "loss" if d > rs else "negligible"
        cls[bname(lo, hi)] = (c, d, d / rm * 100, rs, ts_, rm, tm)
        cols = "\t".join(f"{', '.join(f'{x:.2f}' for x in secs(n, lo, hi))}\t{m(n, lo, hi):.2f}" for n in ref + test)
        f.write(f"{bname(lo, hi)}\t{len(kb)}\t{sum(n_new[k] for k in kb)}\t{cols}\t{rm:.2f}\t{tm:.2f}\t{d:+.2f}\t"
                f"{d / rm * 100:+.2f}\t{rs:.2f}\t{ts_:.2f}\t{c}\n")
        for t, r in zip(test, ref):
            g.write(f"{bname(lo, hi)}\t{t}\t{r}\t{m(t, lo, hi):.2f}\t{m(r, lo, hi):.2f}\t"
                    f"{m(t, lo, hi) - m(r, lo, hi):+.2f}\n")

with open(f"{a.out}/decision.md", "w") as f:
    c, d, pct, rs, ts_, rm, tm = cls["all"]
    f.write(f"{len(keys)} replayed turns in every pass of every load. {' and '.join(test)} against "
            f"{' and '.join(ref)}: {tm:.2f} s a pass against {rm:.2f} ({d:+.2f} s, {pct:+.2f} %), the reference's "
            f"spread {rs:.2f}, the test's {ts_:.2f} → {c}.\n\nBy new tokens:\n")
    for b, (c, d, pct, rs, ts_, rm, tm) in cls.items():
        if b != "all":
            f.write(f"- {b}: {tm:.2f} against {rm:.2f} ({d:+.2f} s, {pct:+.2f} %), spread {rs:.2f} → {c}\n")
print(open(f"{a.out}/decision.md").read())
