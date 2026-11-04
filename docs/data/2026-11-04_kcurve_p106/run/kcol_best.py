#!/usr/bin/env python3
"""kcol_best.py: from bench/mmvq_pascal's mmvq_kcol CSV (--csv), per shape, type and k, the default launch shape
(`pascal`) against the fastest other variant that passed its check, and what the difference is worth per token
(calls_per_token x us). Prints a TSV of every case where another variant is at least MIN_PCT faster, then the sums per
k (ms per token of the shapes run, default against best).
  python3 -I kcol_best.py kcol.csv [MIN_PCT (5)]"""
import collections, csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
cases = collections.defaultdict(dict)
calls = {}
for r in rows:
    if r["check"] != "PASS" or not r["us_median"]:
        continue
    key = (r["shape"], r["type"], int(r["tokens"]))
    cases[key][r["variant"]] = float(r["us_median"])
    calls[key] = float(r["calls_per_token"])
print("shape\ttype\tk\tcalls_per_token\tpascal_us\tbest\tbest_us\tfaster_pct\tms_per_token_saved")
tot = collections.Counter()
for key in sorted(cases, key=lambda k: (k[2], k[0], k[1])):
    v = cases[key]
    if "pascal" not in v:
        continue
    others = {n: us for n, us in v.items() if n not in ("pascal", "prod")}
    best = min(others, key=others.get) if others else "pascal"
    b = min(others[best], v["pascal"]) if others else v["pascal"]
    tot[(key[2], "pascal")] += calls[key] * v["pascal"] / 1000
    tot[(key[2], "best")] += calls[key] * b / 1000
    pct = 100 * (v["pascal"] - b) / v["pascal"]
    if pct >= min_pct:
        print(f"{key[0]}\t{key[1]}\t{key[2]}\t{calls[key]:g}\t{v['pascal']:.1f}\t{best}\t{b:.1f}\t{pct:.0f}\t{calls[key] * (v['pascal'] - b) / 1000:.2f}")
print("\nk\tpascal_ms_per_token\tbest_ms_per_token\tsaved_ms")
for k in sorted({k for k, _ in tot}):
    print(f"{k}\t{tot[(k, 'pascal')]:.2f}\t{tot[(k, 'best')]:.2f}\t{tot[(k, 'pascal')] - tot[(k, 'best')]:.2f}")
