#!/usr/bin/env python3
"""decode_at_mix.py MIX=TURNS.csv ... -- NAME=TURNS.csv ...: decode speed at a fixed mix of contexts.

The same buckets and the same middle context as docs/data/2026-11-18_c854b_p106_check/run/decode_by_context.py (the
context a request started at plus half the tokens it generated), but the mix comes only from the runs before "--":
each bucket is weighted by the tokens those runs generated there. So a new run is read at the mix an earlier table
used, and adding it does not move that table's numbers. Every run, before and after "--", is printed at that mix, over
the buckets all of them decoded in; when that is every bucket the mix runs share, their rows equal the earlier
table's. Ornith has no draft model, so a decode step is one token. Standard library only:
  python3 -I decode_at_mix.py o4c2=a.csv o4c3=b.csv -- o5oc0=c.csv
"""
import csv
import sys

BUCKETS = [(0, 3000), (3000, 4500), (4500, 6000), (6000, 8000), (8000, 10 ** 9)]


def label(lo, hi):
    return f"under {hi:,}" if lo == 0 else f"{lo:,} and over" if hi == 10 ** 9 else f"{lo:,} to {hi:,}"


def read(path):
    b = [[0, 0, 0.0] for _ in BUCKETS]  # requests, tokens, decode ms
    for r in csv.DictReader(open(path)):
        if r["attempt"] != "1" or not r["prompt_n"] or not r["predicted_n"] or int(r["predicted_n"]) == 0:
            continue
        n = int(r["predicted_n"])
        mid = int(r["cache_n"]) + int(r["prompt_n"]) + n / 2
        i = next(j for j, (lo, hi) in enumerate(BUCKETS) if lo <= mid < hi)
        b[i][0] += 1
        b[i][1] += n
        b[i][2] += float(r["predicted_ms"])
    return b


args = sys.argv[1:]
if "--" not in args:
    sys.exit(__doc__)
k = args.index("--")
mix = [(a.split("=", 1)[0], read(a.split("=", 1)[1])) for a in args[:k]]
new = [(a.split("=", 1)[0], read(a.split("=", 1)[1])) for a in args[k + 1:]]
if not mix or not new:
    sys.exit(__doc__)

print("| run | middle context, tokens | requests | tokens generated | decode s | tok/s | ms a token |")
print("|---|---|---:|---:|---:|---:|---:|")
for name, b in new:
    for (lo, hi), (q, n, ms) in zip(BUCKETS, b):
        if q:
            print(f"| {name} | {label(lo, hi)} | {q} | {n} | {ms / 1000:.1f} | {n / ms * 1000:.2f} | {ms / n:.2f} |")
    q, n, ms = (sum(x[i] for x in b) for i in range(3))
    print(f"| {name} | **all** | {q} | {n} | {ms / 1000:.1f} | **{n / ms * 1000:.2f}** | {ms / n:.2f} |")

shared = [i for i in range(len(BUCKETS)) if all(b[i][1] for _, b in mix + new)]
mixshared = [i for i in range(len(BUCKETS)) if all(b[i][1] for _, b in mix)]
w = {i: sum(b[i][1] for _, b in mix) for i in shared}
tot = sum(w.values())
print()
print(f"At the mix of {', '.join(n for n, _ in mix)} (the {len(shared)} buckets every run here decoded in, weighted by "
      f"the {tot:,} tokens those runs generated there: "
      + ", ".join(f"{label(*BUCKETS[i])} {w[i] / tot * 100:.1f} %" for i in shared) + ")"
      + ("; the same buckets as those runs' own table" if shared == mixshared else
         f"; those runs' own table had {len(mixshared)} buckets, so their rows here differ from it") + ":")
print()
print("| run | ms a token at the same mix | tok/s at the same mix |")
print("|---|---:|---:|")
for name, b in mix + new:
    ms = sum(w[i] / tot * b[i][2] / b[i][1] for i in shared)
    print(f"| {name} | {ms:.2f} | {1000 / ms:.2f} |")
