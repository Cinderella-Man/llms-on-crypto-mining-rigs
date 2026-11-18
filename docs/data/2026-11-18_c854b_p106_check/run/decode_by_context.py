#!/usr/bin/env python3
"""decode_by_context.py NAME=TURNS.csv ...: each run's decode speed by context, and at one shared mix of contexts.

Each request (attempt 1, with timings) is put in the bucket of its middle context: the context it started at
(cache_n + prompt_n) plus half the tokens it generated, since a long reply grows its own context and decodes slower as
it goes. Prints one Markdown row a run and bucket (requests, tokens generated, decode s, tok/s, ms a token), then each
run's generation at the same mix: its ms a token in each bucket weighted by the tokens all the runs together generated
in that bucket, so runs whose replies were longer or sat deeper in the context compare on the same depths. Ornith has
no draft model, so a decode step is one token. Standard library only: python3 -I decode_by_context.py a=turns.csv ...
"""
import csv
import sys

BUCKETS = [(0, 3000), (3000, 4500), (4500, 6000), (6000, 8000), (8000, 10 ** 9)]


def label(lo, hi):
    return f"under {hi:,}" if lo == 0 else f"{lo:,} and over" if hi == 10 ** 9 else f"{lo:,} to {hi:,}"


runs = []
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
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
    runs.append((name, b))

print("| run | middle context, tokens | requests | tokens generated | decode s | tok/s | ms a token |")
print("|---|---|---:|---:|---:|---:|---:|")
for name, b in runs:
    for (lo, hi), (q, n, ms) in zip(BUCKETS, b):
        if q:
            print(f"| {name} | {label(lo, hi)} | {q} | {n} | {ms / 1000:.1f} | {n / ms * 1000:.2f} | {ms / n:.2f} |")
    q, n, ms = (sum(x[i] for x in b) for i in range(3))
    print(f"| {name} | **all** | {q} | {n} | {ms / 1000:.1f} | **{n / ms * 1000:.2f}** | {ms / n:.2f} |")

# the shared mix: the tokens all runs generated in each bucket, over the buckets every run has
shared = [i for i in range(len(BUCKETS)) if all(b[i][1] for _, b in runs)]
w = {i: sum(b[i][1] for _, b in runs) for i in shared}
tot = sum(w.values())
print()
print(f"At the same mix of contexts (the {len(shared)} buckets every run decoded in, weighted by the "
      f"{tot:,} tokens all runs generated there: "
      + ", ".join(f"{label(*BUCKETS[i])} {w[i] / tot * 100:.1f} %" for i in shared) + "):")
print()
print("| run | ms a token at the same mix | tok/s at the same mix |")
print("|---|---:|---:|")
for name, b in runs:
    ms = sum(w[i] / tot * b[i][2] / b[i][1] for i in shared)
    print(f"| {name} | {ms:.2f} | {1000 / ms:.2f} |")
