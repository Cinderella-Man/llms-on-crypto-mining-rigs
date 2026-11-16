#!/usr/bin/env python3
"""prompt_by_size.py DIR ARM...: each arm's prompt speed by prompt size, from DIR/agent/<arm>/turns.csv.

Every timed turn (one request) carries the server's own prompt_n (prompt tokens it processed) and prompt_ms. Per arm
and size bucket (under 512, 512-1,023, 1,024-1,999, 2,000 and more prompt tokens): requests, tokens, seconds and
tokens a second (the bucket's tokens over its seconds, as the report's session figure). A turn without timings (an
attempt stopped by the cap) counts in none. Prints a TSV. Runs anywhere with Python 3 (no packages).
"""
import csv
import os
import sys

BUCKETS = [("0-511", 0, 512), ("512-1023", 512, 1024), ("1024-1999", 1024, 2000), ("2000+", 2000, 1 << 30)]
d, arms = sys.argv[1], sys.argv[2:]
print("arm\tprompt_tokens\trequests\ttokens\tprompt_s\tprompt_tok_s")
for arm in arms:
    rows = []
    for t in csv.DictReader(open(os.path.join(d, "agent", arm, "turns.csv"))):
        try:
            rows.append((int(t["prompt_n"]), float(t["prompt_ms"])))
        except (TypeError, ValueError):
            continue
    for name, lo, hi in BUCKETS + [("all", 0, 1 << 30)]:
        b = [r for r in rows if lo <= r[0] < hi]
        n, ms = sum(r[0] for r in b), sum(r[1] for r in b)
        print(f"{arm}\t{name}\t{len(b)}\t{n}\t{ms / 1000:.1f}\t{n / ms * 1000:.1f}" if b else f"{arm}\t{name}\t0\t0\t0.0\t")
