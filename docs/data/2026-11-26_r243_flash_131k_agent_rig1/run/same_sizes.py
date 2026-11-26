#!/usr/bin/env python3
"""R243 (Q1921): the requests that read the same prompt on the same cached prefix in every run (same_prompts.md, from
docs/data/2026-11-18_c854b_p106_check/run/same_prompts.py), summed per group of new tokens, and the line against the
preset by the release check's rule (a gain or a loss only beyond the larger of the two arms' spreads).

    python3 -I same_sizes.py SAME_PROMPTS_MD CONTROL_A ARM_A CONTROL_B ARM_B > same_sizes.tsv

Groups: the request on an empty cache (each run's first request after loading), the requests of fewer than 600 new
tokens, of 600 to 1,400 (both read in the taper's micro-batches of 64 in either arm) and of more than 1,400 on a cached
prefix (micro-batches of 512 or 256), and every matched request but the first. The values are the requests' prompt ms
summed, better lower. Standard library only.
"""
import sys

path, ca, ra, cb, rb = sys.argv[1:6]
head, rows = None, []
for ln in open(path):
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if c and c[0] == "task":
        head = c
        continue
    if head is None or len(c) != len(head) or c[0].startswith("**") or set(c[0]) <= {"-", ":"}:
        if head is not None and len(c) != len(head):
            break
        continue
    rows.append(dict(zip(head, c)))
col = {r: f"{r} prompt ms" for r in (ca, ra, cb, rb)}
groups = [
    ("empty cache (the first request after loading)", lambda n, k: k == 0),
    ("under 600 new tokens", lambda n, k: k > 0 and n < 600),
    ("600 to 1,400 new tokens", lambda n, k: k > 0 and 600 <= n <= 1400),
    ("over 1,400 new tokens, on a cached prefix", lambda n, k: k > 0 and n > 1400),
    ("every matched request but the first", lambda n, k: k > 0),
]
print("group\trequests\tnew_tokens\tarm_runs_ms\tbase_runs_ms\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for label, keep in groups:
    sel = [r for r in rows if keep(int(r["prompt tokens read"]), int(r["from the cache"]))]
    if not sel:
        continue
    tot = {x: sum(float(r[col[x]]) for r in sel) for x in (ca, ra, cb, rb)}
    av, bv = [tot[ra], tot[rb]], [tot[ca], tot[cb]]
    am, bm = sum(av) / 2, sum(bv) / 2
    asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
    cls = "negligible" if abs(d) <= max(asp, bsp) else ("gain" if d < 0 else "loss")
    toks = sum(int(r["prompt tokens read"]) for r in sel)
    print(f"{label}\t{len(sel)}\t{toks}\t{av[0]:.1f} {av[1]:.1f}\t{bv[0]:.1f} {bv[1]:.1f}\t{am:.1f}\t{bm:.1f}\t{asp:.1f}\t"
          f"{bsp:.1f}\t{d:+.1f}\t{d / bm * 100:+.1f} %\t{cls}")
