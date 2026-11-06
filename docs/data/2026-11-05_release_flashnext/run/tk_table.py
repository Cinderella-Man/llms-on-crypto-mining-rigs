#!/usr/bin/env python3
"""tk_table.py LABEL=FILE...: llama-bench -p 1,2,3,4,5 -n 128 jsonl per load -> one row each: the median ms a pass of
k tokens (k = 1..5, over the repetitions) and tg128's mean ms a token."""
import json
import sys
from statistics import median

print("load\tbuild\tk1_ms\tk2_ms\tk3_ms\tk4_ms\tk5_ms\ttg128_ms_per_token")
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pp = {r["n_prompt"]: median(r["samples_ns"]) / 1e6 for r in rows if r["n_prompt"] and not r["n_gen"]}
    tg = [r["avg_ns"] / r["n_gen"] / 1e6 for r in rows if r["n_gen"] and not r["n_prompt"]]
    print(label, rows[0]["build_commit"], *("%.1f" % pp[k] for k in range(1, 6)), "%.1f" % tg[0], sep="\t")
