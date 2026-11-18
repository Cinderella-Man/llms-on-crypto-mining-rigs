#!/usr/bin/env python3
"""same_prompts.py NAME=TURNS.csv ...: the requests that read the same prompt in every run, side by side.

A request is matched across runs by (task, turn, prompt tokens read, tokens from the cache): with both counts equal
the server read the same number of new tokens on top of the same cached prefix, so its prompt milliseconds compare
the engines on the same work. Every task's first turn is the same request in every run; later turns match only where
the agent's earlier replies happened to have the same length. Prints one Markdown row a matched request (each run's
prompt ms) and a total per run, then the decode speed of every run over all its requests (tokens generated / decode
seconds) for reference. Standard library only: python3 -I same_prompts.py a=turns.csv b=turns.csv ...
"""
import csv
import sys

runs = []
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    rows = {}
    gen_n = gen_ms = 0.0
    for r in csv.DictReader(open(path)):
        if r["attempt"] != "1":
            continue
        key = (r["task_id"], int(r["turn"]), int(r["prompt_n"]), int(r["cache_n"]))
        rows[key] = float(r["prompt_ms"])
        gen_n += int(r["predicted_n"])
        gen_ms += float(r["predicted_ms"])
    runs.append((name, rows, gen_n, gen_ms))

common = sorted(set.intersection(*(set(r[1]) for r in runs)))
names = [r[0] for r in runs]
print("| task | turn | prompt tokens read | from the cache | " + " | ".join(f"{n} prompt ms" for n in names) + " |")
print("|---|---:|---:|---:|" + "---:|" * len(names))
for k in common:
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(f"{r[1][k]:.1f}" for r in runs) + " |")
tot = [sum(r[1][k] for k in common) for r in runs]
toks = sum(k[2] for k in common)
print(f"| **all {len(common)} matched requests** | | {toks} | | "
      + " | ".join(f"**{t:.1f}** ({toks / t * 1000:.1f} tok/s)" for t in tot) + " |")
first = [k for k in common if k[1] == 1]
tf = [sum(r[1][k] for k in first) for r in runs]
tft = sum(k[2] for k in first)
print(f"| **of them, the {len(first)} first turns** | | {tft} | | "
      + " | ".join(f"**{t:.1f}** ({tft / t * 1000:.1f} tok/s)" for t in tf) + " |")
print()
print("| run | requests | tokens generated | decode s | generation tok/s |")
print("|---|---:|---:|---:|---:|")
for name, rows, gn, gms in runs:
    print(f"| {name} | {len(rows)} | {int(gn)} | {gms / 1000:.1f} | {gn / gms * 1000:.2f} |")
