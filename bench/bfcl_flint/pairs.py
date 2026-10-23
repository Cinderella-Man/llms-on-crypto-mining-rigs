"""Repeatability checks: compare two runs task by task (same file on other GPUs, or a repeat run).

usage: python pairs.py TASKS.jsonl RAW_DIR A:B [A:B ...]
"""
import json
import os
import sys

from score import canon, check

tasks = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
raw_dir = sys.argv[2]


def load(tag):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(raw_dir, f"{tag}.jsonl"))}


for pair in sys.argv[3:]:
    a, b = pair.split(":")
    ra, rb = load(a), load(b)
    common = [i for i in tasks if i in ra and i in rb]
    same_text = sum(ra[i]["content"] == rb[i]["content"] for i in common)
    ca = {i: check(tasks[i], ra[i]["content"]) for i in common}
    cb = {i: check(tasks[i], rb[i]["content"]) for i in common}
    same_call = sum(canon(ca[i][1]) == canon(cb[i][1]) for i in common)
    ka = sum(ca[i][0] for i in common)
    kb = sum(cb[i][0] for i in common)
    flips = sum(ca[i][0] != cb[i][0] for i in common)
    print(f"{a} vs {b}: n={len(common)} identical text {same_text} ({100*same_text/len(common):.1f} %), "
          f"same calls {same_call} ({100*same_call/len(common):.1f} %), correct {ka} vs {kb} "
          f"({100*ka/len(common):.1f} % vs {100*kb/len(common):.1f} %), tasks whose verdict flips: {flips}")
