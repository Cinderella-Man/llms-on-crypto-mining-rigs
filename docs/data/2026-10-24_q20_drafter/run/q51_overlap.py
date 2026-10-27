"""Q20 (2026-10-25): prompts that share a 12-word span with any Q51 test prompt (the rule the answer store uses too), so
that training on them can't flatter the Q51 scores: the edit prompts reuse Q51's instruction texts verbatim. Writes
WORK_DIR/q51-overlap-ids.txt (ids over every prompt file, including prompts-clean*.jsonl) and prints counts per file
and category.

usage: python3 -I q51_overlap.py WORK_DIR [Q51_PROMPTS.jsonl]
  WORK_DIR holds the drafter's prompt files (prompts*.jsonl); Q51_PROMPTS defaults to this repository's
  release/flint-v2/reproduce/data/q51/prompts.jsonl.
"""
import collections
import glob
import json
import os
import sys

N = 12
W = sys.argv[1]
Q51 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../release/flint-v2/reproduce/data/q51/prompts.jsonl")


def words(r):
    text = " ".join((m.get("content") or "") for m in r["messages"])
    return text.lower().split()


spans = set()
for line in open(Q51):
    w = words(json.loads(line))
    spans |= {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}
hits, per = [], collections.Counter()
for f in sorted(glob.glob(f"{W}/prompts*.jsonl")):
    if any(x in f for x in ("allused", "all12", "all123", ".v1.")):
        continue
    for line in open(f):
        r = json.loads(line)
        w = words(r)
        if any(" ".join(w[i:i + N]) in spans for i in range(len(w) - N + 1)):
            hits.append(r["id"])
            per[(f.split("/")[-1], r["category"])] += 1
with open(f"{W}/q51-overlap-ids.txt", "w") as f:
    f.write("\n".join(hits) + "\n")
for k, v in sorted(per.items()):
    print(v, k)
print(len(hits), "prompts share a 12-word span with Q51")
