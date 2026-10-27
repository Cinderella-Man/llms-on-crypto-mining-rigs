"""Sampling weights for train.py --rec-weights: edit, agent and code answers twice per epoch, the rest once (code is the
main point of the drafter). Writes one "answer_id weight" line for every id in IDS (e.g. the licence-checked answer
list), taking each id's category from the prompt files.

usage: python3 -I rec_weights.py IDS.txt OUT.txt PROMPTS.jsonl [PROMPTS.jsonl ...]
  e.g. python3 -I rec_weights.py strict-ids.txt rec-weights-code.txt prompts.jsonl prompts-ext.jsonl prompts-ext2.jsonl \
           prompts-tools.jsonl
"""
import json
import sys

if len(sys.argv) < 4:
    raise SystemExit(__doc__)
ids_file, out_file, prompt_files = sys.argv[1], sys.argv[2], sys.argv[3:]
cat = {}
for f in prompt_files:
    for line in open(f):
        r = json.loads(line)
        cat[r["id"]] = r["category"]
ids = [x.strip() for x in open(ids_file) if x.strip()]
with open(out_file, "w") as f:
    for i in ids:
        f.write(f"{i} {2 if cat.get(i) in ('edit', 'agent', 'code') else 1}\n")
print(len(ids), "weights written;", sum(cat.get(i) in ("edit", "agent", "code") for i in ids), "at 2")
