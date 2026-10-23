"""Build the BFCL v4 single-turn task list and render every prompt with Flint's chat template.

usage: python prepare.py TEMPLATE.jinja OUT.jsonl [--per-category N] [--seed S] [--exclude IDS.txt]
Default: the 3,477 Python single-turn tasks that pass selftest.py (all 3,491 minus excluded_single_turn.txt).
--per-category N takes a seeded sample of N tasks per category (all of a category if it has fewer), for pilots.
--exclude '' keeps every task.
"""
import argparse
import json
import os
import random

from bfcl_common import (SINGLE_TURN, build_tools, load_dataset_entry, load_ground_truth_entry, make_template,
                         sha256_of)

ap = argparse.ArgumentParser()
ap.add_argument("template")
ap.add_argument("out")
ap.add_argument("--per-category", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "excluded_single_turn.txt"))
args = ap.parse_args()

tmpl = make_template(open(args.template).read())
exclude = set()
if args.exclude:
    exclude = {line.split()[0] for line in open(args.exclude) if line.strip() and not line.startswith("#")}

tasks = []
for cat in SINGLE_TURN:
    data = [e for e in load_dataset_entry(cat) if e["id"] not in exclude]
    answers = {}
    if "relevance" not in cat:  # (ir)relevance tasks have no ground truth: any call / no call is the answer
        answers = {a["id"]: a["ground_truth"] for a in load_ground_truth_entry(cat)}
    if args.per_category and args.per_category < len(data):
        rng = random.Random(f"{args.seed}-{cat}")
        data = [data[i] for i in sorted(rng.sample(range(len(data)), args.per_category))]
    for e in data:
        assert len(e["question"]) == 1, e["id"]  # single-turn categories only
        messages = e["question"][0]
        tools = build_tools(e["function"])
        tasks.append({
            "id": e["id"],
            "category": cat,
            "messages": messages,
            "functions": e["function"],
            "tools": tools,
            "ground_truth": answers.get(e["id"]),
            "prompt": tmpl.render(messages=messages, tools=tools, add_generation_prompt=True),
        })

with open(args.out, "w") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(len(tasks), "tasks ->", args.out, "sha256", sha256_of({"id": t["id"], "prompt": t["prompt"]} for t in tasks))
