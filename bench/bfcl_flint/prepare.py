"""Pick the fixed 300-task BFCL v3 subset and render every prompt with Flint's chat template.

usage: python prepare.py TEMPLATE.jinja OUT.jsonl [SEED]
"""
import json
import random
import sys

from bfcl_common import BFCL_ROOT, CATEGORIES, COUNTS, build_tools, load_jsonl, make_template

template_path, out_path = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tmpl = make_template(open(template_path).read())

tasks = []
for cat in CATEGORIES:
    data = load_jsonl(f"{BFCL_ROOT}/data/BFCL_v3_{cat}.json")
    answers = {}
    if "irrelevance" not in cat:
        answers = {a["id"]: a["ground_truth"]
                   for a in load_jsonl(f"{BFCL_ROOT}/data/possible_answer/BFCL_v3_{cat}.json")}
    rng = random.Random(f"{seed}-{cat}")
    picked = sorted(rng.sample(range(len(data)), COUNTS[cat]))
    for i in picked:
        e = data[i]
        assert len(e["question"]) == 1, e["id"]  # single-turn categories only
        messages = e["question"][0]
        tools = build_tools(e["function"], cat)
        prompt = tmpl.render(messages=messages, tools=tools, add_generation_prompt=True)
        tasks.append({
            "id": e["id"],
            "category": cat,
            "messages": messages,
            "functions": e["function"],
            "tools": tools,
            "ground_truth": answers.get(e["id"]),
            "prompt": prompt,
        })

with open(out_path, "w") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(len(tasks), "tasks ->", out_path)
