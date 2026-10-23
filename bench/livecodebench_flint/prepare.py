"""Render every LiveCodeBench problem as a Flint prompt: LiveCodeBench's OpenAI-chat messages through Flint's chat
template, ending with the generation prompt (Flint opens its <think> block itself).

usage: python prepare.py TEMPLATE.jinja PROBLEMS.jsonl OUT.jsonl [--start 2024-08-01] [--end 2025-05-01]
                         [--sample N] [--seed S] [--exclude IDS.txt]
The window is start <= contest_date < end; --sample N takes a seeded sample of N problems (pilots). The problems in
excluded_problems.txt are left out (--exclude '' keeps them): 450 of the default window's 454 remain.
"""
import argparse
import json
import os
import random

from lcb_common import load_problems, make_template, messages, sha256_of

ap = argparse.ArgumentParser()
ap.add_argument("template")
ap.add_argument("problems")
ap.add_argument("out")
ap.add_argument("--start", default="2024-08-01")
ap.add_argument("--end", default="2025-05-01")
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "excluded_problems.txt"))
args = ap.parse_args()
exclude = set()
if args.exclude:
    exclude = {line.split()[0] for line in open(args.exclude) if line.strip() and not line.startswith("#")}

tmpl = make_template(open(args.template).read())
tasks = []
for p in load_problems(args.problems, with_tests=False):
    if not args.start <= p.contest_date.date().isoformat() < args.end or p.question_id in exclude:
        continue
    msgs = messages(p)
    tasks.append({"id": p.question_id, "category": p.difficulty.value, "platform": p.platform.value,
                  "contest_id": p.contest_id, "contest_date": p.contest_date.date().isoformat(), "messages": msgs,
                  "prompt": tmpl.render(messages=msgs, add_generation_prompt=True)})
if args.sample and args.sample < len(tasks):
    tasks = sorted(random.Random(args.seed).sample(tasks, args.sample), key=lambda t: (t["contest_date"], t["id"]))
with open(args.out, "w") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(len(tasks), "problems ->", args.out, "sha256", sha256_of({"id": t["id"], "prompt": t["prompt"]} for t in tasks))
