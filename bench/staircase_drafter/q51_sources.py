"""The source files Q51's test prompts are built from (named in the edit prompts' "Here is X:" and the agent prompts'
read_file calls and texts), written as {"source": name} lines for clean_prompts.py --q51 and check_sources.py
--q51-sources, so training prompts never use them.

usage: python3 -I q51_sources.py OUT.jsonl [--q51 Q51_PROMPTS.jsonl]
  --q51 defaults to this repository's release/flint-v2/reproduce/data/q51/prompts.jsonl
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("--q51", default=os.path.join(HERE, "../../release/flint-v2/reproduce/data/q51/prompts.jsonl"))
args = ap.parse_args()

names = set()
for line in open(args.q51):
    r = json.loads(line)
    for m in r["messages"]:
        text = m.get("content") or ""
        names |= set(re.findall(r"Here is ([\w./-]+\.\w+):", text))
        names |= set(re.findall(r"\b([\w-]+\.(?:py|c|cpp|h|hpp|cu|sh|js|ts|go|rs))\b", text))
        for call in m.get("tool_calls") or []:
            a = json.loads(call["function"]["arguments"])
            if "path" in a:
                names.add(a["path"])
with open(args.out, "w") as f:
    for n in sorted(names):
        f.write(json.dumps({"source": n}) + "\n")
print(len(names), "Q51 source names:", sorted(names)[:40])
