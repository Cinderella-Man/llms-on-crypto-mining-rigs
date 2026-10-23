"""Harness self-test without a model: write each task's ground-truth answer as an
assistant tool call through Flint's own chat template, decode it back and run the
BFCL checker on it. Every AST task should pass; every irrelevance task (no call) too.

usage: python selftest.py TEMPLATE.jinja TASKS.jsonl
"""
import json
import re
import sys

from bfcl_common import ast_checker, decode_flint, is_empty_output, make_template

tmpl = make_template(open(sys.argv[1]).read())
tasks = [json.loads(l) for l in open(sys.argv[2])]
ok = bad = 0
for t in tasks:
    if t["ground_truth"] is None:
        text = "I cannot help with that using the available functions."
        good = is_empty_output(decode_flint(text, t["tools"]))
    else:
        calls = []
        for c in t["ground_truth"]:
            (name, params), = c.items()
            args = {}
            for p, vals in params.items():
                v = [x for x in vals if x != ""]
                if v:
                    args[p] = v[0]
            calls.append({"function": {"name": re.sub(r"\.", "_", name), "arguments": args}})
        msgs = t["messages"] + [{"role": "assistant", "content": "", "tool_calls": calls}]
        full = tmpl.render(messages=msgs, tools=t["tools"], add_generation_prompt=False)
        assert full.startswith(t["prompt"]), t["id"]
        gen = full[len(t["prompt"]):]
        dec = decode_flint(gen, t["tools"])
        r = ast_checker(t["functions"], dec, t["ground_truth"], "Python", t["category"], "flint-FC")
        good = r["valid"]
        if not good:
            print("FAIL", t["id"], r["error"][:2], repr(gen[:300]))
    ok += good
    bad += not good
print(f"selftest: {ok} pass, {bad} fail")
