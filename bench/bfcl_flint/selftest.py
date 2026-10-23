"""Harness self-test without a model: write each task's ground-truth answer as an assistant turn (a reasoning block,
then the tool calls) through Flint's own chat template, parse it back with the harness's parser and run the BFCL
checker on it. Every AST task should pass; every irrelevance task (no call) and live_relevance task (a call) too.

usage: python selftest.py TEMPLATE.jinja TASKS.jsonl [FAILED_IDS.txt]
"""
import json
import re
import sys

from bfcl_common import MODEL, Language, ast_checker, decode_ast, is_empty_output, make_template, split_output

tmpl = make_template(open(sys.argv[1]).read())
tasks = [json.loads(line) for line in open(sys.argv[2])]


def generated_text(t, calls, content=""):
    msgs = t["messages"] + [{"role": "assistant", "reasoning_content": "Self-test.", "content": content,
                             "tool_calls": calls}]
    full = tmpl.render(messages=msgs, tools=t["tools"], add_generation_prompt=False)
    assert full.startswith(t["prompt"]), t["id"]
    return full[len(t["prompt"]):]


_OMIT = object()


def choose(allowed, required=False):
    """One allowed value: _OMIT when "" (optional) is allowed and the parameter is not required, else the first
    other allowed value."""
    rest = [x for x in allowed if x != ""]
    if not rest or ("" in allowed and not required):
        return _OMIT
    return rest[0]


def pick_dict(answer):
    """A dict answer lists the allowed values of each key, one level deep (BFCL's dict_checker)."""
    return {k: c for k, vals in answer.items() if (c := choose(vals)) is not _OMIT}


def arguments(t, name, params):
    """One correct argument set for a ground-truth call, typed the way BFCL's checker reads it."""
    doc = next(f for f in t["functions"] if f["name"] == name)
    props, required = doc["parameters"]["properties"], doc["parameters"].get("required", [])
    args = {}
    for p, allowed in params.items():
        v = choose(allowed, p in required)
        if v is _OMIT:
            continue
        ptype = props.get(p, {}).get("type")
        if ptype == "dict":
            v = pick_dict(v)
        elif ptype in ("array", "tuple") and props[p].get("items", {}).get("type") == "dict":
            v = [pick_dict(x) for x in v]
        args[p] = v
    return args


failed = []
for t in tasks:
    cat = t["category"]
    if "irrelevance" in cat:
        text = generated_text(t, [], "None of the available functions can do this.")
        good = is_empty_output(decode_ast(split_output(text, t["tools"])))
    elif cat == "live_relevance":
        name = t["tools"][0]["function"]["name"]
        good = not is_empty_output(decode_ast(split_output(generated_text(t, [{"function": {
            "name": name, "arguments": {}}}]), t["tools"])))
    else:
        calls = [{"function": {"name": re.sub(r"\.", "_", name), "arguments": arguments(t, name, params)}}
                 for c in t["ground_truth"] for name, params in c.items()]
        gen = generated_text(t, calls)
        parsed = split_output(gen, t["tools"])
        dec = decode_ast(parsed)
        r = ast_checker(t["functions"], dec, t["ground_truth"], Language.PYTHON, cat, MODEL)
        good = r["valid"]
        if not good:
            print("FAIL", t["id"], r["error"][:2], repr(gen[:300]))
    if not good:
        failed.append(t["id"])
print(f"selftest: {len(tasks) - len(failed)} pass, {len(failed)} fail")
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write("".join(f"{i}\n" for i in failed))
