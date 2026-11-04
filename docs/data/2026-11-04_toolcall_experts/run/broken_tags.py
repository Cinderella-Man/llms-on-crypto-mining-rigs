"""Broken tool-call tags per `<parameter=` site in agent-check transcripts (Q897; the metric agentionai's Gyro-Coder
v1.2 reports). Reads the proxy log of bench/agent_check (proxy.jsonl and proxy.jsonl.bodies.gz: every request with
the tools it offered and the reply llama-server assembled: reasoning, text and parsed tool calls).
  broken_tags.py LOGDIR [LOGDIR...] --out PREFIX [--arms a,b,...]
The server parses the model's <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call> text
into calls, so a tag the model broke shows up as one of:
  leaked       `<parameter=` text in the reply's text: a call the parser could not read
  in_reasoning `<parameter=` text in the reasoning
  bad_key      a parsed parameter whose name the tool does not have (a garbled `<parameter=...>` tag)
  tag_in_value a parsed value holding tag text (`<parameter=`, `</parameter>`, `<function=`, `</function>`,
               `<tool_call>`, `</tool_call>`): a closing tag missed, so the next tag ran into the value
  missing      a required parameter absent from a call (a dropped `<parameter=` tag)
  bad_json     arguments the server handed over that are not a JSON object (counted as one site)
Sites: every parsed parameter, every `<parameter=` in text or reasoning, every missing required parameter and every
unreadable argument list. Broken per site = the sum of the six kinds over the sites. Also counted: calls to a tool
the request did not offer (bad_function, a garbled `<function=` tag) and other tag text left in text or reasoning.
Writes PREFIX.tsv (per arm), PREFIX.md (the same as a table) and PREFIX_cases.tsv (one row per broken site: arm, the
request's tag, finish reason, kind, tool, parameter name). Standard library only; bench/copy_spans/common.py reads the
log.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "bench", "copy_spans")))
from common import load_run  # noqa: E402

PARAM = re.compile(r"<parameter=")
TAG = re.compile(r"</?(?:parameter|function)[=>]|</?tool_call>")
KINDS = ("leaked", "in_reasoning", "bad_key", "tag_in_value", "missing", "bad_json")


def strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)


def schema(body):
    out = {}
    for t in body.get("tools") or []:
        fn = t.get("function", t)
        p = fn.get("parameters") or {}
        out[fn.get("name")] = (set((p.get("properties") or {}).keys()), set(p.get("required") or []))
    return out


def check(row, add):
    """Sites and broken sites of one reply; add(kind, tool, key) records a broken one."""
    reply, tools = row["reply"], schema(row["body"])
    sites = other_tags = bad_function = 0
    for field, kind in (("content", "leaked"), ("reasoning_content", "in_reasoning")):
        text = reply.get(field) or ""
        for _m in PARAM.finditer(text):
            sites += 1
            add(kind, "-", "-")
        other_tags += len(TAG.findall(text))
    for tc in reply.get("tool_calls") or []:
        fn = tc.get("function", tc)
        name, args = fn.get("name"), fn.get("arguments")
        if name not in tools:
            bad_function += 1
        props, required = tools.get(name, (set(), set()))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = None
        if not isinstance(args, dict):
            sites += 1
            add("bad_json", name, "-")
            continue
        for key, val in args.items():
            sites += 1
            if name in tools and key not in props:
                add("bad_key", name, key)
            elif any(TAG.search(s) for s in strings(val)):
                add("tag_in_value", name, key)
        for key in sorted(required - set(args)):
            sites += 1
            add("missing", name, key)
    return sites, other_tags, bad_function, len(reply.get("tool_calls") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms")
    a = ap.parse_args()
    arms = set(a.arms.split(",")) if a.arms else None
    stats = defaultdict(Counter)
    cases = []
    for path in a.logs:
        if not os.path.exists(os.path.join(path, "proxy.jsonl.bodies.gz")):
            print(f"{path}: no proxy.jsonl.bodies.gz, skipped", file=sys.stderr)
            continue
        for row in load_run(path, os.path.basename(os.path.normpath(path))):
            if arms and row["arm"] not in arms:
                continue
            st = stats[row["arm"]]
            tag = f"{row['arm']}/{row['task']}/{row['attempt']}/{row['try_']}/{row['turn']}"

            def add(kind, tool, key, st=st, tag=tag, finish=row["finish"]):
                st[kind] += 1
                cases.append((tag.split("/")[0], tag, finish, kind, tool or "-", key))

            sites, other, bad_fn, calls = check(row, add)
            st["requests"] += 1
            st["calls"] += calls
            st["sites"] += sites
            st["other_tags"] += other
            st["bad_function"] += bad_fn
    cols = ["requests", "calls", "sites", *KINDS, "broken", "broken_per_site_pct", "bad_function", "other_tags"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tsv", "w") as f, open(a.out + ".md", "w") as g:
        f.write("arm\t" + "\t".join(cols) + "\n")
        g.write("| arm | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for arm in sorted(stats):
            st = stats[arm]
            st["broken"] = sum(st[k] for k in KINDS)
            vals = [str(st[c]) for c in cols[:-3]]
            vals += [f"{100 * st['broken'] / st['sites']:.3f}" if st["sites"] else "-", str(st["bad_function"]),
                     str(st["other_tags"])]
            f.write(arm + "\t" + "\t".join(vals) + "\n")
            g.write(f"| {arm} | " + " | ".join(vals) + " |\n")
    with open(a.out + "_cases.tsv", "w") as f:
        f.write("arm\trequest\tfinish\tkind\ttool\tparameter\n")
        f.writelines("\t".join(map(str, c)) + "\n" for c in cases)
    sys.stdout.write(open(a.out + ".md").read())


if __name__ == "__main__":
    main()
