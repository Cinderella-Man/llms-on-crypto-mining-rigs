# C908: reads the server check's runs (run.sh, step server) and writes one table row per run: the switch value, the
# pipeline's counts from the server log, the token ids against the base's pipelined run, and whether the pipeline's
# grammar mode was on. Grammar mode on shows as grammar rejections, or as more guesses than the base made (the pipe
# then also runs inside the two JSON-schema requests). Usage: python3 -I compare.py <dir> > server.tsv
import json
import os
import re
import sys

d = sys.argv[1]

# run name, binary, LLAMA_PIPE_GRAMMAR as set ("unset" = not in the environment), server flags
RUNS = [
    ("base-plain", "602691153", "unset", "-bs"),
    ("head-plain", "head", "unset", "-bs"),
    ("base-pipe", "602691153", "unset", "-bs --pipe-depth 3"),
    ("head-pipe", "head", "unset", "-bs --pipe-depth 3"),
    ("head-pipe-gram-empty", "head", "empty", "-bs --pipe-depth 3"),
    ("head-pipe-gram0", "head", "0", "-bs --pipe-depth 3"),
    ("head-pipe-gram-true", "head", "true", "-bs --pipe-depth 3"),
    ("head-pipe-gram1", "head", "1", "-bs --pipe-depth 3"),
]


def load(name):
    path = os.path.join(d, name + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["req"]: r for r in map(json.loads, f)}


def pipe_stats(name):
    path = os.path.join(d, name + ".log")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    s = {"requests": 0, "guesses": 0, "right": 0, "wrong": 0, "rejections": 0, "field": 0}
    for x in re.finditer(r"pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong(?:.*?(\d+) grammar rejections)?", text):
        s["requests"] += 1
        s["guesses"] += int(x.group(1))
        s["right"] += int(x.group(2))
        s["wrong"] += int(x.group(3))
        if x.group(4) is not None:
            s["field"] += 1
            s["rejections"] += int(x.group(4))
    s["failures"] = [l for l in text.splitlines()
                     if re.search(r"abort|segfault|GGML_ASSERT|terminate called|pipelined decoding failed", l)]
    return s


def same_tokens(a, b):
    ra, rb = load(a), load(b)
    if ra is None or rb is None:
        return "missing"
    n = same = 0
    errs = []
    for req, x in ra.items():
        y = rb.get(req, {})
        n += 1
        if "error" in x or "error" in y:
            errs.append(req)
        elif x["tokens"] == y["tokens"]:
            same += 1
    return f"{same} of {n}" + (f" (errors: {', '.join(errs)})" if errs else "")


base_pipe = pipe_stats("base-pipe")
print("\t".join(["run", "binary", "LLAMA_PIPE_GRAMMAR", "flags", "pipelined requests", "guesses", "right", "wrong",
                 "grammar rejections", "same tokens as", "grammar mode"]))
for name, binary, env, flags in RUNS:
    s = pipe_stats(name)
    if s is None:
        print(f"{name}\t{binary}\t{env}\t{flags}\tno log")
        continue
    ref = "base-plain" if "pipe" not in name else "base-pipe"
    if name.startswith("base-"):
        ref_txt = "-"
    else:
        ref_txt = f"{ref}: {same_tokens(ref, name)}"
    if "pipe" not in name:
        mode = "- (no pipe)"
    elif s["rejections"] > 0 or (base_pipe and s["guesses"] > base_pipe["guesses"]):
        mode = "ON"
    else:
        mode = "off"
    rej = str(s["rejections"]) if s["field"] or s["requests"] == 0 else "- (not printed)"
    print("\t".join([name, binary, env, flags, str(s["requests"]), str(s["guesses"]), str(s["right"]), str(s["wrong"]),
                     rej, ref_txt, mode + (f"; FAILURE LINES: {s['failures'][:3]}" if s["failures"] else "")]))
