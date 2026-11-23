# Pull requests 9 and 10 (2026-11-23): reads the server check's runs (run.sh, step server) and writes one table row per
# run: the LLAMA_PIPE_MEDIA value, the pipeline's counts from the server log, and the token ids against the base's run of
# the same mode. No projector is loaded, so every value of the switch must give the base's tokens and the base's pipeline
# counts. Usage: python3 -I compare.py <dir> > server.tsv (C908's compare.py with these runs)
import json
import os
import re
import sys

d = sys.argv[1]

# run name, binary, LLAMA_PIPE_MEDIA as set ("unset" = not in the environment), server flags
RUNS = [
    ("base-plain", "3dfad876f", "unset", "-bs"),
    ("head-plain", "head", "unset", "-bs"),
    ("base-pipe", "3dfad876f", "unset", "-bs --pipe-depth 3"),
    ("head-pipe", "head", "unset", "-bs --pipe-depth 3"),
    ("head-pipe-media0", "head", "0", "-bs --pipe-depth 3"),
    ("head-pipe-media1", "head", "1", "-bs --pipe-depth 3"),
    ("head-pipe-media2", "head", "2", "-bs --pipe-depth 3"),
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
print("\t".join(["run", "binary", "LLAMA_PIPE_MEDIA", "flags", "pipelined requests", "guesses", "right", "wrong",
                 "grammar rejections", "same tokens as", "pipeline counts as base-pipe"]))
for name, binary, env, flags in RUNS:
    s = pipe_stats(name)
    if s is None:
        print(f"{name}\t{binary}\t{env}\t{flags}\tno log")
        continue
    ref = "base-plain" if "pipe" not in name else "base-pipe"
    ref_txt = "-" if name.startswith("base-") else f"{ref}: {same_tokens(ref, name)}"
    if "pipe" not in name or name == "base-pipe":
        counts = "-"
    elif base_pipe is None:
        counts = "no base-pipe log"
    else:
        keys = ("requests", "guesses", "right", "wrong", "rejections")
        counts = "same" if all(s[k] == base_pipe[k] for k in keys) else "DIFFERENT"
    rej = str(s["rejections"]) if s["field"] or s["requests"] == 0 else "- (not printed)"
    print("\t".join([name, binary, env, flags, str(s["requests"]), str(s["guesses"]), str(s["right"]), str(s["wrong"]),
                     rej, ref_txt, counts + (f"; FAILURE LINES: {s['failures'][:3]}" if s["failures"] else "")]))
