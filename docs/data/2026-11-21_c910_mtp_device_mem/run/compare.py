# C910: compares the server runs of run.sh (steps server and rpc) pair by pair. For each pair: the token ids of the five
# requests (client.py's jsonl), and for MTP-draft runs the drafted tokens of every verify step (the server's
# LLAMA_SPEC_DUMP file), plus the pipeline's counts from the server log. Usage: python3 -I compare.py <dir> > server.tsv
import json
import os
import re
import sys

d = sys.argv[1]

# run, reference run, what differs between them
PAIRS = [
    ("head-plain", "base-plain", "rig builds, dense model, -bs, nothing set"),
    ("head-pipe", "base-pipe", "rig builds, dense model, -bs --pipe-depth 3, nothing set"),
    ("head-mtp", "base-mtp", "rig builds, qwen35moe MTP model, -bs --spec-type draft-mtp, nothing set"),
    ("rpc-head-nd-1", "rpc-base-nd-1", "CPU builds on two RPC devices, MTP draft, --mtp-device unset"),
    ("rpc-base-nd0", "rpc-base-nd-1", "base: --mtp-device 0 against unset"),
    ("rpc-head-nd0", "rpc-head-nd-1", "head: --mtp-device 0 against unset"),
    ("rpc-head-nd0", "rpc-base-nd0", "--mtp-device 0: head's bounded copy against base's whole-head copy"),
]


def load_tokens(name):
    path = os.path.join(d, name + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["req"]: r for r in map(json.loads, f)}


def load_steps(name):
    path = os.path.join(d, name + ".dump.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_drafts(name):
    steps = load_steps(name)
    return None if steps is None else [s.get("draft") for s in steps]


def same_candidates(a, b):
    # the MTP head's top-k candidates and their probabilities at every drafted position (LLAMA_SPEC_DUMP's cand and p)
    sa, sb = load_steps(a), load_steps(b)
    if sa is None and sb is None:
        return "-"
    if sa is None or sb is None or len(sa) != len(sb):
        return "missing or a different step count"
    n = sum(len(s.get("cand") or []) for s in sa)
    diff = sum(1 for x, y in zip(sa, sb) if x.get("cand") != y.get("cand") or x.get("p") != y.get("p"))
    return f"identical at all {n} positions" if diff == 0 else f"DIFFER at {diff} of {len(sa)} steps"


def pipe_counts(name):
    path = os.path.join(d, name + ".log")
    if not os.path.exists(path):
        return "no log"
    text = open(path, errors="replace").read()
    g = r = w = n = 0
    for x in re.finditer(r"pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong", text):
        n += 1
        g += int(x.group(1))
        r += int(x.group(2))
        w += int(x.group(3))
    bad = [l for l in text.splitlines() if re.search(r"abort|segfault|GGML_ASSERT|terminate called|error while", l)]
    return (f"{n} pipelined requests, {g} guesses, {r} right, {w} wrong" if n else "-") + \
        (f"; FAILURE LINES: {bad[:2]}" if bad else "")


def same_tokens(a, b):
    ra, rb = load_tokens(a), load_tokens(b)
    if ra is None or rb is None:
        return "missing"
    n = same = 0
    errs = []
    for req, x in rb.items():
        y = ra.get(req, {})
        n += 1
        if "error" in x or "error" in y:
            errs.append(req)
        elif x.get("tokens") == y.get("tokens"):
            same += 1
    return f"{same} of {n}" + (f" (errors: {', '.join(errs)})" if errs else "")


def same_drafts(a, b):
    da, db = load_drafts(a), load_drafts(b)
    if da is None and db is None:
        return "-"
    if da is None or db is None:
        return "missing"
    na, nb = sum(len(x or []) for x in da), sum(len(x or []) for x in db)
    first = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), None)
    if da == db:
        return f"identical: {len(da)} verify steps, {na} drafted tokens"
    return f"DIFFER: {len(da)} against {len(db)} steps, {na} against {nb} drafted tokens, first difference at step {first}"


print("\t".join(["run", "reference", "what differs", "same tokens", "drafts", "MTP head's top-8 candidates and probabilities",
                 "pipeline (run)"]))
for a, b, what in PAIRS:
    print("\t".join([a, b, what, same_tokens(a, b), same_drafts(a, b), same_candidates(a, b), pipe_counts(a)]))
