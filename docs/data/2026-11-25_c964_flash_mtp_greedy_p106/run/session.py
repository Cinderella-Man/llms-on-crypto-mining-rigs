#!/usr/bin/env python3
"""C964 (Q1694): C951's three greedy requests against llama-server's /completion endpoint, then the same three again
with the top two candidates of every generated token (n_probs 2). Nothing timed is written.

The requests are C951's (docs/data/2026-11-21_c951_flash_head_gpu/run/session.py), with the same request bodies:
  code   C854b's coding request, greedy (temperature 0, top-k 1), 256 tokens, no prompt cache
  prose  C854b's prose paragraph, greedy
  deep   the most whole lines from the start of a text file (C951: docs/switches.md at 6a9cf9d9, which gives 170 lines
         and 8,189 tokens) whose prompt with the question fits in DEPTH tokens, greedy
Pass "p1" sends them as C951 did; pass "p2" adds n_probs 2 (the probabilities of the two most likely tokens before
sampling, at each generated position), so the first differing token's two candidates can be read. A p2 reply whose
token ids differ from p1's is shown in the table (n_probs takes the server's host sampling path).

Usage: python3 -I session.py <server URL> <output folder> <text file> [DEPTH (8192)]
Writes <out>/<pass>-<request>.json (the server's reply), <out>/deep-prompt.md5 and one line per request in
<out>/session.tsv: pass, request, prompt tokens read, generated tokens, md5 of the token ids, md5 of the text, drafted
tokens, accepted drafts, the stop reason.
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT, TEXT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 8192

P_CODE = ("Write a Python function that parses an ISO 8601 date string into a datetime object, "
          "then three pytest tests for it.")
P_PROSE = "Explain in one paragraph how a hash map handles collisions, then give one example in Python."
QUESTION = ("\n\nQuestion: list the switches above that change how the MTP draft runs, "
            "one line each with what it does.\nAnswer:\n")
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1}


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def n_tokens(text):
    return len(post("/tokenize", {"content": text, "add_special": True}, 120)["tokens"])


def deep_prompt():
    with open(TEXT, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens("".join(lines[:mid]) + QUESTION) <= DEPTH:
            lo = mid
        else:
            hi = mid - 1
    return "".join(lines[:lo]) + QUESTION, lo


def complete(tag, name, prompt, extra):
    body = dict(GREEDY, prompt=prompt, n_predict=256, cache_prompt=False, return_tokens=True, **extra)
    try:
        reply = post("/completion", body)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{tag}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    text = str(reply.get("content", ""))
    t = reply.get("timings") or {}
    row = [tag, name, str(t.get("prompt_n", "")), str(len(toks)),
           hashlib.md5(",".join(str(x) for x in toks).encode()).hexdigest() if toks else "",
           hashlib.md5(text.encode("utf-8")).hexdigest() if toks else "",
           str(t.get("draft_n", "")), str(t.get("draft_n_accepted", "")),
           str(reply.get("stop_type", "")) if toks else "error: " + str(reply.get("error", ""))[:80]]
    with open(os.path.join(OUT, "session.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row), flush=True)


os.makedirs(OUT, exist_ok=True)
deep, n_lines = deep_prompt()
with open(os.path.join(OUT, "deep-prompt.md5"), "w") as f:
    f.write(hashlib.md5(deep.encode("utf-8")).hexdigest() + f"  {len(deep)} characters, {n_lines} lines\n")
for tag, extra in (("p1", {}), ("p2", {"n_probs": 2})):
    complete(tag, "code", P_CODE, extra)
    complete(tag, "prose", P_PROSE, extra)
    complete(tag, "deep", deep, extra)
