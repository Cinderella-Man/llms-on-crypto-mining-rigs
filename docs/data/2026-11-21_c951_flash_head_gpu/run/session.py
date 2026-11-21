#!/usr/bin/env python3
"""C951 (Q1687): the requests each Flash-Next load answers, against llama-server's /completion endpoint. Every request
is a fresh prompt (no prompt cache), 256 tokens, the generated token ids returned; nothing timed is written.

  code     C854b's coding request (docs/data/2026-11-18_c854b_p106_check/), greedy (temperature 0, top-k 1)
  prose    C854b's prose paragraph, greedy
  deep     about 8,192 tokens of this repository's docs/switches.md (whole lines from its start, cut with the server's
           /tokenize so the prompt is at most DEPTH tokens) and a question about it, greedy: the draft attends over a
           cache 8,192 cells deep, the depth of R205's and FN115's deep lane
  deep-s   the deep prompt with the preset's sampler (temperature 1.0, top-p 0.95, top-k 20, min-p 0), seed 7
  code-s   the coding request with the preset's sampler, seed 7

Usage: python3 -I session.py <server URL> <output folder> <round tag> <text file> [DEPTH (8192)]
Writes <out>/<round>-<request>.json (the server's reply) and appends one line per request to <out>/session.tsv:
round, request, prompt tokens read, generated tokens, the md5 of the token ids, the md5 of the text, drafted tokens,
accepted drafts, the stop reason and the reply's first 60 characters (JSON-escaped).
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT, ROUND, TEXT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 8192

P_CODE = ("Write a Python function that parses an ISO 8601 date string into a datetime object, "
          "then three pytest tests for it.")
P_PROSE = "Explain in one paragraph how a hash map handles collisions, then give one example in Python."
QUESTION = ("\n\nQuestion: list the switches above that change how the MTP draft runs, "
            "one line each with what it does.\nAnswer:\n")
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1}
PRESET = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": 7}


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def n_tokens(text):
    return len(post("/tokenize", {"content": text, "add_special": True}, 120)["tokens"])


def deep_prompt():
    with open(TEXT, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    lo, hi = 1, len(lines)  # the most whole lines whose prompt fits in DEPTH tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens("".join(lines[:mid]) + QUESTION) <= DEPTH:
            lo = mid
        else:
            hi = mid - 1
    return "".join(lines[:lo]) + QUESTION


def complete(name, prompt, sampler):
    body = dict(sampler, prompt=prompt, n_predict=256, cache_prompt=False, return_tokens=True)
    try:
        reply = post("/completion", body)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{ROUND}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    text = str(reply.get("content", ""))
    t = reply.get("timings") or {}
    row = [ROUND, name, str(t.get("prompt_n", "")), str(len(toks)),
           hashlib.md5(",".join(str(x) for x in toks).encode()).hexdigest() if toks else "",
           hashlib.md5(text.encode("utf-8")).hexdigest() if toks else "",
           str(t.get("draft_n", "")), str(t.get("draft_n_accepted", "")), str(reply.get("stop_type", "")),
           json.dumps(text[:60] if toks else str(reply.get("error", ""))[:60])]
    with open(os.path.join(OUT, "session.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row[:9]), flush=True)


os.makedirs(OUT, exist_ok=True)
deep = deep_prompt()
with open(os.path.join(OUT, f"{ROUND}-deep-prompt.md5"), "w") as f:
    f.write(hashlib.md5(deep.encode("utf-8")).hexdigest() + f"  {len(deep)} characters\n")
complete("code", P_CODE, GREEDY)
complete("prose", P_PROSE, GREEDY)
complete("deep", deep, GREEDY)
complete("deep-s", deep, PRESET)
complete("code-s", P_CODE, PRESET)
