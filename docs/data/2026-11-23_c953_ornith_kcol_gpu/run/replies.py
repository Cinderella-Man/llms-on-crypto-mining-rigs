#!/usr/bin/env python3
"""C953 (Q1687 point 7): greedy replies from llama-server's /completion endpoint, for comparing two settings of
GGML_CUDA_MMVQ_PASCAL_MT token by token. Three prompts (C854b's coding request and prose paragraph, and a request that
asks for a longer program), each sent ROUNDS times as the preset serves it (temperature 0, top-k 1, 256 tokens, no
prompt cache, the token ids returned), then once with n_probs 2: the server then samples on the host and returns each
token's two most likely candidates with their log-probabilities before sampling, which sizes the first difference
between two settings (the gap between the two candidates where the replies part) and shows whether a setting changed
the numbers at all (the log-probabilities of the same tokens). Nothing timed is written.

Usage: python3 -I replies.py <server URL> <output folder> [ROUNDS (2)]
Writes <out>/<round>-<prompt>.json (the server's reply) and appends one line per request to <out>/replies.tsv: round
(r1, r2 ... or p for the n_probs request), prompt, generated tokens, md5 of the token ids, md5 of the text, drafted
tokens, accepted drafts, the stop reason, the reply's first 60 characters (JSON-escaped).
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

PROMPTS = {
    "code": "Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest "
            "tests for it.",
    "prose": "Explain in one paragraph how a hash map handles collisions, then give one example in Python.",
    "prog": "Write a complete Elixir GenServer that keeps a bounded LRU cache with get, put and size calls, with "
            "comments, then show how to start it under a supervisor.",
}
BODY = {"n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False, "return_tokens": True}


def post(body, timeout=1800):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def md5(x):
    return hashlib.md5(x.encode()).hexdigest()


def one(tag, name, extra):
    body = dict(BODY, prompt=PROMPTS[name], **extra)
    try:
        reply = post(body)
    except Exception as e:  # noqa: BLE001 - recorded, the run goes on
        reply = {"error": repr(e)}
    with open(os.path.join(OUT, f"{tag}-{name}.json"), "w", encoding="utf-8") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    t = reply.get("timings") or {}
    line = [tag, name, str(len(toks)), md5(json.dumps(toks)), md5(reply.get("content", "")),
            str(t.get("draft_n", "")), str(t.get("draft_n_accepted", "")), str(reply.get("stop_type", "")),
            json.dumps(reply.get("content", reply.get("error", ""))[:60])]
    with open(os.path.join(OUT, "replies.tsv"), "a", encoding="utf-8") as f:
        f.write("\t".join(line) + "\n")


os.makedirs(OUT, exist_ok=True)
for r in range(1, ROUNDS + 1):
    for name in PROMPTS:
        one(f"r{r}", name, {})
for name in PROMPTS:
    one("p", name, {"n_probs": 2})
