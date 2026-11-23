#!/usr/bin/env python3
"""C962 (Q1692): C950's sampled request R14 sent again and again to one llama-server, fresh each time, in the variants
that tell apart the two possible causes of its two replies (logits that differ between loads, or draws that depend on
more than the seed and the tokens).

R14 is a plain-text turn of C950's coding-agent session (run/r14_prompt.txt: the prompt C950's session.py builds for
it from its earlier greedy replies, 7,705 characters), 256 tokens with ignore_eos, temperature 0.7, top-p 1, min-p 0,
seed 7. Every request here has cache_prompt false, so the server reads the whole prompt each time, the same way.

  k8          C950's settings, top-k 8: the backend chain top-k -> temperature -> dist, pipelined when the server has
              --pipe-depth (a request without n_probs)
  k8probs     the same with n_probs 8 and post_sampling_probs: the pipe stays off (n_probs), the same backend chain and
              the same draws, and each token comes with the probabilities the dist step used (sorted by the server)
  k8p95       top-k 8 with top-p 0.95, Flint's preset value: the top-p step sorts the candidates before the dist step
  k0          top-k 0 (no top-k): the dist step over the whole vocabulary, in vocabulary order
  greedyprobs temperature 0 with n_probs 8 (pre-sampling): sampled on the host, the top 8 of the raw softmax per token,
              to compare the logits between loads

Usage: python3 -I requests.py <server URL> <prompt file> <output folder> [plan]
plan: comma-separated kind:count (default k8:5,k8probs:2,k8p95:3,k0:3,greedyprobs:1), sent in that order.
Writes <out>/<kind>-<n>.json (the server's reply) and appends one line per request to <out>/requests.tsv: kind, n,
generated tokens, prompt tokens read, cached tokens, the md5 of the token ids (first 12 hex digits), the first 80
characters of the reply (JSON-escaped). Nothing timed is written.
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, PROMPT, OUT = sys.argv[1].rstrip("/"), open(sys.argv[2]).read(), sys.argv[3]
PLAN = sys.argv[4] if len(sys.argv) > 4 else "k8:5,k8probs:2,k8p95:3,k0:3,greedyprobs:1"

BASE = {"n_predict": 256, "ignore_eos": True, "temperature": 0.7, "top_k": 8, "top_p": 1.0, "min_p": 0.0, "seed": 7,
        "return_tokens": True, "cache_prompt": False}
KINDS = {
    "k8": {},
    "k8probs": {"n_probs": 8, "post_sampling_probs": True},
    "k8p95": {"top_p": 0.95},
    "k0": {"top_k": 0},
    "greedyprobs": {"temperature": 0.0, "n_probs": 8},
}

os.makedirs(OUT, exist_ok=True)
for item in PLAN.split(","):
    kind, count = item.split(":")
    for n in range(1, int(count) + 1):
        body = dict(BASE, prompt=PROMPT, **KINDS[kind])
        req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                reply = json.load(r)
        except Exception as e:  # recorded, so a failed request shows in the table
            reply = {"error": str(e)}
        with open(os.path.join(OUT, f"{kind}-{n}.json"), "w") as f:
            json.dump(reply, f)
        toks = reply.get("tokens") or []
        md5 = hashlib.md5(json.dumps(toks, separators=(",", ":")).encode()).hexdigest()[:12]
        t = reply.get("timings") or {}
        row = [kind, str(n), str(len(toks)), str(t.get("prompt_n", "")), str(t.get("cache_n", "")), md5,
               json.dumps(str(reply.get("content", reply.get("error", "")))[:80])]
        with open(os.path.join(OUT, "requests.tsv"), "a") as f:
            f.write("\t".join(row) + "\n")
        print(" ".join(row[:6]), flush=True)
