#!/usr/bin/env python3
"""C962 (Q1692): C950's session (docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py) up to R14, with R14 sent
in one of five variants, so that R14's probabilities and draws can be read in the session's own context: the prompt
cache that R1 to R13 built (R14 reads 14 new tokens on top of 2,276 cached ones), not a fresh read of R14's 2,290
tokens.

  pipe       R14 as C950 sent it (pipelined, no n_probs)
  probs      R14 with n_probs 8 and post_sampling_probs: the pipe stays off, the same backend chain and draws, and each
             token comes with the dist step's probabilities
  greedy     R14 at temperature 0 with n_probs 8 (pre-sampling: sampled on the host, the top 8 of the raw softmax)
  p95        R14 with top-p 0.95 (Flint's preset value): the top-p step sorts the candidates before the dist step
  k0         R14 with top-k 0: the dist step over the whole vocabulary, in vocabulary order

R1 has no prompt cache, so each call of this script rebuilds the slot's cache from the start the same way. The session's
own requests are C950's (its session.py is run up to R14 with this script's complete(); R15, R10 and R11 are not sent).

Usage: python3 -I session_r14.py <server URL> <C950's session.py> <output folder> <tag> <variant>
Writes <out>/<tag>-<request>.json and appends one line per request to <out>/session_r14.tsv: tag, variant, request,
generated tokens, prompt tokens read, cached tokens, the md5 of the token ids (first 12 hex digits).
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, SESSION, OUT, TAG, VARIANT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
R14_EXTRA = {"pipe": {}, "probs": {"n_probs": 8, "post_sampling_probs": True},
             "greedy": {"temperature": 0.0, "n_probs": 8}, "p95": {"top_p": 0.95}, "k0": {"top_k": 0}}[VARIANT]

src = open(SESSION).read()
head, body = src.split("os.makedirs(OUT, exist_ok=True)", 1)
body = body.split("def tokens(text):", 1)[0]
g = {"__name__": "session"}
# C950's constants (SCHEMA, prompts, files); its URL/OUT/ROUND line reads sys.argv, which this script sets aside
exec(head.replace('URL, OUT, ROUND = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]', 'URL = OUT = ROUND = ""'), g)


class Done(Exception):
    pass


def complete(name, kind, prompt, n, temp=0.0, seed=1, schema=True, cache=False, ignore_eos=False):
    """C950's complete() (the same request body), with R14's variant fields added; stops after R14"""
    body = {
        "prompt": prompt, "n_predict": n, "temperature": temp, "top_k": 8, "top_p": 1.0, "min_p": 0.0,
        "seed": seed, "return_tokens": True, "cache_prompt": cache,
    }
    if schema:
        body["json_schema"] = g["SCHEMA"]
    if ignore_eos:
        body["ignore_eos"] = True
    if name == "R14":
        body.update(R14_EXTRA)
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            reply = json.load(r)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{TAG}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    md5 = hashlib.md5(json.dumps(toks, separators=(",", ":")).encode()).hexdigest()[:12]
    t = reply.get("timings") or {}
    row = [TAG, VARIANT, name, str(len(toks)), str(t.get("prompt_n", "")), str(t.get("cache_n", "")), md5]
    with open(os.path.join(OUT, "session_r14.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row), flush=True)
    if name == "R14":
        raise Done
    return str(reply.get("content", ""))


g["complete"] = complete
os.makedirs(OUT, exist_ok=True)
try:
    exec(body, g)
except Done:
    pass
