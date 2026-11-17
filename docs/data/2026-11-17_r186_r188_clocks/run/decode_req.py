#!/usr/bin/env python3
"""decode_req.py (R186, R188): one decode request to llama-server's /completion, timed by the server.

  python3 -I decode_req.py URL TAG N_PREDICT >> requests.tsv
Prints one tab-separated line: tag, client start and end (epoch s), prompt tokens and ms, predicted tokens, ms and
ms a token. The prompt is fixed and short, the reply runs N_PREDICT tokens (ignore_eos), seed 42.
"""
import json
import sys
import time
import urllib.request

url, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROMPT = ("Write a Python module that implements a red-black tree with insert, delete, search and in-order "
          "traversal. Explain each method in its docstring.\n\n```python\n")
body = json.dumps({"prompt": PROMPT, "n_predict": n, "ignore_eos": True, "seed": 42, "cache_prompt": True}).encode()
req = urllib.request.Request(url.rstrip("/") + "/completion", data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.load(r)
t1 = time.time()
t = out.get("timings", {})
print("\t".join(str(x) for x in (tag, f"{t0:.3f}", f"{t1:.3f}", t.get("prompt_n"), round(t.get("prompt_ms", 0), 2),
                                 t.get("predicted_n"), round(t.get("predicted_ms", 0), 2),
                                 round(t.get("predicted_per_token_ms", 0), 3))), flush=True)
