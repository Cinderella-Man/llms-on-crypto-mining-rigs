#!/usr/bin/env python3
"""kreal_req.py: the requests of kreal.sh. Tokenizes TEXT through the server (/tokenize), keeps the first N tokens, and
per k sends them as one /completion prompt (token ids, n_predict 1, no cache reuse) whose calls run in micro-batches
of k tokens ("ubatch_taper": "short=1000000:k,min=1"). Appends per k: arm, k, prompt_n, prompt_ms, micro-batches,
ms per micro-batch (T(k)) to --out. Standard library only."""
import argparse, json, os, sys, urllib.request

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--text", required=True)
ap.add_argument("--n", type=int, required=True); ap.add_argument("--ks", type=int, nargs="+", required=True)
ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
toks = post(a.port, "/tokenize", {"content": open(a.text, encoding="utf-8").read(), "add_special": True})["tokens"]
if len(toks) < a.n:
    sys.exit(f"the text has {len(toks)} tokens, fewer than {a.n}")
toks = toks[:a.n]
new = not os.path.exists(a.out)
with open(a.out, "a") as f:
    if new:
        f.write("arm\tk\tprompt_n\tprompt_ms\tmicro_batches\tms_per_micro_batch\n")
    for k in a.ks:
        r = post(a.port, "/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": False, "temperature": 0,
                                         "ubatch_taper": f"short=1000000:{k},min=1"})
        t = r.get("timings", {})
        n, ms = t.get("prompt_n", 0), t.get("prompt_ms", float("nan"))
        mb = -(-n // k) if n else 0
        f.write(f"{a.arm}\t{k}\t{n}\t{ms:.1f}\t{mb}\t{ms / mb if mb else float('nan'):.2f}\n")
        f.flush()
        print(f"{a.arm} k={k}: {n} tokens in {ms:.0f} ms, {mb} micro-batches, {ms / mb if mb else float('nan'):.1f} ms each", flush=True)
