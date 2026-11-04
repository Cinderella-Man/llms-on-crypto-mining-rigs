#!/usr/bin/env python3
# lightprof_requests.py PORT ARM OUT_TSV [N_WARM N_TOK REPS]: the requests of one C142 server line (Q681), sent to a
# llama-server on this machine: one warm-up request of N_WARM tokens (16), then REPS (3) requests of N_TOK tokens (64),
# all the same prompt, greedy, end-of-text ignored, no prompt cache, so every arm decodes the same tokens from the same
# state. Appends one TSV line per request (arm, rep, prompt and decode counts and times from the server's timings,
# its ms per token, the client's seconds, the output's md5 prefix) and prints it. Standard library only.
import hashlib, json, sys, time, urllib.request

port, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_warm, n_tok, reps = (int(x) for x in (sys.argv[4:7] + ["16", "64", "3"][len(sys.argv[4:7]):]))
PROMPT = ("You are reviewing a small Python module that parses a configuration file of key = value lines. "
          "Explain, step by step, how you would make it reject duplicate keys, keep comments, and report the "
          "line number of every error. Then write the new function.\n\n")


def ask(n):
    body = dict(prompt=PROMPT, n_predict=n, temperature=0, top_k=1, seed=1, ignore_eos=True, cache_prompt=False)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read())
    return res, time.time() - t0


with open(out, "a") as f:
    for rep in ["warm"] + [str(i) for i in range(1, reps + 1)]:
        res, secs = ask(n_warm if rep == "warm" else n_tok)
        t = res.get("timings", {})
        line = [arm, rep, t.get("prompt_n", ""), f"{t.get('prompt_ms', 0):.1f}", t.get("predicted_n", ""),
                f"{t.get('predicted_ms', 0):.1f}", f"{t.get('predicted_per_token_ms', 0):.2f}", f"{secs:.2f}",
                hashlib.md5(res.get("content", "").encode()).hexdigest()[:12]]
        print("\t".join(str(x) for x in line), file=f, flush=True)
        print("\t".join(str(x) for x in line), flush=True)
