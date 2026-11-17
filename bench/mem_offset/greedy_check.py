#!/usr/bin/env python3
"""Greedy output of a running llama-server, hashed, and compared with a reference step (bench/mem_offset: the
per-card check of climb.py; first used in the Q133/Q135 clock-offset sitting).

Sends fixed prompts to /completion at temperature 0 with the top-10 token probabilities of every generated token and
no prompt cache, so each request recomputes its prompt. The hash covers the generated text and every probability as
the server printed it (timings excluded), so a single changed logit anywhere changes it.

  greedy_check.py --url http://127.0.0.1:8133 --out DIR --tag TAG [--ref TAG] [--n 256]

Writes DIR/greedy-TAG.json (the responses) and DIR/greedy-TAG.sha. With --ref, exit 0 when the hash equals the
reference's, 1 when it differs; 3 when a request fails. Prints one line: hash, tokens, prompt and generation tok/s.
"""
import argparse
import hashlib
import json
import sys
import urllib.request

PROMPTS = [
    "The memory controller of a graphics card checks every burst it reads. Explain, step by step, how a CRC on the "
    "data bus detects a transmission error and what the controller does next:",
    "Write a Python function that parses a CSV file of GPU telemetry (time, card, power, temperature) and returns, "
    "for each card, the highest temperature and the average power. Then explain how you would test it.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8133")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ref")
    p.add_argument("--n", type=int, default=256)
    a = p.parse_args()
    results, h = [], hashlib.sha256()
    pp, tg, ntok = [], [], 0
    for prompt in PROMPTS:
        body = {"prompt": prompt, "n_predict": a.n, "temperature": 0, "top_k": 1, "seed": 1, "n_probs": 10,
                "cache_prompt": False, "stream": False}
        req = urllib.request.Request(a.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"greedy {a.tag}: request failed: {e}")
            return 3
        keep = {"content": resp.get("content"), "completion_probabilities": resp.get("completion_probabilities")}
        h.update(json.dumps(keep, sort_keys=True).encode())
        t = resp.get("timings", {})
        pp.append(t.get("prompt_per_second", 0.0))
        tg.append(t.get("predicted_per_second", 0.0))
        ntok += t.get("predicted_n", 0)
        results.append({"prompt": prompt, "response": keep, "timings": t})
    digest = h.hexdigest()[:16]
    json.dump(results, open(f"{a.out}/greedy-{a.tag}.json", "w"), indent=1)
    open(f"{a.out}/greedy-{a.tag}.sha", "w").write(digest + "\n")
    line = (f"greedy {a.tag}: hash {digest}, {ntok} tokens, prompt tok/s "
            f"{' '.join(f'{x:.2f}' for x in pp)}, generation tok/s {' '.join(f'{x:.2f}' for x in tg)}")
    if a.ref:
        ref = open(f"{a.out}/greedy-{a.ref}.sha").read().strip()
        same = ref == digest
        print(line + (f" | SAME as {a.ref}" if same else f" | DIFFERS from {a.ref} ({ref})"))
        return 0 if same else 1
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
