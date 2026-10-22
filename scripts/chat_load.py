#!/usr/bin/env python3
"""Load generator for one or more llama-server instances.

Sends the same chat prompts at temperature 0 and reports each request's generation speed (from the server's own
timings) and the combined throughput (all generated tokens / wall-clock time).

    chat_load.py --endpoints 127.0.0.1:8089 127.0.0.1:8090 --mode alone      # each server on its own, one at a time
    chat_load.py --endpoints 127.0.0.1:8089 127.0.0.1:8090 --mode together   # one request per server, all at once
    chat_load.py --endpoints 127.0.0.1:8089 --mode together --per-endpoint 2 # two concurrent requests on one server
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with type hints and three unit tests.",
    "Explain to a physicist how the attention mechanism in a transformer works, in about 300 words.",
]


def chat(endpoint, prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://{endpoint}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    t = r.get("timings", {})
    return {"endpoint": endpoint, "n": t.get("predicted_n", 0), "tps": t.get("predicted_per_second", 0),
            "wall": time.time() - t0}


def run_batch(jobs, max_tokens):
    results, threads = [], []
    lock = threading.Lock()

    def worker(endpoint, prompt):
        res = chat(endpoint, prompt, max_tokens)
        with lock:
            results.append(res)

    t0 = time.time()
    for endpoint, prompt in jobs:
        th = threading.Thread(target=worker, args=(endpoint, prompt))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - t0
    for r in sorted(results, key=lambda r: r["endpoint"]):
        print(f"  {r['endpoint']}: {r['n']} tokens at {r['tps']:.2f} t/s")
    total = sum(r["n"] for r in results)
    print(f"  combined: {total} tokens in {wall:.1f} s = {total / wall:.2f} t/s")
    return total / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--mode", choices=["alone", "together"], required=True)
    ap.add_argument("--per-endpoint", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    for prompt in PROMPTS:
        print(f"prompt: {prompt[:60]}...")
        if a.mode == "alone":
            for ep in a.endpoints:
                run_batch([(ep, prompt)], a.max_tokens)
        else:
            run_batch([(ep, prompt) for ep in a.endpoints for _ in range(a.per_endpoint)], a.max_tokens)


if __name__ == "__main__":
    main()
