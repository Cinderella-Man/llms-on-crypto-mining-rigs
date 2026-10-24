#!/usr/bin/env python3
"""Served requests for the release matrix (bench/release_matrix/README.md), against a llama-server on 127.0.0.1.

  client.py PORT greedy N PROMPT_FILE            one request at temperature 0, N tokens; prints one JSON line with the
                                                 SHA-256 of the reply (reasoning and answer) and its token count
  client.py PORT chat N SEEDS PROMPT_FILE...     one request per prompt file and seed (SEEDS: "1 2 3"), N tokens at
                                                 most, the makers' sampling (T 1.0, top-p 0.95, top-k 20, min-p 0);
                                                 prints one JSON line per request with the server's own timings

Speeds are the server's `timings.predicted_per_second`, which counts every generated token, reasoning included.
Standard library only, so it runs on the rig's system Python. Exit status: 0 done, 1 a request failed.
"""
import hashlib
import json
import sys
import urllib.request

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def ask(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def reply_text(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n</reasoning>\n" + (m.get("content") or "")


def main():
    if len(sys.argv) < 5 or sys.argv[2] not in ("greedy", "chat"):
        print(__doc__, file=sys.stderr)
        return 2
    port, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == "greedy":
        prompt = open(sys.argv[4]).read().strip()
        r = ask(port, {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "seed": 1,
                       "max_tokens": n})
        t = r.get("timings", {})
        print(json.dumps({"name": "greedy", "sha256": hashlib.sha256(reply_text(r).encode()).hexdigest(),
                          "predicted_n": t.get("predicted_n"), "tok_s": t.get("predicted_per_second")}), flush=True)
        return 0
    seeds, files = sys.argv[4].split(), sys.argv[5:]
    for f in files:
        prompt = open(f).read().strip()
        for s in seeds:
            name = f"{f.rsplit('/', 1)[-1]} seed {s}"
            try:
                r = ask(port, {"messages": [{"role": "user", "content": prompt}], "seed": int(s), "max_tokens": n,
                               **SAMPLING})
            except Exception as e:  # report and stop: a failed request makes the median meaningless
                print(json.dumps({"name": name, "error": str(e)}), flush=True)
                return 1
            t = r.get("timings", {})
            print(json.dumps({"name": name, "predicted_n": t.get("predicted_n"), "tok_s": t.get("predicted_per_second"),
                              "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
