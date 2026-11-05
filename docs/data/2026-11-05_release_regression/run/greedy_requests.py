#!/usr/bin/env python3
"""Greedy requests to a llama-server on this machine's loopback, for an identity check between engine builds.

For every prompt of PROMPTS (JSONL: name, messages, optional attach = a file whose text is appended to the last
message, read from ATTACH_DIR by its base name) the server renders the chat template (/apply-template), then the
rendered prompt is sent twice to /completion at temperature 0 with N tokens and no prompt cache:
  pipe   n_probs 0: the server's own decoding path, pipelined speculation included when the line enables it
  probs  n_probs 2: the top-2 log-probabilities of every generated token (llama-server decodes such a request plainly)
One JSON line per request goes to OUT: name, mode, the rendered prompt's sha1 and token count, the generated token ids,
the text, the stop type, the server's timings and, for probs, [[id, logprob], [id, logprob]] per token.
Standard library only; it runs on the rig next to the server.
usage: greedy_requests.py PORT PROMPTS N OUT [ATTACH_DIR]
"""
import hashlib
import json
import os
import sys
import time
import urllib.request


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    port, prompts, n, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    adir = sys.argv[5] if len(sys.argv) > 5 else os.path.dirname(os.path.abspath(prompts))
    with open(prompts) as f, open(out, "w") as o:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            msgs = [dict(m) for m in p["messages"]]
            if p.get("attach"):
                with open(os.path.join(adir, os.path.basename(p["attach"]))) as a:
                    msgs[-1]["content"] += "\n\n```\n" + a.read() + "\n```"
            prompt = post(port, "/apply-template", {"messages": msgs})["prompt"]
            sha = hashlib.sha1(prompt.encode()).hexdigest()
            for mode, nprobs in (("pipe", 0), ("probs", 2)):
                body = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False,
                        "return_tokens": True, "n_probs": nprobs, "seed": 1}
                t0 = time.time()
                r = post(port, "/completion", body)
                tm = r.get("timings", {})
                rec = {"name": p["name"], "mode": mode, "prompt_sha1": sha, "prompt_n": tm.get("prompt_n"),
                       "tokens": r.get("tokens"), "content": r.get("content"), "stop_type": r.get("stop_type"),
                       "timings": tm, "wall_s": round(time.time() - t0, 3)}
                if nprobs:
                    rec["probs"] = [[[t["id"], t["logprob"]] for t in e.get("top_logprobs", [])]
                                    for e in r.get("completion_probabilities", [])]
                o.write(json.dumps(rec) + "\n")
                o.flush()
                print(f"{p['name']} {mode}: {len(rec['tokens'] or [])} tokens, "
                      f"{tm.get('predicted_per_second', 0):.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
