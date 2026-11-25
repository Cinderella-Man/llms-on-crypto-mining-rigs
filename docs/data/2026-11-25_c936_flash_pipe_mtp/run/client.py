# C936 (Q1675): the requests sent to one llama-server load, in order, one JSON line each (token ids, text, stop reason,
# the server's timings, the client's start and end). Usage: python3 -I client.py <port> <out.jsonl> [probs|probsall]
# With "probs" a second copy of the 256-token reply asks for the top two tokens at every position (n_probs 2); it runs
# only on the reference loads, since token probabilities take a slot out of the pipe. With "probsall" the 512-token and
# JSON-schema replies ask for the top two tokens as well (the extra plain load, plain_probs).
import json
import sys
import time
import urllib.request

port, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else ""
probs = mode in ("probs", "probsall")
extra = {"n_probs": 2} if mode == "probsall" else {}
base = f"http://127.0.0.1:{port}"

CODE = "Write a Python class implementing an LRU cache with a per-entry time-to-live, with unit tests."
CHAT = "Explain how a Bloom filter works, when it is the right tool, and when it is not."
SCHEMA_Q = ("A small Python project implements an LRU cache with a per-entry time-to-live. Name the project and list "
            "the files it needs, as JSON.")
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"},
                         "files": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6}},
          "required": ["name", "files"]}


def post(path, body, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def template(text):
    return post("/apply-template", {"messages": [{"role": "user", "content": text}]}, 60)["prompt"]


greedy = {"temperature": 0.0, "cache_prompt": False, "return_tokens": True, "stream": False}
p_code, p_chat, p_schema = template(CODE), template(CHAT), template(SCHEMA_Q)
reqs = [("reply256", dict(greedy, prompt=p_code, n_predict=256), 600)]
if probs:
    reqs.append(("reply256_probs", dict(greedy, prompt=p_code, n_predict=256, n_probs=2), 600))
reqs.append(("reply512", dict(greedy, prompt=p_chat, n_predict=512, **extra), 900))
reqs.append(("schema", dict(greedy, prompt=p_schema, n_predict=256, json_schema=SCHEMA, **extra), 300))

with open(out, "w") as f:
    f.write(json.dumps({"req": "prompts", "code": p_code, "chat": p_chat, "schema": p_schema,
                        "json_schema": SCHEMA}) + "\n")
    for name, body, timeout in reqs:
        t0 = time.time()
        row = {"req": name, "t_start": t0}
        try:
            d = post("/completion", body, timeout)
            row.update(content=d.get("content"), tokens=d.get("tokens"), n=d.get("tokens_predicted"),
                       stop_type=d.get("stop_type"), timings=d.get("timings"))
            if "completion_probabilities" in d:
                row["top2"] = [[(c.get("id"), c.get("token"), c.get("logprob")) for c in p.get("top_logprobs", [])]
                               for p in d["completion_probabilities"]]
        except Exception as e:  # a timed-out or failed request is a result too (a hang shows as a timeout)
            row["error"] = repr(e)
        row["t_end"] = time.time()
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"{name}: {row.get('n')} tokens, {row.get('stop_type')}, {row['t_end'] - t0:.1f} s"
              + (f", error {row['error']}" if "error" in row else ""), flush=True)
        if "error" in row:
            break  # the server may be stuck: the driver stops it
