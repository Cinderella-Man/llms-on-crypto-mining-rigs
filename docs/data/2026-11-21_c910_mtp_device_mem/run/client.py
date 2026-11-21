# C910 (from C908, which took it from pull request 7's review, unchanged): the server check's client. Sends one
# request set to a llama-server on 127.0.0.1 and writes one JSON line per request (the content, the token ids, the
# count and the stop reason). Usage: python3 -I client.py <port> <out.jsonl>
import json
import sys
import urllib.request

port, out = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{port}/completion"
schema = {"type": "object", "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
          "required": ["name", "count"]}
prompt = "Write a short story about a robot who learns to paint. Once upon a time"
greedy = {"temperature": 0.0}
makers = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": 7}
requests = [
    ("t0", dict(greedy, n_predict=96)),
    ("t1-seed7", dict(makers, n_predict=96)),
    ("schema-t0", dict(greedy, n_predict=64, json_schema=schema)),
    ("schema-t1-seed7", dict(makers, n_predict=64, json_schema=schema)),
    ("t0-again", dict(greedy, n_predict=96)),
]
with open(out, "w") as f:
    for name, body in requests:
        body = dict(body, prompt=prompt, cache_prompt=False, return_tokens=True)
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            row = {"req": name, "content": d.get("content"), "tokens": d.get("tokens"),
                   "n": d.get("tokens_predicted"), "stop_type": d.get("stop_type")}
        except Exception as e:  # a failed request is a result too
            row = {"req": name, "error": repr(e)}
        f.write(json.dumps(row) + "\n")
        f.flush()
