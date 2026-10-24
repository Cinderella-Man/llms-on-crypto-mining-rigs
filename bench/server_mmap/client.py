# One chat request against llama-server, non-streaming or streaming; prints a JSON summary line (Q118, Q119).
# Usage: client.py nostream|stream [port] [max_tokens]
import json, sys, time, urllib.request
stream = sys.argv[1] == "stream"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8089
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
body = {
    "messages": [{"role": "user", "content": "Write a detailed technical explanation of how a B-tree handles insertion, deletion and node splits, with a worked example and Python code."}],
    "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 42,
    "stream": stream,
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
out = {"stream": stream}
with urllib.request.urlopen(req, timeout=600) as r:
    if not stream:
        d = json.loads(r.read())
        out["timings"] = d.get("timings")
        out["usage"] = d.get("usage")
    else:
        arrivals, timings, n = [], None, 0
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            arrivals.append(time.time())
            d = json.loads(payload)
            if d.get("timings"):
                timings = d["timings"]
            n += 1
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        gaps.sort()
        out["timings"] = timings
        out["chunks"] = n
        if gaps:
            out["chunk_gap_ms_mean"] = 1000 * sum(gaps) / len(gaps)
            out["chunk_gap_ms_median"] = 1000 * gaps[len(gaps) // 2]
            out["chunk_gap_ms_p95"] = 1000 * gaps[int(0.95 * (len(gaps) - 1))]
out["wall_s"] = time.time() - t0
print(json.dumps(out))
