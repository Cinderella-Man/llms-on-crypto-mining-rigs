# Q55 client: the same tools request with tool_choice none (backend sampling on / off) and auto, T 0 and T 1.0.
import json, sys, urllib.request
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": req}}}
TOOLS = [tool("bash", "Run a shell command", {"command": "string", "description": "string", "timeout": "number"}, ["command", "description"]),
         tool("read", "Read a file", {"filePath": "string", "offset": "number", "limit": "number"}, ["filePath"]),
         tool("write", "Write a file", {"filePath": "string", "content": "string"}, ["filePath", "content"]),
         tool("edit", "Replace text in a file", {"filePath": "string", "oldString": "string", "newString": "string"},
              ["filePath", "oldString", "newString"])]
MSGS = [{"role": "system", "content": "You are a coding agent. Use the tools."},
        {"role": "user", "content": "Create src/stats.py with a RunningStats class (Welford's algorithm: push, extend, "
                                    "variance, stdev, merge) and a short demo under __main__. Write it with the write tool."}]
for temp, seed in [(0.0, 1), (1.0, 11), (1.0, 12)]:
    for tc, bs in [("none", True), ("none", False), ("auto", True)]:
        body = {"messages": MSGS, "tools": TOOLS, "tool_choice": tc, "backend_sampling": bs, "temperature": temp,
                "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": seed, "max_tokens": max_tokens, "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        t = d["timings"]
        print(json.dumps({"temp": temp, "seed": seed, "tool_choice": tc, "backend_sampling": bs,
                          "predicted_n": t["predicted_n"], "predicted_per_second": t["predicted_per_second"],
                          "predicted_per_token_ms": t["predicted_per_token_ms"],
                          "finish_reason": d["choices"][0].get("finish_reason")}), flush=True)
