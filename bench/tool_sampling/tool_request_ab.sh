#!/usr/bin/env bash
# Q55: what a Flint tool request loses without backend sampling. One llama-server with -bs, the same coding-agent
# request sent three ways: tool_choice "none" (GPU sampling, same prompt), "none" with "backend_sampling": false (host
# sampling, no grammar) and "auto" (host sampling with the lazy tool grammar, which switches -bs off). Temperature 0,
# then 1.0 with fixed seeds, at the makers' top-p 0.95 / top-k 20 / min-p 0. Prints tokens and tok/s per request and
# counts the server's "not compatible with grammar" warnings (one per "auto" request).
#
#   ssh rig "BIN=/opt/llama-builds/<build>/bin MODEL=/mnt/models2/flint/flint-S4L0.gguf GPUS=0,1,2,3,4,5 \
#       LLAMA_NO_MOE_DOWN_RESCALE=1 bash -s" < bench/tool_sampling/tool_request_ab.sh
#
# Results go to /tmp/tool_request_ab/<TAG>/ (summary.txt, results.jsonl, server.log). Host-RAM safety as in
# scripts/server_bench.sh: --cache-ram 0, --ctx-checkpoints 0 and a watchdog.
# Env: BIN, MODEL, TAG (default tools_ab), GPUS (default all nine), CTX (default 32768), MAX_TOKENS (default 8192),
# PORT (default 8089), SERVER_ARGS (extra llama-server flags).
set -u

BIN=${BIN:?set BIN}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:-tools_ab}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
CTX=${CTX:-32768}
MAX_TOKENS=${MAX_TOKENS:-8192}
PORT=${PORT:-8089}
SERVER_ARGS=${SERVER_ARGS:-}
OUT=/tmp/tool_request_ab/$TAG
mkdir -p "$OUT"
echo "model $MODEL, GPUs '$GPUS', context $CTX, $(date -Is)" | tee "$OUT/summary.txt"

CUDA_VISIBLE_DEVICES=$GPUS "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" -np 1 \
    --no-webui -c "$CTX" -b 512 -ub 128 -bs --cache-ram 0 --ctx-checkpoints 0 $SERVER_ARGS > "$OUT/server.log" 2>&1 \
    < /dev/null &
srv=$!
trap 'kill $srv 2> /dev/null; wait $srv 2> /dev/null' EXIT
(
    while kill -0 $srv 2> /dev/null; do
        avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
            echo "watchdog: MemAvailable ${avail} MiB, killing llama-server" | tee -a "$OUT/server.log" >&2
            kill -9 $srv; break
        fi
        sleep 0.5
    done
) &
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 $srv 2> /dev/null; then echo "server exited during load:"; tail -15 "$OUT/server.log"; exit 1; fi
    sleep 2
done

python3 -u - "$PORT" "$MAX_TOKENS" "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys, urllib.request
port, max_tokens, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": req}}}

TOOLS = [tool("bash", "Run a shell command", {"command": "string", "description": "string", "timeout": "number"},
              ["command", "description"]),
         tool("read", "Read a file", {"filePath": "string", "offset": "number", "limit": "number"}, ["filePath"]),
         tool("write", "Write a file", {"filePath": "string", "content": "string"}, ["filePath", "content"]),
         tool("edit", "Replace text in a file", {"filePath": "string", "oldString": "string", "newString": "string"},
              ["filePath", "oldString", "newString"])]
MSGS = [{"role": "system", "content": "You are a coding agent. Use the tools."},
        {"role": "user", "content": "Create src/stats.py with a RunningStats class (Welford's algorithm: push, extend, "
                                    "variance, stdev, merge) and a short demo under __main__. Write it with the write tool."}]

with open(f"{out}/results.jsonl", "w") as f:
    for temp, seed in [(0.0, 1), (1.0, 11), (1.0, 12)]:
        for tc, bs in [("none", True), ("none", False), ("auto", True)]:
            body = {"messages": MSGS, "tools": TOOLS, "tool_choice": tc, "backend_sampling": bs, "temperature": temp,
                    "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": seed, "max_tokens": max_tokens,
                    "cache_prompt": False}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            t, ch = r["timings"], r["choices"][0]
            calls = len(ch["message"].get("tool_calls") or [])
            f.write(json.dumps({"temp": temp, "seed": seed, "tool_choice": tc, "backend_sampling": bs,
                                "finish": ch.get("finish_reason"), "tool_calls": calls, "timings": t}) + "\n")
            print(f"T {temp} seed {seed:2d} tool_choice {tc:4s} backend_sampling {bs!s:5s}: {t['predicted_n']:5d} tok at "
                  f"{t['predicted_per_second']:6.2f} t/s ({1000 / t['predicted_per_second']:5.1f} ms), "
                  f"{calls} tool calls, {ch.get('finish_reason')}")
EOF
echo "grammar warnings (one per \"auto\" request): $(grep -c "not compatible with grammar" "$OUT/server.log")" \
    | tee -a "$OUT/summary.txt"
