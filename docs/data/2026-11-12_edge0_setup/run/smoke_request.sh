#!/usr/bin/env bash
# One chat request to a running Edge0 server: prints the reply, the token counts and the engine's own timings.
# A check that the model loads and answers, not a benchmark: one request, whatever else the machine is doing.
#
#   PORT=8000 MODEL=edge0_145b bash smoke_request.sh ["a prompt"]
#
# Optional: MAX_TOKENS (default 400), OUT (a file for the raw JSON answer).
set -eu
PORT=${PORT:-8000}; MODEL=${MODEL:-edge0_145b}; MAX_TOKENS=${MAX_TOKENS:-400}; OUT=${OUT:-}
PROMPT=${1:-Write a Python function that returns the n-th Fibonacci number iteratively, then explain it in two sentences.}
BODY=$(python3 -I -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": int(sys.argv[2]), "temperature": 0,
  "messages": [{"role": "user", "content": sys.argv[3]}]}))' "$MODEL" "$MAX_TOKENS" "$PROMPT")
T0=$(date +%s.%N)
ANSWER=$(curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")
T1=$(date +%s.%N)
[ -z "$OUT" ] || printf '%s\n' "$ANSWER" > "$OUT"
printf '%s' "$ANSWER" | python3 -I -c '
import json, sys
wall = float(sys.argv[1])
try:
    a = json.load(sys.stdin)
except Exception as e:
    print("not JSON:", e); sys.exit(1)
if "choices" not in a:
    print("no choices:", json.dumps(a)[:600]); sys.exit(1)
m = a["choices"][0]["message"]
r = m.get("reasoning_content") or ""
print("finish_reason:", a["choices"][0].get("finish_reason"))
print("reasoning chars:", len(r), "| answer chars:", len(m.get("content") or ""))
print("--- answer"); print((m.get("content") or "")[:900])
print("--- usage:", json.dumps(a.get("usage")))
print("--- timings:", json.dumps(a.get("timings")))
print("--- wall seconds: %.2f" % wall)
' "$(echo "$T1 - $T0" | bc)"
