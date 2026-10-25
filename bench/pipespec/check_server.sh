#!/usr/bin/env bash
# Checks llama-server's pipelined speculation (--pipe-depth, fork branch pascal-p106-v2-pipeserve) against itself
# without guesses. With --pipe-depth on, every sampled token's random draw is keyed by its position, so the output of a
# request must not depend on the depth or the guesser: each configuration below has to return, for every request, the
# same reasoning, content and token count as the reference (--pipe-depth 1 --pipe-guess none: one token in flight, no
# guess). At temperature 0 the output must also equal a server without --pipe-depth.
#
# Requests (one server per configuration, one slot): each prompt in PROMPTS as one user message through
# /v1/chat/completions, at temperature 0 and at the makers' sampling (T 1, top-p 0.95, top-k 20, min-p 0) with seeds
# SEEDS, NTOK tokens; one streamed request; one request with a stop string taken from the reference's output, so that
# a stop arrives with guesses in flight. The server log must show guesses, right and wrong ones ("pipelined = ...").
# FLOORS adds configurations with no guesser and --pipe-floor F (every guess a filler token, Q154).
# The list ends with a chat UI's "regenerate": the first prompt's T 0 request three times in a row, each re-reading a
# fully cached prompt (one prompt token evaluated again; on a sliding-window model a context checkpoint superseded).
# A server that dies is reported with its exit status (139: a segfault) and fails the check.
#
# The rig's scheduler mode: with two or more cards the scheduler keeps several copies of every split input (pipeline
# parallelism), and on compute 6.1 llama.cpp uploads the host inputs packed (LLAMA_INPUT_UPLOAD=2 by default there).
# To cover that combination on other GPUs, run once with LLAMA_INPUT_UPLOAD=2 in the environment and two devices in
# ARGS (--device CUDA0,CUDA0 makes two of one card).
#
# Usage (environment variables):
#   BIN=<dir with llama-server> MODEL=<gguf> [ARGS='-ngl 99 -fa on -c 8192'] [DEPTHS='3 5 8'] [GUESSES='cont copy']
#   [PROMPTS='prompts/chat.txt prompts/rewrite.txt prompts/edit.txt'] [SEEDS='1 2'] [NTOK=256] [PORT=8099]
#   [FLOORS='3'] [OUT=/tmp/pipespec_server] [KEEP_SPEED=0] bench/pipespec/check_server.sh
# The server runs with -bs (backend sampling) and -np 1. A model with sliding-window layers needs -ub of at least
# 2 × (depth + 1) in ARGS. Exit status 0 when every configuration matches.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=${ARGS:--ngl 99 -fa on -c 8192}
DEPTHS=${DEPTHS:-3 5 8}
GUESSES=${GUESSES:-cont copy}
PROMPTS=${PROMPTS:-$HERE/prompts/chat.txt $HERE/prompts/rewrite.txt $HERE/prompts/edit.txt}
SEEDS=${SEEDS:-1 2}
NTOK=${NTOK:-256}
PORT=${PORT:-8099}
OUT=${OUT:-/tmp/pipespec_server}
KEEP_SPEED=${KEEP_SPEED:-0}
FLOORS=${FLOORS-3}
mkdir -p "$OUT"

# run_config <name> <server args...>: starts the server, sends every request, writes $OUT/<name>.jsonl
run_config() {
    local name=$1
    shift
    # shellcheck disable=SC2086
    "$BIN/llama-server" -m "$MODEL" $ARGS -np 1 -bs --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" "$@" \
        > "$OUT/$name.log" 2>&1 &
    local srv=$!
    for _ in $(seq 1 900); do
        curl -sf "127.0.0.1:$PORT/health" > /dev/null && break
        kill -0 $srv 2> /dev/null || break
        sleep 2
    done
    python3 -I - "$PORT" "$NTOK" "$OUT/$name.jsonl" "$OUT/reference.jsonl" "$KEEP_SPEED" "$SEEDS" $PROMPTS <<'EOF'
import json, sys, urllib.request
port, ntok, out, ref_path, keep_speed, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5] == "1", sys.argv[6]
prompts = sys.argv[7:]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "samplers": ["top_k", "top_p", "min_p", "temperature"]}

def post(body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        if not body.get("stream"):
            return json.load(r)
        text, reasoning, n = "", "", 0
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
            if "timings" in d:
                n = d["timings"].get("predicted_n", n)
        return {"choices": [{"message": {"content": text, "reasoning_content": reasoning}}], "timings": {"predicted_n": n}}

reqs = []
for p in prompts:
    msgs = [{"role": "user", "content": open(p).read()}]
    reqs.append((p.split("/")[-1] + " T0", dict(messages=msgs, temperature=0.0, seed=1, max_tokens=ntok, **SAMPLING)))
    for s in seeds.split():
        reqs.append((p.split("/")[-1] + f" T1 seed {s}", dict(messages=msgs, temperature=1.0, seed=int(s), max_tokens=ntok, **SAMPLING)))
p0 = prompts[0]
msgs0 = [{"role": "user", "content": open(p0).read()}]
reqs.append((p0.split("/")[-1] + " T1 seed 1 streamed", dict(messages=msgs0, temperature=1.0, seed=1, max_tokens=ntok, stream=True, **SAMPLING)))

# a stop string from the middle of the reference's first T1 output, so that the stop lands with guesses in flight
stop = None
try:
    ref = [json.loads(l) for l in open(ref_path)]
    txt = next(r for r in ref if r["name"].endswith("T1 seed 1"))["content"]
    if len(txt) > 40:
        stop = txt[len(txt) // 2: len(txt) // 2 + 6]
except (FileNotFoundError, StopIteration):
    pass
if stop:
    reqs.append((p0.split("/")[-1] + " T1 seed 1 stop", dict(messages=msgs0, temperature=1.0, seed=1, max_tokens=ntok, stop=[stop], **SAMPLING)))

# regenerate: the same request three times, each on a fully cached prompt
for i in (1, 2, 3):
    reqs.append((p0.split("/")[-1] + f" T0 regenerate {i}", dict(messages=msgs0, temperature=0.0, seed=1, max_tokens=ntok, **SAMPLING)))

with open(out, "w") as f:
    for name, body in reqs:
        try:
            r = post(body)
        except Exception as e:
            print(f"request {name}: FAILED ({type(e).__name__}: {e})")
            sys.exit(1)
        m = r["choices"][0]["message"]
        t = r.get("timings", {})
        row = {"name": name, "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
               "n": t.get("predicted_n")}
        if keep_speed:
            row["tok_s"] = t.get("predicted_per_second")
        f.write(json.dumps(row) + "\n")
EOF
    local rc=$?
    # a server stopped here exits with 0 (or 143); any other status means it died on its own
    kill $srv 2> /dev/null; wait $srv 2> /dev/null
    local status=$?
    if [ $status -ne 0 ] && [ $status -ne 143 ]; then
        echo "$name: the server died (exit status $status; 139 = segfault), log: $OUT/$name.log"
        rc=1
    fi
    sed -i -E '/ time =|tokens per second|t\/s/d' "$OUT/$name.log"
    return $rc
}

compare() { # <name> <reference jsonl>: prints and returns the verdict
    python3 -I - "$OUT/$1.jsonl" "$2" "$1" <<'EOF'
import json, sys
a = {r["name"]: r for r in map(json.loads, open(sys.argv[1]))}
b = {r["name"]: r for r in map(json.loads, open(sys.argv[2]))}
same = [k for k in b if k in a and (a[k]["reasoning"], a[k]["content"], a[k]["n"]) == (b[k]["reasoning"], b[k]["content"], b[k]["n"])]
diff = [k for k in b if k not in same]
print(f"{sys.argv[3]:24} {len(same)} of {len(b)} requests identical" + (f"; differ: {', '.join(diff)}" if diff else ""))
sys.exit(1 if diff else 0)
EOF
}

fail=0
run_config reference --pipe-depth 1 --pipe-guess none || fail=1
cp "$OUT/reference.jsonl" "$OUT/reference.first.jsonl"
# the stop-string request needs the reference's output: run the reference again with it
run_config reference --pipe-depth 1 --pipe-guess none || fail=1
compare reference "$OUT/reference.first.jsonl" > /dev/null || { echo "the reference is not reproducible"; fail=1; }

for g in $GUESSES; do
    for d in $DEPTHS; do
        run_config "d$d-$g" --pipe-depth "$d" --pipe-guess "$g" || fail=1
        compare "d$d-$g" "$OUT/reference.jsonl" || fail=1
        grep -o "pipelined = .*" "$OUT/d$d-$g.log" | awk '{g += $3; h += $5; m += $7} END {printf "%24s guesses %d, right %d, wrong %d\n", "", g, h, m}'
    done
done

for f in $FLOORS; do
    run_config "floor$f" --pipe-depth "$f" --pipe-guess none --pipe-floor "$f" || fail=1
    compare "floor$f" "$OUT/reference.jsonl" || fail=1
    grep -o "pipelined = .*" "$OUT/floor$f.log" | awk '{g += $3; h += $5; m += $7} END {printf "%24s guesses %d, right %d, wrong %d\n", "", g, h, m}'
done

# temperature 0 is the same with and without --pipe-depth
run_config plain || fail=1
python3 -I - "$OUT/plain.jsonl" "$OUT/reference.jsonl" <<'EOF' || fail=1
import json, sys
a = {r["name"]: r for r in map(json.loads, open(sys.argv[1])) if " T0" in r["name"]}
b = {r["name"]: r for r in map(json.loads, open(sys.argv[2])) if " T0" in r["name"]}
diff = [k for k in b if k not in a or (a[k]["reasoning"], a[k]["content"], a[k]["n"]) != (b[k]["reasoning"], b[k]["content"], b[k]["n"])]
print(f"{'no --pipe-depth, T 0':24} {len(b) - len(diff)} of {len(b)} requests identical" + (f"; differ: {', '.join(diff)}" if diff else ""))
sys.exit(1 if diff else 0)
EOF
exit $fail
