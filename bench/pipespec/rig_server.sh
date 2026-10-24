#!/usr/bin/env bash
# llama-server's pipelined speculation (--pipe-depth) on the rig: one server per depth, the same requests, their speed
# from the server's timings, and identity between the depths. Build: fork branch pascal-p106-v2-pipeserve (cd49e507c),
# which includes the window-cache fix for depths >= 4 (fbb1b24f3), so no depth cap is needed.
#
# Per depth in DEPTHS (0 = off, the reference) one server (-bs, -np 1), and these requests, each at temperature 0 and
# at the makers' sampling (T 1, top-p 0.95, top-k 20, min-p 0, seed SEED), with prompt caching off:
#   - chat: prompts/chat.txt, code.txt and edit.txt as one user message each, MAX_TOKENS tokens;
#   - tool: the coding-agent request of bench/tool_sampling/tool_request_ab.sh (four tools, tool_choice auto, Flint's
#     lazy tool grammar), TOOL_MAX_TOKENS tokens. On pascal-p106-v2-pipeserve a request with a grammar decodes batched
#     at every depth; the pascal-p106-v2-pipeserve-tools build pipelines it until the tool call starts.
# Identity: at temperature 0 every depth must return the reference's text (reasoning, content, calls). At temperature 1
# the depths >= 1 must agree with each other: their random draws are keyed by position, while depth 0 draws as the
# server always does (another output with the same distribution).
#
# Run it on the rig, from this repository (~35 min on nine cards: four model loads and 16 requests per depth):
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5,6,7,8 \
#       [TS=6,6,6,6,6,6,6,5,2] [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1'] [DEPTHS='0 1 2 3'] [GUESS=cont] [CTX=32768] \
#       [MAX_TOKENS=768] [TOOL_MAX_TOKENS=2048] [SEED=1] bash /tmp/pipespec/rig_server.sh \
#       > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# Output in /tmp/pipespec/out/<TAG>/: server-d<D>.log, results.jsonl (one line per request and depth), summary.tsv
# (tok/s per request and depth, the speed-up over depth 0, guesses right and wrong, identity), printed at the end.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
DEPTHS=${DEPTHS:-0 1 2 3}
GUESS=${GUESS:-cont}
CTX=${CTX:-32768}
MAX_TOKENS=${MAX_TOKENS:-768}
TOOL_MAX_TOKENS=${TOOL_MAX_TOKENS:-2048}
SEED=${SEED:-1}
PORT=${PORT:-8089}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"
: > "$OUT/results.jsonl"
echo "$(date -Is) start $TAG: depths $DEPTHS, GPUs $GPUS, model $MODEL"

for d in $DEPTHS; do
    pipe=""
    [ "$d" -gt 0 ] && pipe="--pipe-depth $d --pipe-guess $GUESS"
    # shellcheck disable=SC2086
    env ${ENVS:-} CUDA_VISIBLE_DEVICES="$GPUS" "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -b 512 -ub 128 \
        -np 1 -bs --cache-ram 0 --ctx-checkpoints 0 --no-webui --host 127.0.0.1 --port "$PORT" ${TS:+-ts $TS} \
        $pipe > "$OUT/server-d$d.log" 2>&1 < /dev/null &
    srv=$!
    # the rig has 4 GB of RAM: stop the server before the kernel has to
    (
        while kill -0 $srv 2> /dev/null; do
            avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
            if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
                echo "watchdog: MemAvailable ${avail} MiB, killing llama-server" | tee -a "$OUT/server-d$d.log" >&2
                kill -9 $srv
                break
            fi
            sleep 0.5
        done
    ) &
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        if ! kill -0 $srv 2> /dev/null; then
            echo "depth $d: the server exited during load"; tail -15 "$OUT/server-d$d.log"; continue 2
        fi
        sleep 2
    done
    echo "$(date -Is) depth $d: server up"

    python3 -u - "$PORT" "$d" "$MAX_TOKENS" "$TOOL_MAX_TOKENS" "$SEED" "$OUT/results.jsonl" "$HERE/prompts" <<'EOF'
import json, sys, urllib.request
port, depth, max_tok, tool_max_tok, seed, out, pdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]

def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": req}}}

TOOLS = [tool("bash", "Run a shell command", {"command": "string", "description": "string", "timeout": "number"},
              ["command", "description"]),
         tool("read", "Read a file", {"filePath": "string", "offset": "number", "limit": "number"}, ["filePath"]),
         tool("write", "Write a file", {"filePath": "string", "content": "string"}, ["filePath", "content"]),
         tool("edit", "Replace text in a file", {"filePath": "string", "oldString": "string", "newString": "string"},
              ["filePath", "oldString", "newString"])]
TOOL_MSGS = [{"role": "system", "content": "You are a coding agent. Use the tools."},
             {"role": "user", "content": "Create src/stats.py with a RunningStats class (Welford's algorithm: push, extend, "
                                         "variance, stdev, merge) and a short demo under __main__. Write it with the write tool."}]
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "samplers": ["top_k", "top_p", "min_p", "temperature"],
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
            "cache_prompt": False}

reqs = []
for p in ["chat", "code", "edit"]:
    msgs = [{"role": "user", "content": open(f"{pdir}/{p}.txt").read()}]
    reqs.append((p, msgs, None, max_tok))
reqs.append(("tool", TOOL_MSGS, TOOLS, tool_max_tok))

with open(out, "a") as f:
    for temp in (0.0, 1.0):
        for name, msgs, tools, n in reqs:
            body = dict(SAMPLING, messages=msgs, temperature=temp, seed=seed, max_tokens=n)
            if tools:
                body.update(tools=tools, tool_choice="auto")
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=7200))
            ch, t = r["choices"][0], r["timings"]
            m = ch["message"]
            calls = [[c["function"]["name"], c["function"]["arguments"]] for c in m.get("tool_calls") or []]
            row = {"depth": depth, "name": f"{name} T{temp:g}", "n": t["predicted_n"], "tok_s": t["predicted_per_second"],
                   "finish": ch.get("finish_reason"), "reasoning": m.get("reasoning_content") or "",
                   "content": m.get("content") or "", "calls": calls}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"depth {depth} {row['name']:8s}: {row['n']:5d} tok at {row['tok_s']:6.2f} t/s, {len(calls)} calls, {row['finish']}")
EOF
    kill $srv 2> /dev/null; wait $srv 2> /dev/null
done

# the "pipelined = G guesses, R right, W wrong" line of each request, in request order
python3 - "$OUT" "$DEPTHS" <<'EOF' | tee "$OUT/summary.tsv" | column -t -s "$(printf '\t')"
import json, re, sys
out, depths = sys.argv[1], [int(x) for x in sys.argv[2].split()]
rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
guesses = {}
for d in depths:
    try:
        lines = re.findall(r"pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong", open(f"{out}/server-d{d}.log").read())
    except FileNotFoundError:
        lines = []
    names = [r["name"] for r in rows if r["depth"] == d]
    for name, g in zip(names, lines):
        guesses[(d, name)] = g
by = {(r["depth"], r["name"]): r for r in rows}
names = list(dict.fromkeys(r["name"] for r in rows))
key = lambda r: (r["reasoning"], r["content"], json.dumps(r["calls"]))
cols = ["request"] + [f"d{d} tok/s" for d in depths] + [f"d{d} x" for d in depths if d > 0] + \
       [f"d{d} guesses (right/wrong)" for d in depths if d > 0] + ["identical"]
print("\t".join(cols))
for name in names:
    base = by.get((0, name))
    rest = [by.get((d, name)) for d in depths]
    line = [name] + [f"{r['tok_s']:.2f}" if r else "-" for r in rest]
    line += [f"{by[(d, name)]['tok_s'] / base['tok_s']:.2f}" if base and (d, name) in by else "-" for d in depths if d > 0]
    line += ["{}/{}".format(*guesses[(d, name)][1:]) if (d, name) in guesses else "-" for d in depths if d > 0]
    # T 0: every depth equals depth 0; T 1: the depths >= 1 equal each other
    if name.endswith("T0"):
        ref = base
        cmp = [by[(d, name)] for d in depths if d > 0 and (d, name) in by]
    else:
        keyed = [by[(d, name)] for d in depths if d > 0 and (d, name) in by]
        ref, cmp = (keyed[0], keyed[1:]) if keyed else (None, [])
    same = ref is not None and all(key(r) == key(ref) for r in cmp)
    line.append(("yes" if same else "NO") + ("" if name.endswith("T0") else " (depths >= 1)"))
    print("\t".join(line))
EOF
echo "$(date -Is) done $TAG"
