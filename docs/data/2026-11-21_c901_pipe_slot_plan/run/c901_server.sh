#!/usr/bin/env bash
# C901 (Q1629): llama-server with pipelined speculation (--pipe-depth 4) on a random-weight model of Flint's
# architecture, the same requests with LLAMA_PIPE_SLOT_PLAN unset and =1, on one CPU and, when RPCBIN is given, on two
# RPC devices on this host (two rpc-server processes on 127.0.0.1, each a CPU backend with its own buffers: the model's
# layers split over them as over two cards, so the pipe slots' graphs have a change of device that one CPU cannot
# have; -fit off, as the fit cannot size RPC devices; LLAMA_GRAPH_REUSE_DISABLE=1, since ggml's RPC server drops the
# graphs it caches when a buffer is reallocated, a re-plan's growth among them, and then fails when a reused graph is
# run again by its id: with reuse off every step sends its graph whole, and the slots re-plan at the same points, as a
# graph is rebuilt anyway whenever the context grows by a step of the cache). Every arm runs with LLAMA_UBATCH_TRACE=1 (a sched-replan line for every re-plan, with the node whose device
# changed) and logs the server's per-request counts, including the new "pipe reallocs" line.
# Every arm must give the same tokens per request; the switch should leave fewer pipe-slot re-plans or none.
# Needs curl and jq. Env: BIN (a build's bin folder: llama-server, test-llama-archs), RPCBIN (optional: a build's bin
# folder with GGML_RPC on: llama-server, ggml-rpc-server), VOCAB (a GGUF vocabulary, e.g. models/ggml-vocab-gpt-2.gguf from
# the llama.cpp tree), OUT (output folder), THREADS (6), PORT (18491), RPC_PORTS (50152 50153).
#
# Requests (cache_prompt false, seed fixed; the server's own sampling is Flint's preset: temp 1.0, top-p 0.95, top-k 20,
# min-p 0):
#   Q1  plain text, the server's sampling (no sampling fields), 700 tokens (ignore_eos): the reply crosses 256-cell steps
#   Q2  plain text, another chain (temperature 0.7, top-k 8, top-p 1.0), 700 tokens: a request whose sampler graph is not
#       the one the load planned
#   Q3  a tool call's JSON schema, temperature 0.7, top-k 8, up to 96 tokens (the pipe is off under a grammar unless
#       LLAMA_PIPE_GRAMMAR=1; these tokens go through the shared scheduler)
#   Q4  Q1 again: a second request whose context starts small again
set -u
BIN=${BIN:?BIN=<bin folder with llama-server and test-llama-archs>}
RPCBIN=${RPCBIN:-}
VOCAB=${VOCAB:?VOCAB=<GGUF vocabulary>}
OUT=${OUT:?OUT=<output folder>}
THREADS=${THREADS:-6}
PORT=${PORT:-18491}
RPC_PORTS=${RPC_PORTS:-50152 50153}
ARGS="-fa on -c 4096 -np 1 -b 64 -ub 16 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
# Flint's preset switches that apply on a CPU (release/flint-v5/settings.env), and the trace
PRESET_ENV="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_UBATCH_TRACE=1"
mkdir -p "$OUT/model" "$OUT/out"
rm -f "$OUT"/out/*

# the model: test-llama-archs's random Flint model (seed 1, stdev 0.1) with the given vocabulary
rm -f "$OUT"/model/*.gguf
LLAMA_ARCHS_VOCAB=$VOCAB LLAMA_ARCHS_N_CTX=4096 "$BIN/test-llama-archs" -a '^flint$' -s 1 -d 0.1 -o "$OUT/model" \
    > "$OUT/model/save.log" 2>&1 || { echo "model: test-llama-archs failed"; tail -n 5 "$OUT/model/save.log"; }
M=$(ls "$OUT"/model/*.gguf 2>/dev/null | head -n 1)
[ -n "$M" ] || { echo "no model saved"; exit 1; }
echo "model: $(basename "$M") ($(stat -c %s "$M") bytes)"

SCHEMA='{"type":"object","properties":{"name":{"enum":["read_file","write_file","list_dir"]},"arguments":{"type":"object","properties":{"path":{"type":"string","maxLength":16},"line":{"type":"integer","minimum":0,"maximum":999}},"required":["path","line"]}},"required":["name","arguments"]}'
P1="def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))\n# next:"
P2="You are a coding agent. Call one tool as JSON to read the file that holds the bug, then wait for its result:"

RPC_PIDS=""
rpc_start() {
    local p i=0
    for p in $RPC_PORTS; do
        "$RPCBIN/ggml-rpc-server" -H 127.0.0.1 -p "$p" -t 2 > "$OUT/out/rpc-$p.log" 2>&1 &
        RPC_PIDS="$RPC_PIDS $!"
        i=$((i + 1))
    done
    sleep 2
}
rpc_stop() {
    local p
    for p in $RPC_PIDS; do kill "$p" 2> /dev/null; done
    sleep 1
    for p in $RPC_PIDS; do kill -9 "$p" 2> /dev/null; done
    RPC_PIDS=""
}

serve() { # <arm> <bin dir> <env> <extra args>
    local arm=$1 bin=$2 envs=$3 extra=$4
    env $PRESET_ENV $envs "$bin/llama-server" -m "$M" $ARGS $extra -t "$THREADS" --no-webui \
        --host 127.0.0.1 --port "$PORT" > "$OUT/out/$arm.server.log" 2>&1 &
    echo $! > "$OUT/out/$arm.pid"
    for _ in $(seq 1 240); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
        kill -0 "$(cat "$OUT/out/$arm.pid")" 2> /dev/null || { echo "$arm: server exited"; tail -n 20 "$OUT/out/$arm.server.log"; return 1; }
        sleep 1
    done
    echo "$arm: server did not start"; return 1
}

stop() { # <arm>
    kill "$(cat "$OUT/out/$1.pid")" 2> /dev/null
    for _ in $(seq 1 20); do
        kill -0 "$(cat "$OUT/out/$1.pid")" 2> /dev/null || return 0
        sleep 0.5
    done
    kill -9 "$(cat "$OUT/out/$1.pid")" 2> /dev/null
    sleep 1
}

complete() { # <out> <json body>
    local out=$1 body=$2 t0
    echo "$body" > "$out.req"
    t0=$(date +%s%N)
    curl -s --max-time 600 -H "Content-Type: application/json" --data @"$out.req" \
        "http://127.0.0.1:$PORT/completion" > "$out"
    echo "curl_rc=$? ms=$(( ($(date +%s%N) - t0) / 1000000 ))" > "$out.curl"
    rm -f "$out.req"
}

run_arm() { # <arm> <bin dir> <env> <extra args>
    local arm=$1 bin=$2 envs=$3 extra=$4 p1
    serve "$arm" "$bin" "$envs" "$extra" || return 1
    local o=$OUT/out/$arm
    p1=$(printf '%b' "$P1")
    complete "$o.Q1.json" "$(jq -n --arg p "$p1" '{prompt: $p, n_predict: 700, seed: 11, ignore_eos: true,
        return_tokens: true, cache_prompt: false}')"
    complete "$o.Q2.json" "$(jq -n --arg p "$p1" '{prompt: $p, n_predict: 700, temperature: 0.7, top_k: 8, top_p: 1.0,
        min_p: 0.0, seed: 12, ignore_eos: true, return_tokens: true, cache_prompt: false}')"
    complete "$o.Q3.json" "$(jq -n --arg p "$P2" --argjson js "$SCHEMA" '{prompt: $p, n_predict: 96, temperature: 0.7,
        top_k: 8, top_p: 1.0, min_p: 0.0, seed: 13, json_schema: $js, return_tokens: true, cache_prompt: false}')"
    complete "$o.Q4.json" "$(jq -n --arg p "$p1" '{prompt: $p, n_predict: 700, seed: 14, ignore_eos: true,
        return_tokens: true, cache_prompt: false}')"
    sleep 1
    stop "$arm"
    local log=$OUT/out/$arm.server.log
    for r in Q1 Q2 Q3 Q4; do
        echo "$arm $r $(cat "$o.$r.json.curl") $(jq -c '{n: (.tokens | length), prompt_n: .timings.prompt_n,
            predicted_n: .timings.predicted_n}' "$o.$r.json" 2>/dev/null || echo 'no JSON reply') tokens_md5=$(jq -c .tokens \
            "$o.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$arm: per request, the server's counts (pipelined guesses, graph reallocs, pipe reallocs):"
    grep -aE 'pipelined = |graph reallocs =|pipe reallocs =' "$log" | sed -E 's/^.*\| task/  task/; s/ +/ /g'
    echo "$arm: sched-replan lines $(grep -ac 'sched-replan:' "$log"), of them after a change of device" \
        "$(grep -a 'sched-replan:' "$log" | grep -c 'backends_changed = 1')"
    echo "$arm: plans over the whole memory (pipe_slot_plan_full lines) $(grep -ac 'pipe_slot_plan_full:' "$log")"
    grep -a 'pipe_slot_plan_full:\|LLAMA_PIPE_SLOT_PLAN' "$log" | sed -E "s/^.*(pipe_slot_plan_full|pipe_init)/$arm: \1/" | head -n 24
    echo "$arm: $(grep -a 'pipelined speculation:' "$log" | sed -E 's/^.*load_model: //' | head -n 1)"
    echo "$arm: lines naming RPC: $(grep -ac 'RPC' "$log"); failed connections: $(grep -ac 'Failed to connect' "$log")"
    grep -a 'sched-replan:' "$log" | sed -E "s/^.*sched-replan: /$arm: sched-replan /" | head -n 6
}

run_arm cpu_off "$BIN" "" ""
run_arm cpu_on  "$BIN" "LLAMA_PIPE_SLOT_PLAN=1" ""
ARMS="cpu_off cpu_on"
if [ -n "$RPCBIN" ]; then
    RPC_LIST=$(echo $RPC_PORTS | sed -E 's/([0-9]+)/127.0.0.1:\1/g; s/ /,/g')
    rpc_start
    run_arm rpc_off "$RPCBIN" "LLAMA_GRAPH_REUSE_DISABLE=1" "--rpc $RPC_LIST -fit off -ngl 99 -ts 1,1"
    rpc_stop; rpc_start
    run_arm rpc_on  "$RPCBIN" "LLAMA_GRAPH_REUSE_DISABLE=1 LLAMA_PIPE_SLOT_PLAN=1" "--rpc $RPC_LIST -fit off -ngl 99 -ts 1,1"
    rpc_stop
    ARMS="$ARMS rpc_off rpc_on"
fi

echo "--- tokens of each request in each arm (md5 of the token list, first 12 hex digits)"
for r in Q1 Q2 Q3 Q4; do
    line="$r"
    for a in $ARMS; do
        line="$line $a=$(jq -c .tokens "$OUT/out/$a.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$line"
done
