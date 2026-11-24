#!/usr/bin/env bash
# C929 (Q1667): C901's server check (../../2026-11-21_c901_pipe_slot_plan/run/c901_server.sh) with graph reuse on over
# RPC. llama-server with pipelined speculation (--pipe-depth 4) on a random-weight model of Flint's architecture, the
# same four requests in every arm, on one CPU or on two RPC devices on this host (two ggml-rpc-server processes on
# 127.0.0.1, each a CPU backend with its own buffers: the model's layers split over them as over two cards, so the pipe
# slots' graphs change device and their schedulers re-plan, and a re-plan that grows a buffer frees the old one; -fit
# off, as the fit cannot size RPC devices). C901 had to run its RPC arms with LLAMA_GRAPH_REUSE_DISABLE=1: ggml's RPC
# server drops every graph it caches when a buffer is freed, the client kept their ids and asked for one again, and the
# client aborted at the first reused pipe-slot graph.
# Arms (ARMS, any of): cpu_off, cpu_on (one CPU); rpc_noreuse_off, rpc_noreuse_on (two RPC devices, graph reuse off,
# as C901); rpc_reuse_off, rpc_reuse_on (two RPC devices, graph reuse on). "_on" sets LLAMA_PIPE_SLOT_PLAN=1, C901's
# switch, which only a build with C901's commit knows. Every arm runs with LLAMA_UBATCH_TRACE=1 and Flint's CPU-side
# preset switches; each RPC arm starts two fresh servers with GGML_RPC_DEBUG=RPC_DEBUG (default 2), so their logs list
# every graph_compute (a graph sent whole, with its uid), graph_recompute (a cached graph run by its uid), free_buffer,
# and every rejected uid ("graph with uid ... not found", after which the server closes the connection).
# Every arm should give the same tokens per request.
# Needs curl, jq and md5sum. Env: BIN (a build's bin folder with GGML_RPC on: llama-server, ggml-rpc-server,
# test-llama-archs), VOCAB (a GGUF vocabulary, e.g. models/ggml-vocab-gpt-2.gguf from the llama.cpp tree), OUT (output
# folder), ARMS ('cpu_off rpc_noreuse_off rpc_reuse_off'), MODEL (optional: an existing model instead of a new one),
# THREADS (6), PORT (18491), RPC_PORTS (50152 50153), RPC_DEBUG (2).
#
# Requests (cache_prompt false, seed fixed; the server's own sampling is Flint's preset: temp 1.0, top-p 0.95, top-k 20,
# min-p 0):
#   Q1  plain text, the server's sampling (no sampling fields), 700 tokens (ignore_eos): the reply crosses 256-cell steps
#   Q2  plain text, another chain (temperature 0.7, top-k 8, top-p 1.0), 700 tokens
#   Q3  a tool call's JSON schema, temperature 0.7, top-k 8, up to 96 tokens (the pipe is off under a grammar)
#   Q4  Q1 again: a second request whose context starts small again
set -u
BIN=${BIN:?BIN=<bin folder with llama-server, ggml-rpc-server and test-llama-archs>}
VOCAB=${VOCAB:-}
OUT=${OUT:?OUT=<output folder>}
ARMS=${ARMS:-cpu_off rpc_noreuse_off rpc_reuse_off}
THREADS=${THREADS:-6}
PORT=${PORT:-18491}
RPC_PORTS=${RPC_PORTS:-50152 50153}
RPC_DEBUG=${RPC_DEBUG:-2}
ARGS="-fa on -c 4096 -np 1 -b 64 -ub 16 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
# Flint's preset switches that apply on a CPU (release/flint-v5/settings.env), and the trace
PRESET_ENV="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_UBATCH_TRACE=1"
mkdir -p "$OUT/model" "$OUT/out"
rm -f "$OUT"/out/*

# the model: test-llama-archs's random Flint model (seed 1, stdev 0.1) with the given vocabulary, or MODEL
if [ -n "${MODEL:-}" ]; then
    M=$MODEL
else
    : "${VOCAB:?VOCAB=<GGUF vocabulary> (or MODEL=<model>)}"
    rm -f "$OUT"/model/*.gguf
    LLAMA_ARCHS_VOCAB=$VOCAB LLAMA_ARCHS_N_CTX=4096 "$BIN/test-llama-archs" -a '^flint$' -s 1 -d 0.1 -o "$OUT/model" \
        > "$OUT/model/save.log" 2>&1 || { echo "model: test-llama-archs failed"; tail -n 5 "$OUT/model/save.log"; }
    M=$(ls "$OUT"/model/*.gguf 2>/dev/null | head -n 1)
fi
[ -n "$M" ] && [ -f "$M" ] || { echo "no model"; exit 1; }
echo "model: $(basename "$M") ($(stat -c %s "$M") bytes, md5 $(md5sum < "$M" | cut -c1-32))"

SCHEMA='{"type":"object","properties":{"name":{"enum":["read_file","write_file","list_dir"]},"arguments":{"type":"object","properties":{"path":{"type":"string","maxLength":16},"line":{"type":"integer","minimum":0,"maximum":999}},"required":["path","line"]}},"required":["name","arguments"]}'
P1="def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))\n# next:"
P2="You are a coding agent. Call one tool as JSON to read the file that holds the bug, then wait for its result:"

RPC_PIDS=""
rpc_start() { # <arm>
    local p
    for p in $RPC_PORTS; do
        GGML_RPC_DEBUG=$RPC_DEBUG "$BIN/ggml-rpc-server" -H 127.0.0.1 -p "$p" -t 2 > "$OUT/out/$1.rpc-$p.log" 2>&1 &
        RPC_PIDS="$RPC_PIDS $!"
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

serve() { # <arm> <env> <extra args>
    local arm=$1 envs=$2 extra=$3
    env $PRESET_ENV $envs "$BIN/llama-server" -m "$M" $ARGS $extra -t "$THREADS" --no-webui \
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

# one RPC server's log: commands by kind, graphs sent whole again under a uid it had seen, uids it rejected
rpc_counts() { # <log>
    awk '/\[graph_compute\] device:/ { n_c++; for (i = 1; i <= NF; i++) if ($i == "uid:") { u = $(i + 1); sub(",", "", u);
             if (u != 0 && (u in seen)) n_again++; seen[u] = 1 } }
         /\[graph_recompute\] device:/ { n_r++ }
         /\[free_buffer\] remote_ptr/ { n_f++ }
         /\[alloc_buffer\]/ { n_a++ }
         /graph with uid .* not found/ { n_nf++ }
         END { printf "graph_compute %d (of them a uid seen before %d), graph_recompute %d, alloc_buffer %d, free_buffer %d, rejected uids %d\n",
               n_c, n_again, n_r, n_a, n_f, n_nf }' "$1"
}

run_arm() { # <arm>
    local arm=$1 envs="" extra="" p1 r
    case $arm in
        cpu_off|cpu_on|rpc_noreuse_off|rpc_noreuse_on|rpc_reuse_off|rpc_reuse_on) ;;
        *) echo "$arm: unknown arm"; return 1 ;;
    esac
    case $arm in *_on) envs="LLAMA_PIPE_SLOT_PLAN=1" ;; esac
    case $arm in *noreuse*) envs="$envs LLAMA_GRAPH_REUSE_DISABLE=1" ;; esac
    case $arm in rpc_*)
        extra="--rpc $(echo $RPC_PORTS | sed -E 's/([0-9]+)/127.0.0.1:\1/g; s/ /,/g') -fit off -ngl 99 -ts 1,1"
        rpc_start "$arm" ;;
    esac
    serve "$arm" "$envs" "$extra" || { rpc_stop; return 1; }
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
    local alive=yes
    kill -0 "$(cat "$o.pid")" 2> /dev/null || alive=no
    stop "$arm"
    case $arm in rpc_*) rpc_stop ;; esac
    local log=$o.server.log
    echo "$arm: env '$envs'; llama-server alive after the four requests: $alive"
    for r in Q1 Q2 Q3 Q4; do
        echo "$arm $r $(cat "$o.$r.json.curl") $(jq -c '{n: (.tokens | length), prompt_n: .timings.prompt_n,
            predicted_n: .timings.predicted_n}' "$o.$r.json" 2>/dev/null || echo 'no JSON reply') tokens_md5=$(jq -c .tokens \
            "$o.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$arm: per request, the server's counts (pipelined guesses, graph reallocs, pipe reallocs):"
    grep -aE 'pipelined = |graph reallocs =|pipe reallocs =' "$log" | sed -E 's/^.*\| task/  task/; s/ +/ /g'
    echo "$arm: sched-replan lines $(grep -ac 'sched-replan:' "$log"), of them after a change of device" \
        "$(grep -a 'sched-replan:' "$log" | grep -c 'backends_changed = 1'), galloc-grow lines $(grep -ac 'galloc-grow:' "$log")"
    echo "$arm: $(grep -a 'pipelined speculation:' "$log" | sed -E 's/^.*load_model: //' | head -n 1)"
    echo "$arm: client aborts ('Remote RPC server crashed'): $(grep -ac 'Remote RPC server crashed' "$log"); failed connections: $(grep -ac 'Failed to connect' "$log")"
    grep -a -m 3 'Remote RPC server crashed\|GGML_ASSERT\|terminate called' "$log" | sed -E "s/^/$arm:   /"
    case $arm in rpc_*)
        for p in $RPC_PORTS; do
            echo "$arm: rpc server $p: $(rpc_counts "$o.rpc-$p.log")"
            grep -a -m 2 'not found' "$o.rpc-$p.log" | sed -E "s/^/$arm:   rpc $p: /"
        done ;;
    esac
}

RAN=""
for a in $ARMS; do
    run_arm "$a"
    RAN="$RAN $a"
done

echo "--- tokens of each request in each arm (md5 of the token list, first 12 hex digits)"
for r in Q1 Q2 Q3 Q4; do
    line="$r"
    for a in $RAN; do
        line="$line $a=$(jq -c .tokens "$OUT/out/$a.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$line"
done
