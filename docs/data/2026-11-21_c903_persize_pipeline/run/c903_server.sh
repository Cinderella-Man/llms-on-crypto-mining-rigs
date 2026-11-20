#!/usr/bin/env bash
# C903 (Q1632): llama-server on one CPU with a random-weight model of Flint's architecture, the same requests with the
# per-size schedulers off, on in the base build (before C903), and on in the change's build: as built (one CPU has no
# pipeline parallelism, so they are made as before), with LLAMA_SCHED_PER_SIZE_PP=0, and with =2 (made with the input
# copies of pipeline parallelism: the test mode, traced with LLAMA_UBATCH_TRACE=1 so the log shows each scheduler's
# copies and each pass's copy). Every arm gets the same tokens, or the change broke something.
# Needs curl and jq. Env: NEW and BASE (folders with the two builds' llama-server and test-llama-archs), VOCAB (a GGUF
# vocabulary, e.g. models/ggml-vocab-gpt-2.gguf from the llama.cpp tree), OUT (output folder), THREADS (6), PORT (18391),
# ARGS (extra server arguments, default Flint's preset flags that run on a CPU).
#
# Requests (cache_prompt false unless said):
#   R1  temperature 0, plain text, 320 tokens (ignore_eos): the reply crosses 256 cells of the cache, where the graphs
#       are built again (and a scheduler with four copies takes its next one)
#   R2  temperature 0, a tool call's JSON schema, up to 96 tokens
#   R3  temperature 0.7, top-k 8, seed 7, the same schema, up to 96 tokens
#   R4  temperature 0, R2's prompt and R2's reply and more text, the same schema, cache_prompt true (a second turn)
set -u
NEW=${NEW:?NEW=<folder with the change's llama-server and test-llama-archs>}
BASE=${BASE:?BASE=<folder with the base's llama-server>}
VOCAB=${VOCAB:?VOCAB=<GGUF vocabulary>}
OUT=${OUT:?OUT=<output folder>}
THREADS=${THREADS:-6}
PORT=${PORT:-18391}
ARGS=${ARGS:--fa on -c 4096 -np 1 -b 64 -ub 16 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4}
# Flint's preset switches that apply on a CPU (release/flint-v5/settings.env)
PRESET_ENV="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SCHED_PER_SIZE_RESERVE=2"
mkdir -p "$OUT/model" "$OUT/out"
rm -f "$OUT"/out/*

# the model: test-llama-archs's random Flint model (seed 1, stdev 0.1) with the given vocabulary
rm -f "$OUT"/model/*.gguf
LLAMA_ARCHS_VOCAB=$VOCAB LLAMA_ARCHS_N_CTX=4096 "$NEW/test-llama-archs" -a '^flint$' -s 1 -d 0.1 -o "$OUT/model" \
    > "$OUT/model/save.log" 2>&1 || { echo "model: test-llama-archs failed"; tail -n 5 "$OUT/model/save.log"; }
M=$(ls "$OUT"/model/*.gguf 2>/dev/null | head -n 1)
[ -n "$M" ] || { echo "no model saved"; exit 1; }
echo "model: $M ($(stat -c %s "$M") bytes)"
ls -la "$OUT/model"

SCHEMA='{"type":"object","properties":{"name":{"enum":["read_file","write_file","list_dir"]},"arguments":{"type":"object","properties":{"path":{"type":"string","maxLength":16},"line":{"type":"integer","minimum":0,"maximum":999}},"required":["path","line"]}},"required":["name","arguments"]}'
P1="def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))\n# next:"
P2="You are a coding agent. Call one tool as JSON to read the file that holds the bug, then wait for its result:"

serve() { # <arm> <bin dir> <env>
    local arm=$1 bin=$2 envs=$3
    env $PRESET_ENV $envs "$bin/llama-server" -m "$M" $ARGS -ngl 0 -t "$THREADS" --no-webui \
        --host 127.0.0.1 --port "$PORT" > "$OUT/out/$arm.server.log" 2>&1 &
    echo $! > "$OUT/out/$arm.pid"
    for _ in $(seq 1 180); do
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

complete() { # <out> <prompt> <n_predict> <temperature> <seed> <schema: 0|1> <cache_prompt: true|false>
    local out=$1 prompt=$2 n=$3 temp=$4 seed=$5 schema=$6 cache=$7
    if [ "$schema" = 1 ]; then
        jq -n --arg p "$prompt" --argjson n "$n" --argjson t "$temp" --argjson s "$seed" --argjson js "$SCHEMA" \
            --argjson c "$cache" '{prompt: $p, n_predict: $n, temperature: $t, top_k: 8, top_p: 1.0, min_p: 0.0,
            seed: $s, json_schema: $js, return_tokens: true, cache_prompt: $c}' > "$out.req"
    else
        jq -n --arg p "$prompt" --argjson n "$n" --argjson t "$temp" --argjson s "$seed" --argjson c "$cache" \
            '{prompt: $p, n_predict: $n, temperature: $t, top_k: 8, top_p: 1.0, min_p: 0.0, seed: $s,
              ignore_eos: true, return_tokens: true, cache_prompt: $c}' > "$out.req"
    fi
    local t0
    t0=$(date +%s%N)
    curl -s --max-time 300 -H "Content-Type: application/json" --data @"$out.req" \
        "http://127.0.0.1:$PORT/completion" > "$out"
    echo "curl_rc=$? ms=$(( ($(date +%s%N) - t0) / 1000000 ))" > "$out.curl"
    rm -f "$out.req"
}

run_arm() { # <arm> <bin dir> <env>
    local arm=$1 bin=$2 envs=$3
    serve "$arm" "$bin" "$envs" || return 1
    local o=$OUT/out/$arm
    complete "$o.R1.json" "$(printf '%b' "$P1")" 320 0 1 0 false
    complete "$o.R2.json" "$P2" 96 0 1 1 false
    complete "$o.R3.json" "$P2" 96 0.7 7 1 false
    local r2
    r2=$(jq -r '.content // ""' "$o.R2.json")
    complete "$o.R4.json" "$P2 $r2
Result: line 42 divides by zero when the list is empty. Call the next tool as JSON:" 96 0 1 1 true
    sleep 1
    stop "$arm"
    for r in R1 R2 R3 R4; do
        echo "$arm $r $(cat "$o.$r.json.curl") $(jq -c '{n: (.tokens | length), prompt_n: .timings.prompt_n,
            cache_n: .timings.cache_n, tokens_md5: (.tokens | tostring), content: .content[0:100]}' "$o.$r.json" 2>/dev/null \
            | sed 's/"tokens_md5":"[^"]*"/"tokens_md5":"'"$(jq -c .tokens "$o.$r.json" 2>/dev/null | md5sum | cut -c1-12)"'"/' \
            || echo 'no JSON reply')"
    done
    echo "$arm per-size scheduler lines: $(grep -ac 'scheduler for [0-9]*-token batches' "$OUT/out/$arm.server.log")"
    grep -a 'scheduler for [0-9]*-token batches\|LLAMA_SCHED_PER_SIZE' "$OUT/out/$arm.server.log" | sed "s/^/$arm: /" | head -n 12
}

run_arm off     "$NEW"  ""
run_arm base_on "$BASE" "LLAMA_SCHED_PER_SIZE=1"
run_arm new_on  "$NEW"  "LLAMA_SCHED_PER_SIZE=1"
run_arm new_pp0 "$NEW"  "LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=0"
run_arm new_pp2 "$NEW"  "LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=2 LLAMA_UBATCH_TRACE=1"

echo "--- tokens of each request in each arm (md5 of the token list, first 12 hex digits)"
for r in R1 R2 R3 R4; do
    line="$r"
    for a in off base_on new_on new_pp0 new_pp2; do
        line="$line $a=$(jq -c .tokens "$OUT/out/$a.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$line"
done
echo "--- new_pp2: the copies its passes used (sched-trace 'copy = N'), by copy"
grep -ao 'sched-trace: copy = [0-9]*' "$OUT/out/new_pp2.server.log" | sort | uniq -c
echo "--- new_pp2: decode calls and their micro-batches in order (a call of several whose last has at most 8 tokens"
echo "    runs that one on a per-size scheduler while the earlier ones are in flight)"
grep -aoE '(decode|ubatch)-trace: [a-z]+ n_tokens = [0-9]+( n_seqs = [0-9]+ n_outputs = [0-9]+ graph = [a-z]+)?' \
    "$OUT/out/new_pp2.server.log" | sed 's/ n_seqs = [0-9]*//' > "$OUT/out/new_pp2.passes.txt"
wc -l < "$OUT/out/new_pp2.passes.txt"
head -n 40 "$OUT/out/new_pp2.passes.txt"
