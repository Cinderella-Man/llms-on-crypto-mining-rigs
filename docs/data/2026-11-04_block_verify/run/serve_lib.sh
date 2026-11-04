# Shared by smoke.sh and stat.sh (sourced, inside the check's container): start and stop a llama-server arm, send
# requests. Needs T (target model), DR (draft model), THREADS, PORT; writes under $OUT.
DRAFT="-md $DR -ngld 0 --spec-type draft-simple --spec-draft-n-max 4 --spec-draft-sampling probabilistic --spec-draft-p-min 0"

serve() { # <arm> <bin> <server args> [env...]
    local arm=$1 bin=$2 sargs=$3; shift 3
    env "$@" $bin -m "$T" $sargs -ngl 0 -t "$THREADS" -c 4096 -np 1 --jinja --no-webui --host 127.0.0.1 \
        --port "$PORT" > "$OUT/$arm.server.log" 2>&1 &
    echo $! > "$OUT/$arm.pid"
    for _ in $(seq 1 180); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
        kill -0 "$(cat "$OUT/$arm.pid")" 2> /dev/null || { echo "$arm: server exited"; return 1; }
        sleep 1
    done
    echo "$arm: server did not start"; tail -20 "$OUT/$arm.server.log"; return 1
}

stop() { # <arm>
    kill "$(cat "$OUT/$1.pid")" 2> /dev/null
    for _ in $(seq 1 20); do
        kill -0 "$(cat "$OUT/$1.pid")" 2> /dev/null || return 0
        sleep 0.5
    done
    kill -9 "$(cat "$OUT/$1.pid")" 2> /dev/null
    sleep 1
}

# /completion: <out file> <prompt> <n_predict> <temperature> <seed> [grammar file]
complete() {
    local out=$1 prompt=$2 n=$3 temp=$4 seed=$5 gfile=${6:-}
    if [ -n "$gfile" ]; then
        jq -n --arg p "$prompt" --argjson n "$n" --argjson t "$temp" --argjson s "$seed" --rawfile g "$gfile" \
            '{prompt: $p, n_predict: $n, temperature: $t, top_k: 8, top_p: 1.0, min_p: 0.0, seed: $s,
              grammar: $g, return_tokens: true, cache_prompt: false}' > "$out.req"
    else
        jq -n --arg p "$prompt" --argjson n "$n" --argjson t "$temp" --argjson s "$seed" \
            '{prompt: $p, n_predict: $n, temperature: $t, top_k: 8, top_p: 1.0, min_p: 0.0, seed: $s,
              ignore_eos: true, return_tokens: true, cache_prompt: false}' > "$out.req"
    fi
    curl -s --max-time 600 -H "Content-Type: application/json" --data @"$out.req" \
        "http://127.0.0.1:$PORT/completion" > "$out"
    rm -f "$out.req"
}

