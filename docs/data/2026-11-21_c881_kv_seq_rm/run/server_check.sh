#!/usr/bin/env bash
# C881 (Q1590): llama-server on a CPU with random-weight models, requests that remove KV cells, the reference build
# against the rework: the same tokens? Runs inside a container with no network (needs curl, jq).
# Env: WORK (build.sh's work dir, with $WORK/t/<tree>/build/bin), C (output dir), THREADS (6), PORT (18295).
#
# Models: test-llama-archs saves random models with GPT-2's tokenizer from the tree (LLAMA_ARCHS_VOCAB), each as a
# target (stdev 0.1) and a drafter (0.13, the same seed): llama (attention only: a rejected draft is removed from the KV
# cache) and qwen4exp (Flash-Next's hybrid architecture: with --spec-type draft-simple the server asks for no rollback
# snapshots, saves a checkpoint before each verify pass and restores it after a rejection).
# Server: -md <drafter> --spec-type draft-simple --spec-draft-n-max 4, probabilistic draft sampling, -bs, -c 4096.
#   llama, -np 2 -kvu (both slots in one stream, so one slot's removal leaves holes between the other's cells), each
#   request pinned to a slot (id_slot), cache_prompt on:
#     A1  slot 0, temperature 0, plain text, 64 tokens          speculative rollbacks
#     A2  slot 1, temperature 0, plain text, 48 tokens          slot 1's cells after slot 0's
#     A3  slot 0, streamed, 400 tokens, the client hangs up after 12 chunks (a cancel; not compared: its length depends
#         on when the server sees the hang-up)
#     A4  slot 0, temperature 0, A1's prompt extended, 48 tokens: the cached prompt reused, the rest of slot 0 removed
#     A5  slot 1, temperature 0.7, top-k 8, seed 7, under a JSON tool-call grammar, 64 tokens: slot 1's tail removed
#   qwen4exp, -np 1, cache_prompt on:
#     Q1  temperature 0, plain text, 48 tokens                  checkpoint restores
#     Q2  streamed, 400 tokens, hung up after 12 chunks (a cancel; not compared)
#     Q3  temperature 0, Q1's prompt extended, 48 tokens        the prompt cache after a cancel
#     Q4  temperature 0.7, top-k 8, seed 7, the grammar, 64 tokens
#     Q5  temperature 0, the grammar, 64 tokens, cut by the client at R_MAX seconds
# Arms: base (twice: is the check repeatable?), base with LLAMA_KV_SEQ_RM_BOUND=1, c881, c881 with LLAMA_KV_SEQ_RM_BOUND=1.
set -u
WORK=${WORK:?}
C=${C:?}
THREADS=${THREADS:-6}
PORT=${PORT:-18295}
R_MAX=${R_MAX:-60}
OUT=$C/out
mkdir -p "$C/models" "$OUT"
rm -f "$OUT"/*

export LLAMA_ARCHS_VOCAB=$WORK/t/c881/src/models/ggml-vocab-gpt-2.gguf LLAMA_ARCHS_N_CTX=4096
TA=$WORK/t/c881/build/bin/test-llama-archs
for arch in llama qwen4exp; do
    for m in t:0.1 d:0.13; do
        d=$C/models/$arch-${m%%:*}
        rm -rf "$d"; mkdir -p "$d"
        "$TA" -a "^$arch\$" -s 1 -d "${m#*:}" -o "$d" > "$d.log" 2>&1 || { echo "models: $arch ${m%%:*} failed"; tail -n 5 "$d.log"; }
    done
done
unset LLAMA_ARCHS_VOCAB LLAMA_ARCHS_N_CTX
LT=$C/models/llama-t/llama-dense.gguf LD=$C/models/llama-d/llama-dense.gguf
QT=$(ls "$C"/models/qwen4exp-t/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/qwen4exp-t/*.gguf | head -n 1)
QD=$(ls "$C"/models/qwen4exp-d/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/qwen4exp-d/*.gguf | head -n 1)
echo "models: $LT $LD $QT $QD"
md5sum "$LT" "$LD" "$QT" "$QD"

GRAMMAR=$C/json_tool.gbnf
cat > "$GRAMMAR" <<'EOF'
root ::= "{\"name\": \"" name "\", \"arguments\": {\"path\": \"" path "\", \"line\": " num "}}"
name ::= "read_file" | "write_file" | "list_dir"
path ::= [a-z/_.] [a-z/_.]? [a-z/_.]? [a-z/_.]? [a-z/_.]? [a-z/_.]? [a-z/_.]? [a-z/_.]?
num  ::= [0-9] [0-9]? [0-9]?
EOF

serve() { # <arm> <bin> <env> <args...>
    local arm=$1 bin=$2 envs=$3
    shift 3
    # shellcheck disable=SC2086
    env LLAMA_TRACE=1 $envs "$bin" "$@" -bs -ngl 0 -t "$THREADS" -c 4096 --no-webui --host 127.0.0.1 --port "$PORT" \
        > "$OUT/$arm.server.log" 2>&1 &
    echo $! > "$OUT/$arm.pid"
    for _ in $(seq 1 180); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
        kill -0 "$(cat "$OUT/$arm.pid")" 2> /dev/null || { echo "$arm: server exited"; tail -n 20 "$OUT/$arm.server.log"; return 1; }
        sleep 1
    done
    echo "$arm: server did not start"; return 1
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

# req <out> <slot or -1> <prompt> <n_predict> <temperature> <seed> <max seconds> [grammar file]
req() {
    local out=$1 slot=$2 prompt=$3 n=$4 temp=$5 seed=$6 maxs=$7 gfile=${8:-}
    jq -n --arg p "$prompt" --argjson n "$n" --argjson t "$temp" --argjson s "$seed" --argjson id "$slot" \
        --arg g "$( [ -n "$gfile" ] && cat "$gfile")" \
        '{prompt: $p, n_predict: $n, temperature: $t, top_k: 8, top_p: 1.0, min_p: 0.0, seed: $s, id_slot: $id,
          return_tokens: true, cache_prompt: true} + (if $g == "" then {ignore_eos: true} else {grammar: $g} end)' > "$out.req"
    local t0
    t0=$(date +%s%N)
    curl -s --max-time "$maxs" -H "Content-Type: application/json" --data @"$out.req" \
        "http://127.0.0.1:$PORT/completion" > "$out"
    echo "curl_rc=$? ms=$(( ($(date +%s%N) - t0) / 1000000 ))" > "$out.curl"
}

# cancel <out> <slot> <prompt>: a streamed request that the client leaves after 12 chunks
cancel() {
    local out=$1 slot=$2 prompt=$3
    jq -n --arg p "$prompt" --argjson id "$slot" \
        '{prompt: $p, n_predict: 400, temperature: 0, stream: true, id_slot: $id, ignore_eos: true, cache_prompt: true}' > "$out.req"
    curl -s -N --max-time 120 -H "Content-Type: application/json" --data @"$out.req" \
        "http://127.0.0.1:$PORT/completion" | head -n 24 > "$out"
    sleep 2
}

P1="def fibonacci(n):"
P1X="def fibonacci(n):
    # iterative, with two variables"
P2="The rig has nine cards and"
PJ="Call one tool as JSON:"

run_arm() { # <arm> <tree> <env>
    local arm=$1 bin=$WORK/t/$2/build/bin/llama-server envs=$3
    [ -x "$bin" ] || { echo "$arm: no binary $bin"; return 1; }
    serve "$arm-llama" "$bin" "$envs" -m "$LT" -md "$LD" -ngld 0 --spec-type draft-simple --spec-draft-n-max 4 \
        --spec-draft-sampling probabilistic --spec-draft-p-min 0 -np 2 -kvu || return 1
    req    "$OUT/$arm.A1.json" 0 "$P1"  64 0 1 300
    req    "$OUT/$arm.A2.json" 1 "$P2"  48 0 1 300
    cancel "$OUT/$arm.A3.txt"  0 "$P1"
    req    "$OUT/$arm.A4.json" 0 "$P1X" 48 0 1 300
    req    "$OUT/$arm.A5.json" 1 "$PJ"  64 0.7 7 300 "$GRAMMAR"
    sleep 1
    stop "$arm-llama"
    serve "$arm-qwen4exp" "$bin" "$envs" -m "$QT" -md "$QD" -ngld 0 --spec-type draft-simple --spec-draft-n-max 4 \
        --spec-draft-sampling probabilistic --spec-draft-p-min 0 -np 1 || return 1
    req    "$OUT/$arm.Q1.json" -1 "$P1"  48 0 1 300
    cancel "$OUT/$arm.Q2.txt"  -1 "$P1"
    req    "$OUT/$arm.Q3.json" -1 "$P1X" 48 0 1 300
    req    "$OUT/$arm.Q4.json" -1 "$PJ"  64 0.7 7 300 "$GRAMMAR"
    req    "$OUT/$arm.Q5.json" -1 "$PJ"  64 0 1 "$R_MAX" "$GRAMMAR"
    sleep 1
    stop "$arm-qwen4exp"
    for r in A1 A2 A4 A5 Q1 Q3 Q4 Q5; do
        echo "$arm $r $(cat "$OUT/$arm.$r.json.curl") $(jq -c '{n: (.tokens | length), n_cached: .tokens_cached,
            prompt_n: .timings.prompt_n, draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}' \
            "$OUT/$arm.$r.json" 2>/dev/null || echo 'no JSON reply')"
    done
    echo "$arm cancels: A3 $(grep -c '^data: ' "$OUT/$arm.A3.txt") chunks, Q2 $(grep -c '^data: ' "$OUT/$arm.Q2.txt") chunks;" \
        "restores logged: $(grep -c '(restore checkpoint)' "$OUT/$arm-qwen4exp.server.log");" \
        "cancelled tasks logged: $(grep -ci 'cancel' "$OUT/$arm-llama.server.log") (llama), $(grep -ci 'cancel' "$OUT/$arm-qwen4exp.server.log") (qwen4exp)"
}

run_arm base     base ""
run_arm base2    base ""
run_arm baseB    base "LLAMA_KV_SEQ_RM_BOUND=1"
run_arm c881     c881 ""
run_arm c881B    c881 "LLAMA_KV_SEQ_RM_BOUND=1"

echo "--- the tokens of each reply (first 12 hex digits of the md5 of its token list)"
for r in A1 A2 A4 A5 Q1 Q3 Q4 Q5; do
    line="$r"
    for a in base base2 baseB c881 c881B; do
        line="$line $a=$(jq -c '.tokens' "$OUT/$a.$r.json" 2>/dev/null | md5sum | cut -c1-12)"
    done
    echo "$line"
done | tee "$C/tokens.txt"
awk '{ split($2, b, "="); same = 1; for (i = 3; i <= NF; i++) { split($i, x, "="); if (x[2] != b[2]) same = 0 } print $1, (same ? "SAME" : "DIFFERENT") }' \
    "$C/tokens.txt" | tee "$C/verdicts.txt"
date -Is > "$C/DONE"
