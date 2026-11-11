#!/usr/bin/env bash
# C761 (Q1339): llama-server with the n-gram table's rows prefetched by a helper thread (LLAMA_NGRAM_PREFETCH=1), checked
# on the CPU with a random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp) whose n-gram table is
# mapped (--lazy-mode on), so that the n-gram input prefetches its rows as Flash-Next's 28.8 GB table does. Made from
# C738's server_check.sh (docs/data/2026-11-11_c738_verify_prepare). BASE is the base branch's llama-server, BIN the
# branch's.
# Arms (all -c 8192 -b 512 -ub 64 -np 1 --ctx-checkpoints 4 --cache-ram 0 --lazy-mode on, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0, as the release preset drafts; a 600-token prompt is a call of 512 tokens in
# 8 micro-batches and one of 88 in 2):
#   base     BASE: the reference
#   off      BIN, nothing set
#   on       BIN, LLAMA_NGRAM_PREFETCH=1 (the default cap, 48 MiB)
#   small    BIN, LLAMA_NGRAM_PREFETCH=1 LLAMA_NGRAM_PREFETCH_MB=0.0625 (64 KiB: a few tokens' rows of this table)
#   offtr    BIN, LLAMA_UBATCH_TRACE=1: the inline path, its rows counted per call (ngram-prefetch lines)
#   ontr     BIN, LLAMA_NGRAM_PREFETCH=1 LLAMA_UBATCH_TRACE=1
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42); S2 = G2's prompt, the same with seed 7.
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and the target's
# top tokens.
# Pass: every BIN arm gives base's tokens, top probabilities, draft counts and verify steps on every request; offtr and
# ontr log the same calls with the same rows, distinct pages and row digest per call (ngram_calls.py), ontr's helper
# issued rows and none differed from set_input's, and its window stayed within the cap; no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18761] [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl}, OUT/summary.tsv,
#   OUT/ngram_calls.tsv, OUT/verdict.txt
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18761}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on small offtr ontr}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [on]=$BIN [small]=$BIN [offtr]=$BIN [ontr]=$BIN
)
declare -A ENVS=(
    [on]="LLAMA_NGRAM_PREFETCH=1"
    [small]="LLAMA_NGRAM_PREFETCH=1 LLAMA_NGRAM_PREFETCH_MB=0.0625"
    [offtr]="LLAMA_UBATCH_TRACE=1"
    [ontr]="LLAMA_NGRAM_PREFETCH=1 LLAMA_UBATCH_TRACE=1"
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (a Park-Miller generator, exact in awk's
# doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <comma-separated tokens> <n_predict> <temperature> <seed>
    printf '{"prompt":[%s],"n_predict":%s,"temperature":%s,"top_k":20,"top_p":0.95,"min_p":0,"seed":%s,"n_probs":3,"return_tokens":true,"cache_prompt":false}' \
        "$3" "$4" "$5" "$6" > "$1/$2.req.json"
    curl -s --max-time 1800 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\ttokens\n' > "$OUT/summary.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    if [ -z "${EXE[$arm]:-}" ]; then
        echo "$arm: no binary"; continue
    fi
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" \
        -c 8192 -b 512 -ub 64 -np 1 --ctx-checkpoints 4 --cache-ram 0 --lazy-mode on --no-webui --host 127.0.0.1 \
        --port "$PORT" $MTP $SARGS > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 180); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -n 20 "$d/server.log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')

    p1=$(toks 200 11 "$nv")
    b=$(toks 150 13 "$nv")
    p2="$b,$b,$b,$b"
    complete "$d" G1 "$p1" 96 0 1
    complete "$d" G2 "$p2" 128 0 1
    complete "$d" S1 "$p1" 96 1.0 42
    complete "$d" S2 "$p2" 96 1.0 7

    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps; $(grep -c 'ngram-prefetch: call' "$d/server.log") ngram-prefetch lines; $(grep -c 'LLAMA_NGRAM_PREFETCH: the n-gram rows' "$d/server.log") switch lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

fail=0
same() { # <arm a> <arm b>: tokens, top probabilities and draft counts of every request, and the verify steps
    local r
    for r in $REQS; do
        if [ ! -s "$OUT/$1/$r.json" ] || ! cmp -s <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    if [ ! -s "$OUT/$1/steps.jsonl" ] || ! cmp -s "$OUT/$1/steps.jsonl" "$OUT/$2/steps.jsonl"; then
        echo "$2's verify steps differ from $1's (or $1 wrote none)"; return 1
    fi
    echo "$2 = $1 (tokens, top probabilities and draft counts of $REQS; $(wc -l < "$OUT/$2/steps.jsonl") verify steps the same)"
}
has() { [[ " $ARMS " == *" $1 "* ]]; }
{
    for a in off on small offtr ontr; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    # the switch's line when the context is made, and the context's closing line: the helper issued rows, none differed
    for a in on small ontr; do
        has "$a" || continue
        c=$(grep -o 'C761: LLAMA_NGRAM_PREFETCH: .*' "$OUT/$a/server.log" | tail -n 1)
        h=$(sed -E 's/.* ([0-9]+) n-gram rows issued by the helper.*/\1/' <<< "${c:-x 0 n-gram rows issued by the helper}")
        dif=$(sed -E 's/.* ([0-9]+) differed.*/\1/' <<< "${c:-x 1 differed}")
        if grep -q 'LLAMA_NGRAM_PREFETCH: the n-gram rows' "$OUT/$a/server.log" && [ "${h:-0}" -gt 0 ] && [ "$dif" = 0 ]; then
            echo "$a: ${c#C761: LLAMA_NGRAM_PREFETCH: }"
        else
            echo "$a: the switch did not work (${c:-no closing line})"; fail=1
        fi
    done
    if has offtr && has ontr; then
        python3 -I "$here/ngram_calls.py" "$OUT/offtr/server.log" "$OUT/ontr/server.log" "$OUT/ngram_calls.tsv" || fail=1
    fi
    for a in $ARMS; do
        [ -f "$OUT/$a/server.log" ] || continue
        e=$(grep -ciE 'error|abort' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
