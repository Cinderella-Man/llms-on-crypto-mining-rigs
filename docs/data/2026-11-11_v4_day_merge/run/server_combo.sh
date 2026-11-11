#!/usr/bin/env bash
# The v4 day merge (C773): llama-server's draft-mtp on the CPU with the switches of the merged branches on together,
# against the base branch's llama-server. A random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp)
# whose n-gram table is mapped (-lzm on), as C615's server check (docs/data/2026-11-11_c615_draft_prefetch, whose
# requests, comparison and verdict this script keeps). BASE is the base branch's llama-server, BIN the merged one's.
# Arms (all -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --ctx-checkpoints-vram 4 --cache-ram 0 -lm mmap -lzm on,
# --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0):
#   base   BASE: the reference of off and alln
#   bbase  BASE, -bs (the backend sampler, as the preset): the reference of boff, r90 and all
#   off    BIN, nothing set
#   boff   BIN, -bs, nothing set
#   r90    BIN, -bs, the host switches R90 sets (R90_ENV)
#   all    BIN, -bs, --ctx-checkpoints-incall, every exact host switch of the night's and the day's branches (ALL_ENV)
#   alln   BIN, the same without -bs
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7. Every arm writes
# LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their probabilities and the
# target's top tokens.
# Pass: every arm but the references gives its reference's tokens, top probabilities, draft counts (draft_n,
# draft_n_accepted) and verify steps (the dump without its run, step and task fields) on every request; no arm logs an
# error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the merged llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18773] [THREADS=8] [ARMS='base bbase ...'] [R90_ENV=...] [ALL_ENV=...]
#   [SARGS='<more server arguments>'] bash server_combo.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl}, OUT/summary.tsv,
#   OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the merged llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18773}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base bbase off boff r90 all alln}
SARGS=${SARGS:-}
# R90's host switches (C691, C704, C100: the night's gains on a P106); the CUDA ones change nothing on the CPU
R90_ENV=${R90_ENV:-LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1}
# every host switch of the two merges that computes what the base computes: the night's C620, C703, C100, C720, C719,
# C704, C691 (C217 is the flag below), the day's C619, C738, C740, C741, C761, C615
ALL_ENV=${ALL_ENV:-LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 \
LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_MTP_GRAPH_KEEP=1 \
LLAMA_VERIFY_PREPARE=2 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_NGRAM_PREFETCH=1 LLAMA_DRAFT_PREFETCH=1}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [bbase]=$BASE [off]=$BIN [boff]=$BIN [r90]=$BIN [all]=$BIN [alln]=$BIN
)
declare -A FLAGS=(
    [bbase]="-bs" [boff]="-bs" [r90]="-bs" [all]="-bs --ctx-checkpoints-incall" [alln]="--ctx-checkpoints-incall"
)
declare -A ENVS=(
    [r90]="$R90_ENV" [all]="$ALL_ENV" [alln]="$ALL_ENV"
)
declare -A REF=(
    [off]=base [boff]=bbase [r90]=bbase [all]=bbase [alln]=base
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
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --ctx-checkpoints-vram 4 --cache-ram 0 -lm mmap -lzm on \
        --no-webui --host 127.0.0.1 --port "$PORT" $MTP $SARGS ${FLAGS[$arm]:-} > "$d/server.log" 2>&1 &
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

    # SIGTERM and wait: the contexts' closing lines and the dump are whole
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
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
    for a in $ARMS; do
        [ -n "${REF[$a]:-}" ] || continue
        has "${REF[$a]}" || continue
        same "${REF[$a]}" "$a" || fail=1
    done
    for a in $ARMS; do
        [ -f "$OUT/$a/server.log" ] || continue
        e=$(grep -ciE 'error|abort' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
