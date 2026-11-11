#!/usr/bin/env bash
# C738 (Q1307): llama-server's verify pass prepared during draft-mtp's last drafting pass (LLAMA_VERIFY_PREPARE=1, and
# =2 with the first device split's inputs uploaded too), checked on the CPU with a random-weight qwen4exp model with one
# MTP block (test-llama-archs -o --mtp). Made from C735's server_check.sh (docs/data/2026-11-11_c735_c217_c703). BASE
# is the base branch's llama-server, BIN the branch's; RPC_BIN (optional) a llama-server and ggml-rpc-server built with
# GGML_RPC, whose arms put every layer on an RPC device served by a ggml-rpc-server on the same machine, so that the verify
# pass has a device split whose graph inputs are copied from the host.
# Arms (all -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0, as the release preset drafts; VP = LLAMA_VERIFY_PREPARE):
#   base     BASE: the reference
#   off      BIN, VP unset
#   on1      BIN, VP=1
#   on2      BIN, VP=2 (on the CPU alone there is no device split: nothing to upload)
#   bbase    BASE, -bs (the backend sampler, as the preset): the reference of the b arms
#   boff     BIN, -bs
#   bon1     BIN, -bs, VP=1
#   bon2     BIN, -bs, VP=2
#   roff     RPC_BIN, --rpc and -ngl 99: the reference of the r arms
#   ron1     RPC_BIN, the same, VP=1
#   ron2     RPC_BIN, the same, VP=2 and LLAMA_UBATCH_TRACE=1 (its lines show the inputs not copied again, ":u<n>")
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens.
# Pass: off, on1, on2 give base's tokens, top probabilities, draft counts (draft_n, draft_n_accepted) and verify steps
# (the dump without its run, step and task fields) on every request; boff, bon1, bon2 the same against bbase; ron1 and
# ron2 against roff; every VP arm prepared verify passes (its context's closing line), ron2 uploaded inputs ahead and
# skipped them; no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [RPC_BIN=<dir with llama-server and ggml-rpc-server built with GGML_RPC>] [PORT=18738] [RPC_PORT=50738]
#   [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl}, OUT/summary.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
RPC_BIN=${RPC_BIN:-}
PORT=${PORT:-18738}
RPC_PORT=${RPC_PORT:-50738}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on1 on2 bbase boff bon1 bon2$([ -n "$RPC_BIN" ] && echo ' roff ron1 ron2')}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [on1]=$BIN [on2]=$BIN
    [bbase]=$BASE [boff]=$BIN [bon1]=$BIN [bon2]=$BIN
    [roff]=$RPC_BIN [ron1]=$RPC_BIN [ron2]=$RPC_BIN
)
declare -A FLAGS=(
    [bbase]="-bs" [boff]="-bs" [bon1]="-bs" [bon2]="-bs"
    [roff]="--rpc 127.0.0.1:$RPC_PORT" [ron1]="--rpc 127.0.0.1:$RPC_PORT" [ron2]="--rpc 127.0.0.1:$RPC_PORT"
)
declare -A ENVS=(
    [on1]=LLAMA_VERIFY_PREPARE=1 [on2]=LLAMA_VERIFY_PREPARE=2
    [bon1]=LLAMA_VERIFY_PREPARE=1 [bon2]=LLAMA_VERIFY_PREPARE=2
    [ron1]=LLAMA_VERIFY_PREPARE=1 [ron2]="LLAMA_VERIFY_PREPARE=2 LLAMA_UBATCH_TRACE=1"
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

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\tprepared\tnot_prepared\tuploaded\ttokens\n' > "$OUT/summary.tsv"

rpc_pid=
for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    if [ -z "${EXE[$arm]}" ]; then
        echo "$arm: no binary"; continue
    fi
    if [[ $arm == r* ]] && [ -z "$rpc_pid" ]; then
        "$RPC_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$RPC_PORT" -t "$THREADS" > "$OUT/ggml-rpc-server.log" 2>&1 &
        rpc_pid=$!
        sleep 3
    fi
    ngl=0
    [[ $arm == r* ]] && ngl=99
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -ngl $ngl -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" \
        $MTP $SARGS ${FLAGS[$arm]:-} > "$d/server.log" 2>&1 &
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

    # the target context's closing line (C738): decodes prepared, not prepared, dropped; inputs uploaded ahead
    vp=$(grep -o -E 'C738: [0-9]+ decodes prepared ahead of their last token ids, [0-9]+ not prepared, [0-9]+ dropped; [0-9]+ inputs uploaded ahead' "$d/server.log" | tail -n 1)
    np=$(sed -E 's/C738: ([0-9]+) decodes.*/\1/' <<< "${vp:-C738: 0 decodes}")
    nn=$(sed -E 's/.* ([0-9]+) not prepared.*/\1/' <<< "${vp:-x 0 not prepared}")
    nu=$(sed -E 's/.* ([0-9]+) inputs uploaded.*/\1/' <<< "${vp:-x 0 inputs uploaded}")
    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" --arg np "$np" --arg nn "$nn" --arg nu "$nu" '[$a, $r, .timings.prompt_n,
            .timings.predicted_n, (.timings.draft_n // 0), (.timings.draft_n_accepted // 0), $np, $nn, $nu,
            (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, ${vp:-no C738 line}; $(grep -c ':u[1-9]' "$d/server.log") ubatch-trace lines with inputs not copied again; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done
[ -n "$rpc_pid" ] && { kill "$rpc_pid" 2>/dev/null; wait "$rpc_pid" 2>/dev/null; }

# the verdict
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
col() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/summary.tsv"; }
{
    for a in off on1 on2; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    for a in boff bon1 bon2; do
        has bbase && has "$a" && { same bbase "$a" || fail=1; }
    done
    for a in ron1 ron2; do
        has roff && has "$a" && { same roff "$a" || fail=1; }
    done
    for a in on1 on2 bon1 bon2 ron1 ron2; do
        has "$a" || continue
        np=$(col "$a" 7)
        if [ "${np:-0}" -gt 0 ]; then
            echo "$a: $np verify passes prepared, $(col "$a" 8) calls not prepared, $(col "$a" 9) inputs uploaded ahead"
        else
            echo "$a: no verify pass prepared"; fail=1
        fi
    done
    if has ron2; then
        nu=$(col ron2 9)
        ns=$(grep -c ':u[1-9]' "$OUT/ron2/server.log")
        if [ "${nu:-0}" -gt 0 ] && [ "$ns" -gt 0 ]; then
            echo "ron2: $nu inputs uploaded ahead; $ns evaluations skipped copying some of them again"
        else
            echo "ron2: uploaded ${nu:-0} inputs ahead, $ns evaluations skipped copies: the uploads were not exercised"; fail=1
        fi
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
