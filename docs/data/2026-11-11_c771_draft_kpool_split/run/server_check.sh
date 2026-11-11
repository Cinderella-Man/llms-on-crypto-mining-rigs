#!/usr/bin/env bash
# C771 (Q1357): llama-server's MTP draft with LLAMA_MTP_EMBD_FIRST=1 (the draft's token-row gather built first, so the
# k-pool tables' update of LLAMA_QSA_KPOOL_DEV joins its layer's split), checked on the CPU with a random-weight
# qwen4exp model with one MTP block and that block alone as a separate draft file (test-llama-archs -o --mtp-draft, as
# Flash-Next's MTP draft). Made from C762's server_check.sh (docs/data/2026-11-11_c762_draft_rows). BASE is the base
# branch's llama-server, BIN the branch's; RPC_BIN a llama-server and ggml-rpc-server built with GGML_RPC, whose arms run
# two ggml-rpc-servers on this machine as two devices and lay the draft out as the Flash-Next preset does on the cards:
# its MTP layer on the second device (card 8), its token embeddings on the first (-otd token_embd=<first>, the preset's
# -otd token_embd=CUDA4).
# Arms (all -m MODEL -md DRAFT -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -fit off -lv 4,
# --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64; E = LLAMA_MTP_EMBD_FIRST=1,
# KD = LLAMA_QSA_KPOOL_DEV=1):
#   base   BASE on the CPU: the reference of off and on
#   off    BIN on the CPU
#   on     BIN on the CPU, E
#   the RPC arms, each with LLAMA_UBATCH_TRACE=1 (the scheduler's line per evaluation: every split's device and nodes)
#   and in pairs without and with E:
#   r, re       nothing else
#   rd, rde     KD (R74's dev load)
#   rdc, rdce   KD, LLAMA_QSA_KPOOL_DEV_CHECK=1 (C100's check: every micro-batch's copies read back and compared)
#   rds, rdse   KD, LLAMA_QSA_KPOOL_STAGE=1 (C720's stage)
#   rdk, rdke   KD, LLAMA_MTP_KV_ONLY=1 (C620's catch-up passes without an output row)
#   rdb, rdbe   KD, -bs (the preset's backend sampler)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens.
# Pass: off and on give base's tokens, top probabilities, draft counts (draft_n, draft_n_accepted) and verify steps
# (the dump without its run, step and task fields) on every request; each E arm the same as its pair; splits.py's
# verdict on every pair (the draft's passes one to one, each E pass with one split on the second device, the same nodes
# on each device, the target's passes unchanged); no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> RPC_BIN=<dir with llama-server
#   and ggml-rpc-server built with GGML_RPC> MODEL=<gguf with an MTP block> DRAFT=<its MTP-only draft gguf> OUT=<dir>
#   [PORT=18771] [RPC_PORT=50771 (and RPC_PORT+1)] [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>']
#   bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,trace.txt},
#   OUT/summary.tsv, OUT/kpool.tsv, OUT/splits.tsv, OUT/splits_verdict.txt, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
RPC_BIN=${RPC_BIN:?RPC_BIN=<dir with llama-server and ggml-rpc-server built with GGML_RPC>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
DRAFT=${DRAFT:?DRAFT=<its MTP-only draft gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18771}
RPC_PORT=${RPC_PORT:-50771}
THREADS=${THREADS:-8}
PAIRS="r:re rd:rde rdc:rdce rds:rdse rdk:rdke rdb:rdbe"
ARMS=${ARMS:-base off on r re rd rde rdc rdce rds rdse rdk rdke rdb rdbe}
SARGS=${SARGS:-}
RUN=$(dirname "$(realpath "$0")")
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64"
REQS="G1 G2 S1 S2"
EP1=127.0.0.1:$RPC_PORT
EP2=127.0.0.1:$((RPC_PORT + 1))
# an RPC buffer type is named RPC<device index on its server>[<endpoint>]; each server here has one device
BUFT1="RPC0[$EP1]"
RFLAGS="--rpc $EP1,$EP2 -ngl 99 -ngld 99 -otd token_embd=$BUFT1"
E=LLAMA_MTP_EMBD_FIRST=1
KD=LLAMA_QSA_KPOOL_DEV=1
T=LLAMA_UBATCH_TRACE=1
mkdir -p "$OUT"

declare -A ENVS=(
    [on]=$E
    [r]=$T [re]="$T $E"
    [rd]="$T $KD" [rde]="$T $KD $E"
    [rdc]="$T $KD LLAMA_QSA_KPOOL_DEV_CHECK=1" [rdce]="$T $KD LLAMA_QSA_KPOOL_DEV_CHECK=1 $E"
    [rds]="$T $KD LLAMA_QSA_KPOOL_STAGE=1" [rdse]="$T $KD LLAMA_QSA_KPOOL_STAGE=1 $E"
    [rdk]="$T $KD LLAMA_MTP_KV_ONLY=1" [rdke]="$T $KD LLAMA_MTP_KV_ONLY=1 $E"
    [rdb]="$T $KD" [rdbe]="$T $KD $E"
)
exe()   { case $1 in base) echo "$BASE" ;; off|on) echo "$BIN" ;; *) echo "$RPC_BIN" ;; esac; }
flags() { case $1 in rdb|rdbe) echo "$RFLAGS -bs" ;; r*) echo "$RFLAGS" ;; esac; }

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
printf 'arm\tdevices\tubatches\tfull\trows_uploaded\trows_read\tchecked\tcheck_lines\n' > "$OUT/kpool.tsv"

rpc_pids=
for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    if [[ $arm == r* ]] && [ -z "$rpc_pids" ]; then
        "$RPC_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$RPC_PORT" -t "$THREADS" > "$OUT/ggml-rpc-server-1.log" 2>&1 &
        rpc_pids=$!
        "$RPC_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$((RPC_PORT + 1))" -t "$THREADS" > "$OUT/ggml-rpc-server-2.log" 2>&1 &
        rpc_pids="$rpc_pids $!"
        sleep 3
    fi
    # shellcheck disable=SC2046,SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "$(exe "$arm")/llama-server" -m "$MODEL" -md "$DRAFT" -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -fit off -lv 4 --no-webui --host 127.0.0.1 --port "$PORT" \
        $MTP $SARGS $(flags "$arm") > "$d/server.log" 2>&1 &
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

    # SIGTERM and wait: the dump is whole and the memories log their counts when freed
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    # the scheduler's and the context's trace lines, in their order
    grep -a -o -E '(sched-trace|ubatch-trace): .*' "$d/server.log" > "$d/trace.txt"
    # the k-pool memories' counts when freed (the target's and the draft's), and the check's lines
    grep -a -o 'k-pool tables on [0-9]* devices: .*' "$d/server.log" | while read -r line; do
        set -- $line
        # k-pool tables on N devices: U ubatches, F full, R of D rows uploaded (x a ubatch), C checked
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$4" "$6" "$8" "${10}" "${12}" "${18}" \
            "$(grep -a -c 'ubatches checked: .* copies identical' "$d/server.log")" >> "$OUT/kpool.tsv"
    done
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(grep -a -c '^sched-trace' "$d/trace.txt") scheduler lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done
for p in $rpc_pids; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done

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
{
    for a in off on; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    for p in $PAIRS; do
        a=${p%%:*} b=${p#*:}
        has "$a" && has "$b" && { same "$a" "$b" || fail=1; }
    done
    # not part of the verdict: the two-device arms against the CPU's (the same CPU code on both devices)
    if has base && has r; then
        same base r > /dev/null && echo "(r = base as well)" || echo "(r differs from base: not part of the verdict)"
    fi
    pargs=
    for p in $PAIRS; do
        a=${p%%:*} b=${p#*:}
        has "$a" && has "$b" && pargs="$pargs $a:$OUT/$a/trace.txt:$b:$OUT/$b/trace.txt"
    done
    # shellcheck disable=SC2086
    python3 -I "$RUN/splits.py" --second "$EP2" --tsv "$OUT/splits.tsv" $pargs > "$OUT/splits_verdict.txt" 2>&1 || fail=1
    cat "$OUT/splits_verdict.txt"
    for a in $ARMS; do
        [ -f "$OUT/$a/server.log" ] || continue
        e=$(grep -ciE 'error|abort' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
