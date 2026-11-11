#!/usr/bin/env bash
# C762 (Q1340): llama-server's MTP draft with its token embeddings on its layer's device (LLAMA_DRAFT_EMBD_DEV=layer),
# checked on the CPU with a random-weight qwen4exp model with one MTP block and that block alone as a separate draft
# file (test-llama-archs -o --mtp-draft: the token embeddings in Q8_0, as Flash-Next's MTP draft). Made from C738's
# server_check.sh (docs/data/2026-11-11_c738_verify_prepare). BASE is the base branch's llama-server, BIN the branch's;
# RPC_BIN a llama-server and ggml-rpc-server built with GGML_RPC, whose arms run two ggml-rpc-servers on this machine as
# two devices and lay the draft out as the Flash-Next preset does on the cards: its MTP layer on the second device, its
# token embeddings on the first (-otd token_embd=<first>, the preset's -otd token_embd=CUDA4 with the layer on card 8).
# Arms (all -m MODEL -md DRAFT -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -fit off -lv 4 (the
# library's INFO lines: the switch's and the hashes), --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
# --spec-draft-n-ubatch 64; E = LLAMA_DRAFT_EMBD_DEV=layer):
#   base   BASE on the CPU: the reference of off and on
#   off    BIN on the CPU
#   on     BIN on the CPU, E (the layer is on the CPU, so the table stays there: the switch's path, the same placement)
#   roff   RPC_BIN, --rpc <first>,<second> -ngl 99 -ngld 99 -otd token_embd=<first's buffer type>: the reference of
#          the r arms
#   ron    roff's flags, E: the table on the second device, in place of the -otd rule
#   rboff, rbon   roff and ron with -bs (the preset's backend sampler)
#   rhoff, rhon   roff and ron with LLAMA_TENSOR_HASH=mtp_: every node of the draft's graph whose name starts with
#          mtp_ (its token rows mtp_tok_embd, the norms of the rows and of the target's hidden row, eh_proj) hashed after
#          each pass, with the buffer that holds it
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens.
# Pass: off and on give base's tokens, top probabilities, draft counts (draft_n, draft_n_accepted) and verify steps
# (the dump without its run, step and task fields) on every request; ron = roff, rbon = rboff, rhon = rhoff the same;
# every E arm logs where the table went (the second device in the r arms, the CPU in on) and each r arm with E that
# the -otd rule was not applied, the arms without E none of it; rhon's hashed draft nodes are rhoff's, node for node,
# with mtp_tok_embd in the first device's buffer in rhoff and in the second's in rhon; no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> RPC_BIN=<dir with llama-server
#   and ggml-rpc-server built with GGML_RPC> MODEL=<gguf with an MTP block> DRAFT=<its MTP-only draft gguf> OUT=<dir>
#   [PORT=18762] [RPC_PORT=50762 (and RPC_PORT+1)] [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>']
#   bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,hash.tsv}, OUT/summary.tsv,
#   OUT/placement.tsv, OUT/hash_nodes.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
RPC_BIN=${RPC_BIN:?RPC_BIN=<dir with llama-server and ggml-rpc-server built with GGML_RPC>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
DRAFT=${DRAFT:?DRAFT=<its MTP-only draft gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18762}
RPC_PORT=${RPC_PORT:-50762}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on roff ron rboff rbon rhoff rhon}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64"
REQS="G1 G2 S1 S2"
EP1=127.0.0.1:$RPC_PORT
EP2=127.0.0.1:$((RPC_PORT + 1))
# an RPC buffer type is named RPC<device index on its server>[<endpoint>]; each server here has one device
BUFT1="RPC0[$EP1]"
BUFT2="RPC0[$EP2]"
RFLAGS="--rpc $EP1,$EP2 -ngl 99 -ngld 99 -otd token_embd=$BUFT1"
E=LLAMA_DRAFT_EMBD_DEV=layer
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [on]=$BIN
    [roff]=$RPC_BIN [ron]=$RPC_BIN [rboff]=$RPC_BIN [rbon]=$RPC_BIN [rhoff]=$RPC_BIN [rhon]=$RPC_BIN
)
declare -A FLAGS=(
    [roff]="$RFLAGS" [ron]="$RFLAGS" [rboff]="$RFLAGS -bs" [rbon]="$RFLAGS -bs" [rhoff]="$RFLAGS" [rhon]="$RFLAGS"
)
declare -A ENVS=(
    [on]=$E [ron]=$E [rbon]=$E [rhoff]=LLAMA_TENSOR_HASH=mtp_ [rhon]="$E LLAMA_TENSOR_HASH=mtp_"
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
printf 'arm\tswitch_lines\ttable_in\toverride_not_applied\thashed_draft_nodes\ttok_embd_buffers\n' > "$OUT/placement.tsv"

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
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -md "$DRAFT" -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -fit off -lv 4 --no-webui --host 127.0.0.1 --port "$PORT" \
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

    # SIGTERM and wait: the dump is whole
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    # the draft context's hashed nodes: n, name, type, shape, buffer, hash; read from the line's end, since a view's name
    # has a space ("mtp_h_input (reshaped)", written with "_" here) and a node not read ends in "(not read)"
    grep -a -o 'tensor-hash: d .*' "$d/server.log" | awk '{ u = ($NF == "read)"); if (u) NF -= 2; h = u ? "unread" : $NF
        name = $4; for (i = 5; i <= NF - 4; i++) name = name "_" $i
        print $3 "\t" name "\t" $(NF - 3) "\t" $(NF - 2) "\t" $(NF - 1) "\t" h }' > "$d/hash.tsv"
    sw=$(grep -a -c 'LLAMA_DRAFT_EMBD_DEV' "$d/server.log")
    where=$(grep -a -o 'token_embd.weight ([^)]*) in [^ ]*' "$d/server.log" | sed 's/.* in //' | sort -u | tr '\n' ' ')
    notapp=$(grep -a -o 'the override to [^ ]* is not applied' "$d/server.log" | sed 's/the override to //; s/ is not applied//' | sort -u | tr '\n' ' ')
    tbufs=$(awk -F'\t' '$2 ~ /^mtp_tok_embd/ { print $5 }' "$d/hash.tsv" | sort | uniq -c | awk '{ printf "%s%s x%s", (NR > 1 ? ", " : ""), $2, $1 }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$sw" "${where:--}" "${notapp:--}" "$(wc -l < "$d/hash.tsv")" "${tbufs:--}" >> "$OUT/placement.tsv"
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $sw switch lines, table in ${where:-its default place}, $(wc -l < "$d/hash.tsv") hashed draft nodes; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
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
col() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/placement.tsv"; }
{
    for a in off on; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    for p in "roff ron" "rboff rbon" "rhoff rhon"; do
        set -- $p
        has "$1" && has "$2" && { same "$1" "$2" || fail=1; }
    done
    # where the table went
    for a in base off roff rboff rhoff; do
        has "$a" || continue
        if [ "$(col "$a" 2)" = 0 ]; then
            echo "$a: no LLAMA_DRAFT_EMBD_DEV line (the table where it was)"
        else
            echo "$a: $(col "$a" 2) LLAMA_DRAFT_EMBD_DEV lines without the switch"; fail=1
        fi
    done
    if has on; then
        if [[ "$(col on 3)" == CPU* ]]; then
            echo "on: the table with the layer on the CPU ($(col on 3))"
        else
            echo "on: the table in '$(col on 3)', not the CPU"; fail=1
        fi
    fi
    for a in ron rbon rhon; do
        has "$a" || continue
        if [ "$(col "$a" 3)" = "$BUFT2 " ] && [ "$(col "$a" 4)" = "$BUFT1 " ]; then
            echo "$a: the table in $BUFT2 with the layer; the override to $BUFT1 not applied"
        else
            echo "$a: the table in '$(col "$a" 3)', override not applied: '$(col "$a" 4)'"; fail=1
        fi
    done
    # the hashed draft nodes: the same bytes node for node, the token rows on the other device; per node and shape in
    # hash_nodes.tsv (the passes, each arm's buffer, how many have the same index, name, type, shape and bytes)
    if has rhoff && has rhon; then
        { printf 'node\tshape\tpasses\trhoff_buffer\trhon_buffer\tsame_bytes\n'
          paste "$OUT/rhoff/hash.tsv" "$OUT/rhon/hash.tsv" | awk -F'\t' 'BEGIN { OFS = "\t" }
              { name = $2; sub(/-[0-9]+$/, "", name); k = name SUBSEP $4 SUBSEP $5 SUBSEP $11; n[k]++
                if ($1 == $7 && $2 == $8 && $3 == $9 && $4 == $10 && $6 == $12 && $6 != "unread") ok[k]++ }
              END { for (k in n) { split(k, a, SUBSEP); print a[1], a[2], n[k], a[3], a[4], (ok[k] == n[k] ? "all" : ok[k] + 0) } }' \
              | sort -t$'\t' -k1,1 -k2,2; } > "$OUT/hash_nodes.tsv"
        n=$(wc -l < "$OUT/rhoff/hash.tsv")
        if [ "$n" -gt 0 ] && cmp -s <(cut -f1-4,6 "$OUT/rhoff/hash.tsv") <(cut -f1-4,6 "$OUT/rhon/hash.tsv"); then
            echo "rhon = rhoff: $n hashed draft nodes, the same names, types, shapes and bytes, $(awk -F'\t' '$2 ~ /^mtp_tok_embd/' "$OUT/rhon/hash.tsv" | wc -l) of them mtp_tok_embd"
        else
            echo "rhon's hashed draft nodes differ from rhoff's ($n and $(wc -l < "$OUT/rhon/hash.tsv") lines)"; fail=1
        fi
        b0=$(awk -F'\t' '$2 ~ /^mtp_tok_embd/ { print $5 }' "$OUT/rhoff/hash.tsv" | sort -u | tr '\n' ' ')
        b1=$(awk -F'\t' '$2 ~ /^mtp_tok_embd/ { print $5 }' "$OUT/rhon/hash.tsv" | sort -u | tr '\n' ' ')
        if [[ "$b0" == *"$EP1"* && "$b0" != *"$EP2"* && "$b1" == *"$EP2"* && "$b1" != *"$EP1"* ]]; then
            echo "mtp_tok_embd gathered in ${b0% } (rhoff) and ${b1% } (rhon)"
        else
            echo "mtp_tok_embd's buffers: rhoff '$b0', rhon '$b1' (expected the first device's and the second's)"; fail=1
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
