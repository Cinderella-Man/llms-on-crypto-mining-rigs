#!/usr/bin/env bash
# C812 (Q1420): the MTP draft's scheduler copies with its token table on its layer's device (LLAMA_DRAFT_EMBD_DEV=layer,
# C762) and LLAMA_PP_MULTI_SPLIT=1, checked on a CPU with two devices. Made from C762's server_check.sh
# (docs/data/2026-11-11_c762_draft_rows/run/server_check.sh). A random-weight qwen4exp model with one MTP block, that
# block alone as a separate draft file (test-llama-archs -o --mtp-draft: its token embeddings in Q8_0, as Flash-Next's
# MTP draft), and two ggml-rpc-servers on this machine as two devices (they advertise asynchronous compute and events,
# so llama.cpp turns pipeline parallelism on as it does for the rig's cards). Every arm offloads every layer
# (-ngl 99 -ngld 99), so the draft's MTP layer is on the second device, as the Flash-Next preset's is on card 8.
# Arms (all -m MODEL -md DRAFT -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -fit off -lv 4,
# --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64; E = LLAMA_DRAFT_EMBD_DEV=layer,
# P = LLAMA_PP_MULTI_SPLIT=1):
#   otd      -otd token_embd=<first device>: the table on the first device, the layer on the second, as the v4 line
#            before R123b (-otd token_embd=CUDA4). A tensor override turns pipeline parallelism off for the draft
#   embd     E and no -otd, as R123b's embd rung: the table with the layer, the draft one split on the second device,
#            and pipeline parallelism on for it (the rig: 86.41 MiB of pinned host compute buffer against 21.60)
#   fix      embd's, and P: the draft planned without pipeline parallelism
#   otdfix   otd's, and P: the switch with the override (the draft already without pipeline parallelism, two splits)
#   bembd, bfix   embd and fix with -bs (the preset's backend sampler)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42); S2 = G2's prompt, the same with seed 7. Every arm writes LLAMA_SPEC_DUMP: one line per verify
# step with the draft, the drafter's candidates and their probabilities and the target's top tokens.
# Each "sched_reserve: reserving" block of each server log is one row of reserves.tsv: the context (the target's
# blocks come first; a later block with fewer than half the first block's nodes is the draft's), its nodes, splits,
# scheduler copies, the C812 line, and each compute buffer.
# Pass: embd, fix and otdfix give otd's tokens, top probabilities, draft counts and verify steps on every request, and
# bfix bembd's; the target's scheduler copies are 4 in every arm and no target block has the C812 line; the draft's are
# 1 in otd, 4 in embd and bembd, 1 in fix and bfix with the C812 line in the draft's first block (the reserve that turns
# pipeline parallelism off for the context; its later reserve has one copy from the start), and 1 in otdfix without it;
# the draft's host (CPU) compute buffer in fix is no larger than in otd and smaller than in embd (bfix the same against
# bembd); no arm logs an error.
# Usage: RPC_BIN=<dir with llama-server and ggml-rpc-server built with GGML_RPC> MODEL=<gguf with an MTP block>
#   DRAFT=<its MTP-only draft gguf> OUT=<dir> [PORT=18812] [RPC_PORT=50812 (and RPC_PORT+1)] [THREADS=8] [ARMS='...']
#   [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl}, OUT/summary.tsv,
#   OUT/reserves.tsv, OUT/verdict.txt
set -uo pipefail
RPC_BIN=${RPC_BIN:?RPC_BIN=<dir with llama-server and ggml-rpc-server built with GGML_RPC>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
DRAFT=${DRAFT:?DRAFT=<its MTP-only draft gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18812}
RPC_PORT=${RPC_PORT:-50812}
THREADS=${THREADS:-8}
ARMS=${ARMS:-otd embd fix otdfix bembd bfix}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64"
REQS="G1 G2 S1 S2"
EP1=127.0.0.1:$RPC_PORT
EP2=127.0.0.1:$((RPC_PORT + 1))
# an RPC buffer type is named RPC<device index on its server>[<endpoint>]; each server here has one device
BUFT1="RPC0[$EP1]"
RFLAGS="--rpc $EP1,$EP2 -ngl 99 -ngld 99"
OTD="-otd token_embd=$BUFT1"
E=LLAMA_DRAFT_EMBD_DEV=layer
P=LLAMA_PP_MULTI_SPLIT=1
mkdir -p "$OUT"

declare -A FLAGS=(
    [otd]="$RFLAGS $OTD" [embd]="$RFLAGS" [fix]="$RFLAGS" [otdfix]="$RFLAGS $OTD" [bembd]="$RFLAGS -bs" [bfix]="$RFLAGS -bs"
)
declare -A ENVS=(
    [embd]=$E [fix]="$E $P" [otdfix]=$P [bembd]=$E [bfix]="$E $P"
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

# <arm> <server.log>: one row per reserve block (arm, n, ctx, nodes, splits, copies, c812, buffers)
reserves() {
    awk -v arm="$1" '
        function flush() { if (!open) return; n++
            if (n == 1) first_nodes = nodes + 0
            ctx = (n == 1 || nodes * 2 >= first_nodes) ? "target" : "draft"
            print arm "\t" n "\t" ctx "\t" nodes "\t" splits "\t" copies "\t" c812 "\t" bufs; open = 0 }
        /sched_reserve: reserving \.\.\./ { flush(); open = 1; nodes = ""; splits = ""; copies = ""; c812 = "no"; bufs = ""; next }
        !open { next }
        /LLAMA_PP_MULTI_SPLIT: the graph runs as one split/ { c812 = "yes" }
        /sched_reserve: .* compute buffer size = / { b = $0; sub(/.*sched_reserve: +/, "", b); sub(/ compute buffer size = +/, "=", b); sub(/ MiB.*/, "", b)
            bufs = bufs (bufs == "" ? "" : ", ") b }
        /sched_reserve: graph: nodes = / { x = $0; sub(/.*nodes = /, "", x); sub(/,.*/, "", x); nodes = x + 0
            y = $0; sub(/.*splits = /, "", y); sub(/,.*/, "", y); splits = y }
        /sched_reserve: reserve took .* sched copies = / { z = $0; sub(/.*sched copies = /, "", z); copies = z + 0; flush() }
        END { flush() }' "$2"
}

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\ttokens\n' > "$OUT/summary.tsv"
printf 'arm\tblock\tcontext\tnodes\tsplits\tsched_copies\tc812_line\tcompute_buffers_mib\n' > "$OUT/reserves.tsv"

"$RPC_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$RPC_PORT" -t "$THREADS" > "$OUT/ggml-rpc-server-1.log" 2>&1 &
rpc_pids=$!
"$RPC_BIN/ggml-rpc-server" -H 127.0.0.1 -p "$((RPC_PORT + 1))" -t "$THREADS" > "$OUT/ggml-rpc-server-2.log" 2>&1 &
rpc_pids="$rpc_pids $!"
sleep 3

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "$RPC_BIN/llama-server" -m "$MODEL" -md "$DRAFT" -t "$THREADS" \
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
    reserves "$arm" "$d/server.log" >> "$OUT/reserves.tsv"
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps; $(grep -a -c 'sched_reserve: reserving' "$d/server.log") reserves; $(grep -a -c 'LLAMA_PP_MULTI_SPLIT' "$d/server.log") C812 lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
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
# <arm> <context> <column>: the distinct values of a column over that context's blocks
vals() { awk -F'\t' -v a="$1" -v c="$2" -v k="$3" '$1 == a && $3 == c { print $k }' "$OUT/reserves.tsv" | sort -u | tr '\n' ' ' | sed 's/ $//'; }
# <arm>: the draft's host compute buffer, the largest over its blocks (MiB)
host() { awk -F'\t' -v a="$1" '$1 == a && $3 == "draft" { n = split($8, b, ", "); for (i = 1; i <= n; i++) if (b[i] ~ /^CPU=/) { v = substr(b[i], 5) + 0; if (v > m) m = v } } END { printf "%.2f", m }' "$OUT/reserves.tsv"; }
{
    for a in embd fix otdfix; do
        has otd && has "$a" && { same otd "$a" || fail=1; }
    done
    has bembd && has bfix && { same bembd bfix || fail=1; }
    # the draft's scheduler copies over all its blocks, and the C812 line in each of its blocks in order: with the
    # switch the first reserve logs it and turns pipeline parallelism off for the context, so a later reserve has one
    # copy from the start and no line
    declare -A WANT=([otd]="1 no no" [embd]="4 no no" [fix]="1 yes no" [otdfix]="1 no no" [bembd]="4 no no" [bfix]="1 yes no")
    for a in $ARMS; do
        tc=$(vals "$a" target 6) tl=$(vals "$a" target 7) dc=$(vals "$a" draft 6)
        dl=$(awk -F'\t' -v a="$a" '$1 == a && $3 == "draft" { print $7 }' "$OUT/reserves.tsv" | tr '\n' ' ' | sed 's/ $//')
        want=${WANT[$a]}
        if [ "$tc" = 4 ] && [ "$tl" = no ] && [ "$dc" = "${want%% *}" ] && [ "$dl" = "${want#* }" ]; then
            echo "$a: the target's scheduler copies $tc, the draft's $dc (C812 line in its blocks, in order: $dl), draft splits $(vals "$a" draft 5), host compute buffer $(host "$a") MiB"
        else
            echo "$a: the target's copies '$tc' (C812 line '$tl'), the draft's '$dc' (C812 line in its blocks '$dl'); expected 4 (no) and ${want%% *} (${want#* })"; fail=1
        fi
    done
    cmpbuf() { # <smaller or equal> <reference> [strict]
        local x y w="no larger"
        x=$(host "$1") y=$(host "$2")
        [ -n "${3:-}" ] && w=smaller
        if awk -v x="$x" -v y="$y" -v s="${3:-}" 'BEGIN { exit !(s ? x < y : x <= y) }'; then
            echo "the draft's host compute buffer: $1 $x MiB, $2 $y MiB ($w)"
        else
            echo "the draft's host compute buffer: $1 $x MiB against $2's $y MiB: not $w"; fail=1
        fi
    }
    has fix && has otd && cmpbuf fix otd
    has fix && has embd && cmpbuf fix embd strict
    has bfix && has bembd && cmpbuf bfix bembd strict
    for a in $ARMS; do
        [ -f "$OUT/$a/server.log" ] || continue
        e=$(grep -ciE 'error|abort' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
