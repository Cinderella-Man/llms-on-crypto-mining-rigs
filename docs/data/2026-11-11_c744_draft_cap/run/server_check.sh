#!/usr/bin/env bash
# C744 (Q1314): the MTP drafter stops at the slot's draft cap (LLAMA_MTP_DRAFT_CAP=1), checked on the CPU with a
# random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp). Made from C741's server_check.sh
# (docs/data/2026-11-11_c741_server_yields). BASE is the base branch's llama-server, BIN the branch's.
# One slot (-np 1; -c 8192 -b 512 -ub 128 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0, as the release preset drafts; CAP = LLAMA_MTP_DRAFT_CAP=1):
#   base     BASE: the reference
#   off      BIN, CAP unset
#   cap      BIN, CAP
#   capnt    BIN, CAP, without the step trace (every other arm writes LLAMA_STEP_TRACE)
#   bbase    BASE, -bs (the backend sampler, as the preset): the reference of the b arms
#   boff     BIN, -bs
#   bcap     BIN, -bs, CAP
#   pbase    BASE, --spec-draft-sampling probabilistic: the reference of pcap
#   pcap     BIN, the same, CAP (the cap refuses a probabilistic draft: S1 and S2; G1 and G2 draft greedily)
#   nbase    BASE, --spec-draft-n-min 2: the reference of ncap
#   ncap     BIN, the same, CAP (the cap refuses a cap below n_min: the step with 2 tokens left)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Two slots (-np 2, --metrics, the step trace), C741's run/two_slots.py: A = G2's request, B = G1's posted once A has 16
# tokens, /slots and /metrics read in turn every 2 ms while A or B runs (B's last steps are capped while A drafts in full):
#   p2base   BASE
#   p2off    BIN, CAP unset
#   p2cap    BIN, CAP
# Every arm writes LLAMA_SPEC_DUMP: one line per slot and verify step with the draft, the drafter's candidates (one row
# per drafted token, the cut ones included) and the target's top tokens. steps.jsonl is the dump without its run, step
# and task fields; cut.jsonl the same with the candidate rows cut to the draft's length (what the cap leaves out is the
# rows of tokens the server cuts).
# Pass: off and boff are base's and bbase's replies (tokens, top probabilities, draft counts) and steps.jsonl; cap,
# capnt, bcap, pcap, ncap are their reference's replies and cut.jsonl; base, bbase, pbase, nbase have cut rows and cap,
# capnt, bcap have none, pcap and ncap keep some (refused); in every capped arm the passes the cap left out (the trace's
# "cap" count) are the reference's spare passes less the arm's, and its cut rows the reference's less the arm's; in every
# two-slot arm B came while A ran, both were answered with no failed read, and A and B are base's G2 and G1; no arm logs
# an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> TWO_SLOTS=<C741's two_slots.py> [PORT=18744] [THREADS=8] [ARMS='base off ...'] [SARGS='<more arguments>']
#   bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,cut.jsonl,
#   step_trace.jsonl}, the two-slot arms' OUT/<arm>/{A,B}.json, reads.tsv, client.json; OUT/summary.tsv,
#   OUT/passes.md, OUT/passes.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
TWO_SLOTS=${TWO_SLOTS:?TWO_SLOTS=<two_slots.py of C741>}
PORT=${PORT:-18744}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off cap capnt bbase boff bcap pbase pcap nbase ncap p2base p2off p2cap}
SARGS=${SARGS:-}
RUN=$(dirname "$(realpath "$0")")
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
CAP=LLAMA_MTP_DRAFT_CAP=1
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [cap]=$BIN [capnt]=$BIN
    [bbase]=$BASE [boff]=$BIN [bcap]=$BIN
    [pbase]=$BASE [pcap]=$BIN
    [nbase]=$BASE [ncap]=$BIN
    [p2base]=$BASE [p2off]=$BIN [p2cap]=$BIN
)
declare -A FLAGS=(
    [bbase]="-bs" [boff]="-bs" [bcap]="-bs"
    [pbase]="--spec-draft-sampling probabilistic" [pcap]="--spec-draft-sampling probabilistic"
    [nbase]="--spec-draft-n-min 2" [ncap]="--spec-draft-n-min 2"
)
declare -A ENVS=(
    [cap]=$CAP [capnt]=$CAP [bcap]=$CAP [pcap]=$CAP [ncap]=$CAP [p2cap]=$CAP
)
# the reference of each arm of the branch, and whether it is compared in full (off) or with the cut rows left out
declare -A REF=(
    [off]=base [cap]=base [capnt]=base [boff]=bbase [bcap]=bbase [pcap]=pbase [ncap]=nbase
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (a Park-Miller generator, exact in awk's
# doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

body() { # <file> <comma-separated tokens> <n_predict> <temperature> <seed>
    printf '{"prompt":[%s],"n_predict":%s,"temperature":%s,"top_k":20,"top_p":0.95,"min_p":0,"seed":%s,"n_probs":3,"return_tokens":true,"cache_prompt":false}' \
        "$2" "$3" "$4" "$5" > "$1"
}

complete() { # <arm dir> <name> <comma-separated tokens> <n_predict> <temperature> <seed>
    body "$1/$2.req.json" "$3" "$4" "$5" "$6"
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
    np=1 tr="LLAMA_STEP_TRACE=$d/step_trace.jsonl" more=
    [[ $arm == p2* ]] && np=2 more=--metrics
    [ "$arm" = capnt ] && tr=
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" $tr ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np $np --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" \
        $MTP $SARGS ${FLAGS[$arm]:-} $more > "$d/server.log" 2>&1 &
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
    if [ $np = 1 ]; then
        complete "$d" G1 "$p1" 96 0 1
        complete "$d" G2 "$p2" 128 0 1
        complete "$d" S1 "$p1" 96 1.0 42
        complete "$d" S2 "$p2" 96 1.0 7
        reqs=$REQS
    else
        body "$d/A.req.json" "$p2" 128 0 1
        body "$d/B.req.json" "$p1" 96 0 1
        python3 -I "$TWO_SLOTS" --port "$PORT" --a "$d/A.req.json" --b "$d/B.req.json" --out "$d" --after 16 \
            --poll-ms 2 > "$d/client.out" 2>&1
        rm -f "$d/A.req.json" "$d/B.req.json"
        reqs="A B"
    fi

    # SIGTERM and wait: the dump and the trace are whole, and the drafter logs its total at its end
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $reqs; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id; then with the candidate rows of
    # the tokens the server cut left out
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    jq -c '(.draft | length) as $n | if .cand then .cand |= .[:$n] | .p |= .[:$n] | .pick |= .[:$n] else . end' \
        "$d/steps.jsonl" > "$d/cut.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(cat "$d/step_trace.jsonl" 2>/dev/null | wc -l) trace lines; $(grep -c 'a greedy draft stops at its slot' "$d/server.log") switch lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the table
loads=
for a in $ARMS; do
    [ -s "$OUT/$a/steps.jsonl" ] && loads="$loads $a=$OUT/$a"
done
# shellcheck disable=SC2086
python3 -I "$RUN/passes.py" --tsv "$OUT/passes.tsv" $loads > "$OUT/passes.md"

# the verdict
fail=0
fields='[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]'
same_req() { # <arm a> <req a> <arm b> <req b>
    [ -s "$OUT/$1/$2.json" ] && [ -s "$OUT/$3/$4.json" ] && \
        cmp -s <(jq -c "$fields" "$OUT/$1/$2.json") <(jq -c "$fields" "$OUT/$3/$4.json")
}
same() { # <reference> <arm> <steps file>: tokens, top probabilities and draft counts of every request, and the verify steps
    local r
    for r in $REQS; do
        if ! same_req "$1" "$r" "$2" "$r"; then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    if [ ! -s "$OUT/$1/$3" ] || ! cmp -s "$OUT/$1/$3" "$OUT/$2/$3"; then
        echo "$2's verify steps ($3) differ from $1's (or $1 wrote none)"; return 1
    fi
    echo "$2 = $1 (tokens, top probabilities and draft counts of $REQS; $(wc -l < "$OUT/$2/$3") verify steps the same in $3)"
}
has() { [[ " $ARMS " == *" $1 "* ]]; }
col() { awk -F'\t' -v a="$1" -v c="$2" 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next } $1 == a { print $h[c]; exit }' "$OUT/passes.tsv"; }
{
    for a in off boff cap capnt bcap pcap ncap; do
        r=${REF[$a]}
        has "$a" && has "$r" || continue
        if [[ $a == off || $a == boff ]]; then
            same "$r" "$a" steps.jsonl || fail=1
        else
            same "$r" "$a" cut.jsonl || fail=1
        fi
    done
    # the passes: the references cut rows, the capped arms leave out exactly the passes and rows the cap took
    for a in base bbase pbase nbase off boff; do
        has "$a" || continue
        cr=$(col "$a" cut_rows)
        if [ -n "$cr" ] && [ "$cr" -gt 0 ] && [ "$(col "$a" cap)" = 0 ]; then
            echo "$a: $cr cut rows in $(col "$a" cut_lines) verify steps, $(col "$a" spare) spare passes in $(col "$a" spare_lines) steps; no pass left out"
        else
            echo "$a: cut rows ${cr:-none} or cap $(col "$a" cap) (expected cut rows and no cap)"; fail=1
        fi
    done
    for a in cap capnt bcap pcap ncap; do
        r=${REF[$a]}
        has "$a" && has "$r" || continue
        cr=$(col "$a" cut_rows) cr0=$(col "$r" cut_rows)
        # capnt writes no trace: its passes are the cut rows it no longer has
        if [ "$a" = capnt ]; then
            sp=0 sp0=$(( cr0 - cr )) cp=$(( cr0 - cr ))
        else
            sp=$(col "$a" spare) sp0=$(col "$r" spare) cp=$(col "$a" cap)
        fi
        want=nonzero
        [[ $a == pcap || $a == ncap ]] || want=zero
        if [ -z "$cr" ] || [ -z "$cr0" ] || [ $(( cr0 - cr )) != "$cp" ] || [ $(( sp0 - sp )) != "$cp" ] || [ "$cp" -le 0 ] || \
                { [ $want = zero ] && [ "$cr" != 0 ]; } || { [ $want = nonzero ] && [ "$cr" = 0 ]; }; then
            echo "$a: cut rows $cr (of $r's $cr0), spare passes $sp (of $sp0), left out $cp: not the reference's less the cap"; fail=1
        else
            echo "$a: the cap left out $cp passes; cut rows $cr0 -> $cr, spare passes $sp0 -> $sp$( [ $want = nonzero ] && echo ' (the refused drafts keep theirs)')"
        fi
        if [ "$a" != capnt ]; then
            lg=$(grep -o 'draft length cap: [0-9]* drafting passes left out in [0-9]* drafts' "$OUT/$a/server.log" | tail -n 1)
            echo "$a: log: ${lg:-no total line}"
        fi
    done
    for a in p2base p2off p2cap; do
        has "$a" || continue
        d=$OUT/$a
        if ! jq -e '.b_during_a and .a_status == 200 and .b_status == 200 and .reads_failed == 0' "$d/client.json" > /dev/null 2>&1; then
            echo "$a: B did not come during A, a reply failed or a read failed ($(cat "$d/client.json" 2>/dev/null | tr -d '\n '))"; fail=1; continue
        fi
        ok=1
        has base && { same_req base G2 "$a" A && same_req base G1 "$a" B || ok=0; }
        if [ $ok = 1 ]; then
            echo "$a: A = base's G2 and B = base's G1 (tokens, top probabilities, draft counts); $(jq -r '.reads' "$d/client.json") reads answered; cut rows $(col "$a" cut_rows), passes $(col "$a" passes), left out $(col "$a" cap)"
        else
            echo "$a: A or B differs from base's one-slot reply"; fail=1
        fi
    done
    if has p2cap && [ "$(col p2cap cut_rows)" != 0 ]; then
        echo "p2cap: $(col p2cap cut_rows) cut rows (expected none)"; fail=1
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
