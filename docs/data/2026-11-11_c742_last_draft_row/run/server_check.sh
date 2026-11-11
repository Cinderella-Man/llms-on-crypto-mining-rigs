#!/usr/bin/env bash
# C742 (Q1311): a step's last MTP drafting pass copies no hidden row to the host (LLAMA_MTP_SKIP_LAST_ROW=1), checked on
# the CPU with a random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp). Made from C744's
# server_check.sh (docs/data/2026-11-11_c744_draft_cap). BASE is the base branch's llama-server, BIN the branch's.
# One slot (-np 1; -c 8192 -b 512 -ub 128 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0, as the release preset drafts; SKIP = LLAMA_MTP_SKIP_LAST_ROW=1):
#   base     BASE: the reference
#   off      BIN, SKIP unset
#   on       BIN, SKIP
#   onnt     BIN, SKIP, without the step trace (every other arm writes LLAMA_STEP_TRACE)
#   bbase    BASE, -bs (the backend sampler, as the preset): the reference of the b arms
#   boff     BIN, -bs
#   bon      BIN, -bs, SKIP
#   pbase    BASE, --spec-draft-sampling probabilistic: the reference of pon
#   pon      BIN, the same, SKIP (the last pass still samples: only its row's copy goes)
#   mbase    BASE, --spec-draft-p-min 0.1022 (the random model's drafter puts ~0.102 on its top token,
#            so some drafts end early): the reference of mon
#   mon      BIN, the same, SKIP (a draft that a p_min cut ends before its last pass copies every row)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Two slots (-np 2, --metrics, the step trace), C741's run/two_slots.py: A = G2's request, B = G1's posted once A has 16
# tokens, /slots and /metrics read in turn every 2 ms while A or B runs (a step's passes then draft both slots):
#   p2base   BASE
#   p2off    BIN, SKIP unset
#   p2on     BIN, SKIP
# Every arm writes LLAMA_SPEC_DUMP: one line per slot and verify step with the draft, the drafter's candidates and the
# target's top tokens; steps.jsonl is the dump without its run, step and task fields.
# Pass: every arm of BIN is its reference's replies (tokens, top probabilities, draft counts) and steps.jsonl, in full;
# in every traced arm of BIN each draft (the marks from mtp.drf to mtp.drf.end) copies one hidden row batch to the host
# per drafting pass (the step trace's counts h.host, h.skip), except, with SKIP, a draft whose passes reach n_max (3): its
# last pass copies none (h.skip 1); off has no h.skip; with SKIP, the drafter's total log line holds the trace's sums; in
# every two-slot arm B came while A ran, both were answered with no failed read, and A and B are base's G2 and G1; no arm
# logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> TWO_SLOTS=<C741's two_slots.py> [PORT=18742] [THREADS=8] [ARMS='base off ...'] [SARGS='<more arguments>']
#   bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,step_trace.jsonl}, the
#   two-slot arms' OUT/<arm>/{A,B}.json, reads.tsv, client.json; OUT/summary.tsv, OUT/copies.md, OUT/copies.tsv,
#   OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
TWO_SLOTS=${TWO_SLOTS:?TWO_SLOTS=<two_slots.py of C741>}
PORT=${PORT:-18742}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on onnt bbase boff bon pbase pon mbase mon p2base p2off p2on}
SARGS=${SARGS:-}
RUN=$(dirname "$(realpath "$0")")
N_MAX=3
MTP="--spec-type draft-mtp --spec-draft-n-max $N_MAX --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
SKIP=LLAMA_MTP_SKIP_LAST_ROW=1
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [on]=$BIN [onnt]=$BIN
    [bbase]=$BASE [boff]=$BIN [bon]=$BIN
    [pbase]=$BASE [pon]=$BIN
    [mbase]=$BASE [mon]=$BIN
    [p2base]=$BASE [p2off]=$BIN [p2on]=$BIN
)
# a later --spec-draft-p-min overrides MTP's
declare -A FLAGS=(
    [bbase]="-bs" [boff]="-bs" [bon]="-bs"
    [pbase]="--spec-draft-sampling probabilistic" [pon]="--spec-draft-sampling probabilistic"
    [mbase]="--spec-draft-p-min 0.1022" [mon]="--spec-draft-p-min 0.1022"
)
declare -A ENVS=(
    [on]=$SKIP [onnt]=$SKIP [bon]=$SKIP [pon]=$SKIP [mon]=$SKIP [p2on]=$SKIP
)
# the reference of each arm of the branch
declare -A REF=(
    [off]=base [on]=base [onnt]=base [boff]=bbase [bon]=bbase [pon]=pbase [mon]=mbase
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
    [ "$arm" = onnt ] && tr=
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
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(cat "$d/step_trace.jsonl" 2>/dev/null | wc -l) trace lines; $(grep -c 'last drafting pass copies none' "$d/server.log") switch lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the table
loads=
for a in $ARMS; do
    [ -s "$OUT/$a/steps.jsonl" ] && loads="$loads $a=$OUT/$a"
done
# shellcheck disable=SC2086
python3 -I "$RUN/copies.py" --n-max "$N_MAX" --tsv "$OUT/copies.tsv" $loads > "$OUT/copies.md"

# the verdict
fail=0
fields='[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]'
same_req() { # <arm a> <req a> <arm b> <req b>
    [ -s "$OUT/$1/$2.json" ] && [ -s "$OUT/$3/$4.json" ] && \
        cmp -s <(jq -c "$fields" "$OUT/$1/$2.json") <(jq -c "$fields" "$OUT/$3/$4.json")
}
same() { # <reference> <arm>: tokens, top probabilities and draft counts of every request, and the verify steps
    local r
    for r in $REQS; do
        if ! same_req "$1" "$r" "$2" "$r"; then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    if [ ! -s "$OUT/$1/steps.jsonl" ] || ! cmp -s "$OUT/$1/steps.jsonl" "$OUT/$2/steps.jsonl"; then
        echo "$2's verify steps differ from $1's (or $1 wrote none)"; return 1
    fi
    echo "$2 = $1 (tokens, top probabilities and draft counts of $REQS; $(wc -l < "$OUT/$2/steps.jsonl") verify steps the same)"
}
has() { [[ " $ARMS " == *" $1 "* ]]; }
col() { awk -F'\t' -v a="$1" -v c="$2" 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next } $1 == a { print $h[c]; exit }' "$OUT/copies.tsv"; }
{
    for a in off on onnt boff bon pon mon; do
        r=${REF[$a]}
        has "$a" && has "$r" || continue
        same "$r" "$a" || fail=1
    done
    # the copies: per draft one per pass, less the last pass of a draft that reaches n_max with the switch
    for a in off on boff bon pon mon p2off p2on; do
        has "$a" || continue
        sw=0
        [ -n "${ENVS[$a]:-}" ] && sw=1
        dr=$(col "$a" drafts) ps=$(col "$a" passes) h=$(col "$a" h_host) s=$(col "$a" h_skip) full=$(col "$a" full) bad=$(col "$a" bad)
        cn=$(col "$a" counted)
        if [ "${cn:-0}" = 0 ] || [ "${dr:-0}" = 0 ] || [ "$bad" != 0 ] || [ $(( ${h:-0} + ${s:-0} )) != "$ps" ] || \
                { [ $sw = 1 ] && [ "$s" != "$full" ]; } || { [ $sw = 0 ] && [ "$s" != 0 ]; }; then
            echo "$a: drafts $dr, passes $ps, copied $h, left out $s, drafts at n_max $full, drafts off the rule $bad: not one copy a pass less the marked last ones"; fail=1
        else
            echo "$a: $dr drafts, $ps passes ($full reach n_max), $h hidden rows copied to the host and $s left out: $(col "$a" per_draft) copies a draft"
        fi
        if [ $sw = 1 ]; then
            lg=$(grep -o 'draft hidden rows to the host: [0-9]* copied, [0-9]* left out (the last pass) in [0-9]* drafts' "$OUT/$a/server.log" | tail -n 1)
            lh=$(echo "$lg" | sed -E 's/.*: ([0-9]+) copied, ([0-9]+) left out.*/\1/') ls=$(echo "$lg" | sed -E 's/.* ([0-9]+) left out.*/\1/')
            if [ "$lh" = "$h" ] && [ "$ls" = "$s" ]; then
                echo "$a: log: $lg"
            else
                echo "$a: log '${lg:-no total line}' is not the trace's $h copied, $s left out"; fail=1
            fi
        fi
    done
    if has onnt; then
        lg=$(grep -o 'draft hidden rows to the host: [0-9]* copied, [0-9]* left out (the last pass) in [0-9]* drafts' "$OUT/onnt/server.log" | tail -n 1)
        if [ -n "$lg" ] && has on && [ "$lg" = "$(grep -o 'draft hidden rows to the host: [0-9]* copied, [0-9]* left out (the last pass) in [0-9]* drafts' "$OUT/on/server.log" | tail -n 1)" ]; then
            echo "onnt: log = on's: $lg"
        else
            echo "onnt: log '${lg:-no total line}' is not on's"; fail=1
        fi
    fi
    for a in p2base p2off p2on; do
        has "$a" || continue
        d=$OUT/$a
        if ! jq -e '.b_during_a and .a_status == 200 and .b_status == 200 and .reads_failed == 0' "$d/client.json" > /dev/null 2>&1; then
            echo "$a: B did not come during A, a reply failed or a read failed ($(cat "$d/client.json" 2>/dev/null | tr -d '\n '))"; fail=1; continue
        fi
        ok=1
        has base && { same_req base G2 "$a" A && same_req base G1 "$a" B || ok=0; }
        if [ $ok = 1 ]; then
            echo "$a: A = base's G2 and B = base's G1 (tokens, top probabilities, draft counts); $(jq -r '.reads' "$d/client.json") reads answered; drafts of both slots $(col "$a" two)"
        else
            echo "$a: A or B differs from base's one-slot reply"; fail=1
        fi
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
