#!/usr/bin/env bash
# C741 (Q1310): llama-server's yields to the task queue cut (LLAMA_SERVER_YIELD=1: a generation step's draft and
# catch-up without a yield; =2: also a yield wakes the queue's worker thread only for a task), checked on the CPU with a
# random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp). Made from C738's server_check.sh
# (docs/data/2026-11-11_c738_verify_prepare). BASE is the base branch's llama-server, BIN the branch's.
# One slot (-np 1; -c 8192 -b 512 -ub 128 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0, as the release preset drafts; Y = LLAMA_SERVER_YIELD):
#   base     BASE: the reference
#   off      BIN, Y unset
#   y1       BIN, Y=1
#   y2       BIN, Y=2
#   y2nt     BIN, Y=2, without the step trace (every other arm writes LLAMA_STEP_TRACE)
#   bbase    BASE, -bs (the backend sampler, as the preset): the reference of the b arms
#   boff     BIN, -bs
#   by1      BIN, -bs, Y=1
#   by2      BIN, -bs, Y=2
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Two slots (-np 2, --metrics, the step trace), run/two_slots.py: A = G2's request, B = G1's posted once A has 16
# tokens, /slots and /metrics read in turn every 2 ms while A or B runs:
#   p2base   BASE
#   p2off    BIN, Y unset
#   p2y1     BIN, Y=1
#   p2y2     BIN, Y=2
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens.
# Pass: off, y1, y2, y2nt give base's tokens, top probabilities, draft counts (draft_n, draft_n_accepted) and verify
# steps (the dump without its run, step and task fields) on every request; boff, by1, by2 the same against bbase; in
# every two-slot arm B came while A ran and both were answered with no failed read, and A and B are base's G2 and G1
# (tokens, top probabilities, draft counts); the generation steps' yields and wakes (yields.py) are 3 and 3 in off, 1
# and 1 in y1, 1 and 0 in y2; no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18741] [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,step_trace.jsonl},
#   the two-slot arms' OUT/<arm>/{A,B}.json, reads.tsv, client.json; OUT/summary.tsv, OUT/yields.md,
#   OUT/two_slots.md, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18741}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off y1 y2 y2nt bbase boff by1 by2 p2base p2off p2y1 p2y2}
SARGS=${SARGS:-}
RUN=$(dirname "$(realpath "$0")")
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [y1]=$BIN [y2]=$BIN [y2nt]=$BIN
    [bbase]=$BASE [boff]=$BIN [by1]=$BIN [by2]=$BIN
    [p2base]=$BASE [p2off]=$BIN [p2y1]=$BIN [p2y2]=$BIN
)
declare -A FLAGS=(
    [bbase]="-bs" [boff]="-bs" [by1]="-bs" [by2]="-bs"
)
declare -A ENVS=(
    [y1]=LLAMA_SERVER_YIELD=1 [y2]=LLAMA_SERVER_YIELD=2 [y2nt]=LLAMA_SERVER_YIELD=2
    [by1]=LLAMA_SERVER_YIELD=1 [by2]=LLAMA_SERVER_YIELD=2
    [p2y1]=LLAMA_SERVER_YIELD=1 [p2y2]=LLAMA_SERVER_YIELD=2
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
    [ "$arm" = y2nt ] && tr=
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
        python3 -I "$RUN/two_slots.py" --port "$PORT" --a "$d/A.req.json" --b "$d/B.req.json" --out "$d" --after 16 \
            --poll-ms 2 > "$d/client.out" 2>&1
        rm -f "$d/A.req.json" "$d/B.req.json"
        reqs="A B"
    fi

    # SIGTERM and wait: the dump and the trace are whole
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $reqs; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(cat "$d/step_trace.jsonl" 2>/dev/null | wc -l) trace lines; $(grep -c 'LLAMA_SERVER_YIELD=' "$d/server.log") switch lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the tables
loads=
for a in off y1 y2 boff by1 by2 base bbase; do
    [ -s "$OUT/$a/step_trace.jsonl" ] && loads="$loads $a=$OUT/$a/step_trace.jsonl"
done
# shellcheck disable=SC2086
python3 -I "$RUN/yields.py" --tsv "$OUT/yields.tsv" $loads > "$OUT/yields.md"
loads=
for a in p2base p2off p2y1 p2y2; do
    [ -s "$OUT/$a/client.json" ] && loads="$loads $a=$OUT/$a"
done
# shellcheck disable=SC2086
[ -n "$loads" ] && python3 -I "$RUN/yields.py" --two-slots --tsv "$OUT/two_slots.tsv" $loads > "$OUT/two_slots.md"

# the verdict
fail=0
fields='[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]'
same_req() { # <arm a> <req a> <arm b> <req b>
    [ -s "$OUT/$1/$2.json" ] && [ -s "$OUT/$3/$4.json" ] && \
        cmp -s <(jq -c "$fields" "$OUT/$1/$2.json") <(jq -c "$fields" "$OUT/$3/$4.json")
}
same() { # <arm a> <arm b>: tokens, top probabilities and draft counts of every request, and the verify steps
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
ytsv() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/yields.tsv"; }
{
    for a in off y1 y2 y2nt; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    for a in boff by1 by2; do
        has bbase && has "$a" && { same bbase "$a" || fail=1; }
    done
    # the yields of the generation steps: every step's count, from yields.md's distribution column
    for spec in off:3:3 y1:1:1 y2:1:0 boff:3:3 by1:1:1 by2:1:0; do
        a=${spec%%:*} rest=${spec#*:}
        ey=${rest%%:*} ew=${rest#*:}
        has "$a" || continue
        line=$(grep -E "^\| $a \|" "$OUT/yields.md")
        ycol=$(awk -F'|' '{ gsub(/^ +| +$/, "", $4); print $4 }' <<< "$line")
        wcol=$(awk -F'|' '{ gsub(/^ +| +$/, "", $5); print $5 }' <<< "$line")
        n=$(ytsv "$a" 2)
        if [[ "$ycol" == "$ey: $n" ]] && [[ "$wcol" == "$ew: $n" ]]; then
            echo "$a: each of $n generation steps $ey yields and $ew wakes of the worker"
        else
            echo "$a: generation steps' yields ($ycol) or wakes ($wcol) not $ey and $ew in each of $n"; fail=1
        fi
    done
    for a in p2base p2off p2y1 p2y2; do
        has "$a" || continue
        d=$OUT/$a
        if ! jq -e '.b_during_a and .a_status == 200 and .b_status == 200 and .reads_failed == 0' "$d/client.json" > /dev/null 2>&1; then
            echo "$a: B did not come during A, a reply failed or a read failed ($(cat "$d/client.json" 2>/dev/null | tr -d '\n '))"; fail=1; continue
        fi
        ok=1
        has base && { same_req base G2 "$a" A && same_req base G1 "$a" B || ok=0; }
        if [ $ok = 1 ]; then
            echo "$a: A = base's G2 and B = base's G1 (tokens, top probabilities, draft counts); B came after A's 16th token; $(jq -r '.reads' "$d/client.json") reads answered"
        else
            echo "$a: A or B differs from base's one-slot reply"; fail=1
        fi
    done
    # within one step: B's prompt in the step after the one B came in (the server's marks); the reads as the server saw
    # them (answered in a yield of the step they came in, or left to its end, where the loop takes every task)
    stsv() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/two_slots.tsv"; }
    for a in p2off p2y1 p2y2; do
        has "$a" && has p2off && [ -s "$OUT/two_slots.tsv" ] || continue
        sb=$(stsv "$a" 5) sb0=$(stsv p2off 5)
        lim_s=$(( ${sb0:-1} > 1 ? ${sb0:-1} : 1 ))
        if [ -n "$sb" ] && [ "$sb" -le "$lim_s" ]; then
            echo "$a: B's prompt $sb step(s) after the step it came in (p2off: $sb0); of $(stsv "$a" 8) reads posted in a step $(stsv "$a" 9) answered in a yield of that step, $(stsv "$a" 10) at its end"
        else
            echo "$a: B's prompt ${sb:-never} step(s) after the step it came in (p2off: $sb0)"; fail=1
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
