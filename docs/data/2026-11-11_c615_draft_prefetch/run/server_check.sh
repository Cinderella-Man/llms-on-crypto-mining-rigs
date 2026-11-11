#!/usr/bin/env bash
# C615 (Q1100): llama-server's draft-mtp with LLAMA_DRAFT_PREFETCH (each drafted token's n-gram rows read ahead as it is
# sampled, and id_last's before the first drafting pass), checked on the CPU with a random-weight qwen4exp model with one
# MTP block (test-llama-archs -o --mtp) whose n-gram table is mapped (-lzm on: its rows are read on demand, as Flash-Next's
# 27 GB table on the rig). Made from C738's server_check.sh (docs/data/2026-11-11_c738_verify_prepare). BASE is the base
# branch's llama-server, BIN the branch's, MBIN (optional) a test merge's.
# Arms (all -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -lm mmap -lzm on, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0; DP = LLAMA_DRAFT_PREFETCH, S = LLAMA_DRAFT_PREFETCH_STATS=1):
#   base   BASE: the reference
#   off    BIN, DP unset
#   on     BIN, DP=1
#   ons    BIN, DP=1, S (a draft-prefetch: line per evaluation)
#   cap0   BIN, DP=1, LLAMA_DRAFT_PREFETCH_KB=0, S: the control, every token over the cap, nothing issued
#   offs   BIN, DP unset, S: the counts alone
#   bbase  BASE, -bs (the backend sampler, as the preset): the reference of bons
#   bons   BIN, -bs, DP=1, S
#   moff   MBIN, nothing set
#   mons   MBIN, DP=1, S
#   mboth  MBIN, DP=1, S and LLAMA_NGRAM_PREFETCH=1 (C761's helper for prompts' later micro-batches)
#   troff  BIN, LLAMA_UBATCH_TRACE=1 (C701's trace; it turns the counts on too): the rig row's trace load, switch off
#   trc    BIN, LLAMA_UBATCH_TRACE=1, DP=1: the rig row's trace load, switch on (c615_read.py reads both logs)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42, the preset's sampling); S2 = G2's prompt, the same with seed 7.
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens.
# Pass: every arm but the references gives its reference's tokens, top probabilities, draft counts (draft_n,
# draft_n_accepted) and verify steps (the dump without its run, step and task fields) on every request; in every arm with
# DP=1 and S each evaluation after a draft finds every one of its rows issued ahead (ahead = rows) and the context's C615
# line is logged; cap0 issues nothing and finds nothing ahead; offs makes no call; no arm logs an error.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [MBIN=<dir with a test merge's llama-server>] [PORT=18615] [THREADS=8] [ARMS='base off ...']
#   [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,steps.jsonl,prefetch.tsv}, OUT/summary.tsv,
#   OUT/prefetch.tsv (per arm), OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
MBIN=${MBIN:-}
PORT=${PORT:-18615}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on ons cap0 offs bbase bons$([ -n "$MBIN" ] && echo ' moff mons mboth')}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
REQS="G1 G2 S1 S2"
S=LLAMA_DRAFT_PREFETCH_STATS=1
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [on]=$BIN [ons]=$BIN [cap0]=$BIN [offs]=$BIN
    [bbase]=$BASE [bons]=$BIN
    [moff]=$MBIN [mons]=$MBIN [mboth]=$MBIN
    [troff]=$BIN [trc]=$BIN
)
declare -A FLAGS=(
    [bbase]="-bs" [bons]="-bs"
)
declare -A ENVS=(
    [on]="LLAMA_DRAFT_PREFETCH=1" [ons]="LLAMA_DRAFT_PREFETCH=1 $S"
    [cap0]="LLAMA_DRAFT_PREFETCH=1 LLAMA_DRAFT_PREFETCH_KB=0 $S" [offs]="$S"
    [bons]="LLAMA_DRAFT_PREFETCH=1 $S"
    [mons]="LLAMA_DRAFT_PREFETCH=1 $S" [mboth]="LLAMA_DRAFT_PREFETCH=1 $S LLAMA_NGRAM_PREFETCH=1"
    [troff]="LLAMA_UBATCH_TRACE=1" [trc]="LLAMA_UBATCH_TRACE=1 LLAMA_DRAFT_PREFETCH=1"
)
declare -A REF=(
    [off]=base [on]=base [ons]=base [cap0]=base [offs]=base [bons]=bbase [moff]=base [mons]=base [mboth]=base
    [troff]=base [trc]=base
)
# the arms whose every evaluation after a draft must find all its rows issued ahead
PF_ARMS="ons bons mons mboth trc"

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

# the draft-prefetch: lines of a log as a table, one row per evaluation (the trailing "us" is the counting's own time)
pf_table() { # <server.log>
    printf 'eval\tn_tokens\trows\tahead\tpages\tresident\tcalls\ttokens\tissued\tdup\tcapped\tbytes\tcall_us\tus\n'
    grep -a -o 'draft-prefetch: .*' "$1" | awk '{
        n = 0; for (i = 1; i <= NF; i++) if ($(i + 1) == "=") v[++n] = $(i + 2)
        printf "%d", NR; for (i = 1; i <= n; i++) printf "\t%s", v[i]; printf "\n" }'
}

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\ttokens\n' > "$OUT/summary.tsv"
printf 'arm\tevals\tpf_evals\tpf_rows\tpf_ahead\tpf_full\tcalls\ttokens\tissued\tdup\tcapped\tbytes_max\tcall_us\tpages_checked\tresident\tc615\n' > "$OUT/prefetch.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    if [ -z "${EXE[$arm]}" ]; then
        echo "$arm: no binary"; continue
    fi
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 -lm mmap -lzm on --no-webui --host 127.0.0.1 \
        --port "$PORT" $MTP $SARGS ${FLAGS[$arm]:-} > "$d/server.log" 2>&1 &
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

    # the prefetch counts: evaluations after a draft (calls > 0), their rows, the rows issued ahead, those with every
    # row ahead, the calls, tokens, rows issued, rows already issued, tokens over the cap, the most bytes a window, the
    # calls' time, pages checked by mincore and resident
    pf_table "$d/server.log" > "$d/prefetch.tsv"
    c615=$(grep -a -c 'C615: draft prefetch' "$d/server.log")
    awk -F'\t' -v a="$arm" -v c="$c615" 'NR > 1 {
            ev++
            if ($7 > 0) { pe++; pr += $3; pa += $4; if ($4 == $3) pf++ }
            ca += $7; tk += $8; is += $9; du += $10; cp += $11; if ($12 > bm) bm = $12; cu += $13
            if ($6 >= 0) { pc += $5; rs += $6 } }
        END { printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", a, ev, pe, pr, pa, pf, ca, tk, is, du, cp, bm, cu, pc, rs, c }' \
        "$d/prefetch.tsv" >> "$OUT/prefetch.tsv"
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(($(wc -l < "$d/prefetch.tsv") - 1)) draft-prefetch lines, $c615 C615 lines; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

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
col() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/prefetch.tsv"; }
{
    for a in $ARMS; do
        [ -n "${REF[$a]:-}" ] || continue
        has "${REF[$a]}" || continue
        same "${REF[$a]}" "$a" || fail=1
    done
    for a in $PF_ARMS; do
        has "$a" || continue
        pe=$(col "$a" 3) pr=$(col "$a" 4) pa=$(col "$a" 5) pf=$(col "$a" 6) ca=$(col "$a" 7) c=$(col "$a" 16)
        if [ "${pe:-0}" -gt 0 ] && [ "$pf" = "$pe" ] && [ "$pa" = "$pr" ] && [ "${c:-0}" -gt 0 ]; then
            echo "$a: $pe evaluations after a draft, each with every row issued ahead ($pa of $pr rows), $ca prefetch calls; C615 line logged"
        else
            echo "$a: $pf of ${pe:-0} evaluations after a draft had every row issued ahead ($pa of $pr rows), ${c:-0} C615 lines"; fail=1
        fi
    done
    if has cap0; then
        ca=$(col cap0 7) is=$(col cap0 9) pa=$(col cap0 5) cp=$(col cap0 11) tk=$(col cap0 8)
        if [ "${ca:-0}" -gt 0 ] && [ "$is" = 0 ] && [ "$pa" = 0 ] && [ "$tk" = 0 ] && [ "${cp:-0}" -gt 0 ]; then
            echo "cap0: $ca calls, every one of $cp tokens over the cap, nothing issued, nothing found ahead"
        else
            echo "cap0: $ca calls, $is rows issued, $pa found ahead, $tk tokens issued, $cp over the cap: the cap did not hold"; fail=1
        fi
    fi
    if has offs; then
        ca=$(col offs 7) ev=$(col offs 2) pa=$(col offs 5)
        if [ "$ca" = 0 ] && [ "$pa" = 0 ] && [ "${ev:-0}" -gt 0 ]; then
            echo "offs: $ev evaluations counted, no prefetch call"
        else
            echo "offs: $ca prefetch calls with the switch off, $pa rows ahead, $ev evaluations"; fail=1
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
