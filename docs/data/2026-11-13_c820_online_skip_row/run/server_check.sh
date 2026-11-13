#!/usr/bin/env bash
# C820 (Q1434): LLAMA_MTP_SKIP_LAST_ROW=2 (a step's last MTP drafting pass copies no hidden row to the host at any draft
# length) against the skip off and against C742's =1, checked in llama-server on the CPU with a random-weight qwen4exp
# model with one MTP block (test-llama-archs -o --mtp). Made from C815's server_check.sh
# (docs/data/2026-11-12_c815_adapt_while_drafting): the same requests, cost tables and groups. BASE is the base branch's
# llama-server (pascal-p106-v4 before C820), BIN the branch's.
# Every arm: -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 6 --spec-draft-p-min 0, LLAMA_SPEC_ADAPT=1, its log (LLAMA_SPEC_ADAPT_LOG) and LLAMA_SPEC_DUMP.
# Groups (C815's tables; see its server_check.sh for how they were set), each with three arms: <g>off (the skip unset,
# the group's reference), <g>s1 (LLAMA_MTP_SKIP_LAST_ROW=1), <g>s2 (=2):
#   d  TD, S fixed at 1, LLAMA_SPEC_ADAPT_ONLINE=2; also dbase: BASE with the same settings
#   w  TW, every width, S fixed at 1, ONLINE=2 (the bound stops drafts between the allowed lengths)
#   r  TR, every width, S run, ONLINE=2
#   b  TD, ONLINE=2, -bs (the backend sampler)
#   p  TD, ONLINE=2, --spec-draft-sampling probabilistic
#   v  C797's arm settings (FN82's est. v4 table, LLAMA_SPEC_ADAPT_DRAFT_MS=21.87, S run), ONLINE=2
#   n  TD without ONLINE: every draft runs to n_max (C742's case: =2 leaves out what =1 leaves out)
#   c  TD without ONLINE, LLAMA_MTP_DRAFT_CAP=1 (C744: a greedy draft stops at its slot's cap at a reply's end)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42); S2 = G2's prompt, the same with seed 7.
# Pass:
#   - every arm against its group's reference (dbase against doff): every request's tokens, top probabilities and draft
#     counts; the verify steps (LLAMA_SPEC_DUMP without run, step and task) and the rule's steps (LLAMA_SPEC_ADAPT_LOG
#     without run, step and task, with n_draft and p: the drafts themselves) the same files;
#   - the copies, from the drafter's closing lines (C742's "draft hidden rows to the host: X copied, Y left out" in s1
#     and s2, C820's line in s2 only): in each group s1's and s2's drafts and copied + left out the same (the same
#     passes); s2's left out minus s1's equal to s2's "below n_max", and its look-ahead count no larger; in n, nothing
#     below n_max and s1's left out above 0; in the ONLINE groups and c, s2's "below n_max" above 0; in n and c, no
#     look-ahead; over the ONLINE groups, look-aheads above 0; no C820 line in off and s1 arms, none of C742's in off;
#   - no error lines.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18820] [THREADS=8] [ARMS='dbase doff ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,adapt.jsonl,steps.jsonl,rule.jsonl},
#   OUT/summary.tsv, OUT/copies.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18820}
THREADS=${THREADS:-8}
ARMS=${ARMS:-dbase doff ds1 ds2 woff ws1 ws2 roff rs1 rs2 boff bs1 bs2 poff ps1 ps2 voff vs1 vs2 noff ns1 ns2 coff cs1 cs2}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0"
ADAPT="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_COST=87.597,94.940,105.432,117.985,139.227,152.703,166.181 LLAMA_SPEC_ADAPT_DRAFT_MS=21.87"
TD=100,105,111.3556,111.4,111.476328,111.477,111.477518
TW=100,110.3028,111.36485,111.474371,111.4856174,111.4970678,112.4970678
TR=100,111.38431,112.557846,112.6788637,112.6912907,112.7039431,113.7039431
REQS="G1 G2 S1 S2"
ONLINE_GROUPS="d w r b p v"
mkdir -p "$OUT"

ref() { echo "${1:0:1}off"; } # the group's reference
kind_env() {
    case "${1:1}" in
        s1) echo "LLAMA_MTP_SKIP_LAST_ROW=1" ;;
        s2) echo "LLAMA_MTP_SKIP_LAST_ROW=2" ;;
    esac
}
group_env() {
    case "$1" in
        d*|b*|p*) echo "LLAMA_SPEC_ADAPT_COST=$TD LLAMA_SPEC_ADAPT_S=1 LLAMA_SPEC_ADAPT_ONLINE=2" ;;
        w*) echo "LLAMA_SPEC_ADAPT_COST=$TW LLAMA_SPEC_ADAPT_S=1 LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7 LLAMA_SPEC_ADAPT_ONLINE=2" ;;
        r*) echo "LLAMA_SPEC_ADAPT_COST=$TR LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7 LLAMA_SPEC_ADAPT_S0=0.905 LLAMA_SPEC_ADAPT_EMA=1e-6 LLAMA_SPEC_ADAPT_DRAFT_MS=0 LLAMA_SPEC_ADAPT_ONLINE=2" ;;
        v*) echo "LLAMA_SPEC_ADAPT_ONLINE=2" ;;
        n*) echo "LLAMA_SPEC_ADAPT_COST=$TD LLAMA_SPEC_ADAPT_S=1" ;;
        c*) echo "LLAMA_SPEC_ADAPT_COST=$TD LLAMA_SPEC_ADAPT_S=1 LLAMA_MTP_DRAFT_CAP=1" ;;
    esac
}
group_flags() { case "$1" in b*) echo "-bs" ;; p*) echo "--spec-draft-sampling probabilistic" ;; esac; }
exe()         { case "$1" in ?base) echo "$BASE" ;; *) echo "$BIN" ;; esac; }

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
printf 'arm\tsteps\tdrafted\tkept\taccepted\tc742_copied\tc742_left_out\tc742_drafts\tc820_steps\tc820_copied\tc820_left_out\tc820_below_n_max\tc820_lookahead\tc815_stopped\tc815_left_out\n' > "$OUT/copies.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    # shellcheck disable=SC2046,SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" LLAMA_SPEC_ADAPT_LOG="$d/adapt.jsonl" $ADAPT $(group_env "$arm") $(kind_env "$arm") \
        "$(exe "$arm")/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" \
        $MTP $SARGS $(group_flags "$arm") > "$d/server.log" 2>&1 &
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

    # SIGTERM and wait: the closing lines and the logs are whole
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    for r in $REQS; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0), (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the verify steps and the rule's steps without the run's start time, the step counter and the task id
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    jq -c 'select(.hdr == null) | del(.run, .step, .task)' "$d/adapt.jsonl" > "$d/rule.jsonl" 2>/dev/null
    # the drafter's closing lines: C742's (=1 and =2), C820's (=2), C815's
    c7=$(grep -a -o -E 'draft hidden rows to the host: [0-9]+ copied, [0-9]+ left out \(the last pass\) in [0-9]+ drafts' "$d/server.log" | tail -n 1 | grep -o -E '[0-9]+' | tr '\n' '\t' | sed 's/\t$//')
    c8=$(grep -a -o -E 'C820: [0-9]+ drafting steps, [0-9]+ hidden rows copied to the host, [0-9]+ left out, [0-9]+ of them below n_max, [0-9]+ of those by the rule' "$d/server.log" | tail -n 1 | grep -o -E '[0-9]+' | sed -n '2,6p' | tr '\n' '\t' | sed 's/\t$//')
    c5=$(grep -a -o -E 'C815: [0-9]+ drafts under the rule taken while drafting, [0-9]+ stopped early, [0-9]+ drafting passes left out' "$d/server.log" | tail -n 1 | grep -o -E '[0-9]+' | sed -n '3,4p' | tr '\n' '\t' | sed 's/\t$//')
    st=$(jq -r -s 'map(select(.hdr == null)) | [length, (map(.n_draft) | add // 0), (map(.n_keep) | add // 0),
         (map(.n_acc) | add // 0)] | @tsv' "$d/adapt.jsonl" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "${st:-0	0	0	0}" "${c7:--	-	-}" "${c8:--	-	-	-	-}" "${c5:--	-}" >> "$OUT/copies.tsv"
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(wc -l < "$d/rule.jsonl") rule steps; C742 line: ${c7:-none}; C820 line: ${c8:-none}; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the verdict
fail=0
has() { [[ " $ARMS " == *" $1 "* ]]; }
col() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/copies.tsv"; }
same_reqs() { # <arm a> <arm b>: tokens, top probabilities and draft counts
    local r
    for r in $REQS; do
        if [ ! -s "$OUT/$1/$r.json" ] || [ ! -s "$OUT/$2/$r.json" ] || ! cmp -s \
                <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$1/$r.json") \
                <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$2/$r.json"); then
            echo "$r"; return 1
        fi
    done
}
same() { # <reference> <arm>
    local a=$1 b=$2 r
    if ! r=$(same_reqs "$a" "$b"); then echo "$b differs from $a at $r"; return 1; fi
    if [ ! -f "$OUT/$a/steps.jsonl" ] || { [ ! -s "$OUT/$a/steps.jsonl" ] && [ "${a:0:1}" != v ]; } ||
            ! cmp -s "$OUT/$a/steps.jsonl" "$OUT/$b/steps.jsonl"; then
        echo "$b's verify steps differ from $a's (or $a wrote none)"; return 1
    fi
    if [ ! -s "$OUT/$a/rule.jsonl" ] || ! cmp -s "$OUT/$a/rule.jsonl" "$OUT/$b/rule.jsonl"; then
        echo "$b's rule steps (with the drafts) differ from $a's (or $a wrote none)"; return 1
    fi
    echo "$b = $a (tokens, top probabilities and draft counts of $REQS; $(wc -l < "$OUT/$b/steps.jsonl") verify steps and $(wc -l < "$OUT/$b/rule.jsonl") rule steps the same; drafted $(col "$b" 3) tokens)"
}
{
    for b in $ARMS; do
        a=$(ref "$b")
        [ "$b" = "$a" ] && continue
        has "$a" || continue
        same "$a" "$b" || fail=1
    done
    rule_total=0
    for g in $ONLINE_GROUPS n c; do
        has "${g}s1" && has "${g}s2" || continue
        x1=$(col "${g}s1" 6) y1=$(col "${g}s1" 7) z1=$(col "${g}s1" 8)
        x2=$(col "${g}s2" 6) y2=$(col "${g}s2" 7) z2=$(col "${g}s2" 8)
        s8=$(col "${g}s2" 9) x8=$(col "${g}s2" 10) y8=$(col "${g}s2" 11) lo=$(col "${g}s2" 12) la=$(col "${g}s2" 13)
        msg="${g}: =1 copied $x1, left out $y1 in $z1 drafts; =2 copied $x2, left out $y2 in $z2 drafts, $lo of them below n_max, $la by the look-ahead"
        ok=1
        for v in "$x1" "$y1" "$z1" "$x2" "$y2" "$z2" "$s8" "$x8" "$y8" "$lo" "$la"; do
            [[ "$v" =~ ^[0-9]+$ ]] || ok=0
        done
        if [ $ok = 0 ]; then
            echo "$msg: a closing line is missing"; fail=1; continue
        fi
        [ "$z1" = "$z2" ] && [ $((x1 + y1)) = $((x2 + y2)) ] || { echo "$msg: not the same passes"; ok=0; }
        [ "$s8" = "$z2" ] && [ "$x8" = "$x2" ] && [ "$y8" = "$y2" ] || { echo "$msg: C820's line ($s8, $x8, $y8) is not C742's"; ok=0; }
        [ $((y2 - y1)) = "$lo" ] && [ "$la" -le "$lo" ] || { echo "$msg: =2's extra rows left out are not those below n_max"; ok=0; }
        case "$g" in
            n) [ "$lo" = 0 ] && [ "$y1" -gt 0 ] && [ "$la" = 0 ] || { echo "$msg: at n_max =2 should be =1"; ok=0; } ;;
            c) [ "$lo" -gt 0 ] && [ "$la" = 0 ] || { echo "$msg: the cap's last passes were not left out"; ok=0; } ;;
            *) [ "$lo" -gt 0 ] || { echo "$msg: nothing below n_max left out"; ok=0; }
               rule_total=$((rule_total + la)) ;;
        esac
        [ $ok = 1 ] && echo "$msg" || fail=1
    done
    # (in d, b and p this model's drafts never let the look-ahead settle a stop: required only when w, r or v ran)
    if [ "$rule_total" -gt 0 ]; then
        echo "the look-ahead left out $rule_total rows over the ONLINE groups"
    elif has ws2 || has rs2 || has vs2; then
        echo "the look-ahead left out no row in the ONLINE groups"; fail=1
    else
        echo "the look-ahead left out no row (no w, r or v group in ARMS)"
    fi
    for a in $ARMS; do
        [ -f "$OUT/$a/server.log" ] || continue
        case "${a:1}" in
            off|base) ! grep -aq -E 'C820: |draft hidden rows to the host' "$OUT/$a/server.log" || { echo "$a: a skip line with the skip off"; fail=1; } ;;
            s1) ! grep -aq 'C820: ' "$OUT/$a/server.log" || { echo "$a: C820's line at =1"; fail=1; } ;;
        esac
        e=$(grep -ciE 'error|abort' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
