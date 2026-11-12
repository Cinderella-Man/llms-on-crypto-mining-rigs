#!/usr/bin/env bash
# C815 (Q1424): LLAMA_SPEC_ADAPT's rule taken while drafting (LLAMA_SPEC_ADAPT_ONLINE=1|2) against the switch as built
# (LLAMA_SPEC_ADAPT=1 alone), checked in llama-server on the CPU with a random-weight qwen4exp model with one MTP block
# (test-llama-archs -o --mtp). Made from C738's server_check.sh (docs/data/2026-11-11_c738_verify_prepare). BASE is the
# base branch's llama-server (pascal-p106-v4 before C815), BIN the branch's.
# Every arm: -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 6 --spec-draft-p-min 0, LLAMA_SPEC_ADAPT=1, its log (LLAMA_SPEC_ADAPT_LOG) and LLAMA_SPEC_DUMP.
# ON = LLAMA_SPEC_ADAPT_ONLINE, VP = LLAMA_VERIFY_PREPARE.
# The random model's draft gives each drafted token a probability near 0.10 (its top 10 nearly even), so under a real
# cost table the rule keeps nothing on every step. The cost tables below (LLAMA_SPEC_ADAPT_COST, T(1) = 100 ms) put
# each test's threshold at the median of what this model's drafts bring to it (from a first run's logs), so that the
# rule's choice varies from step to step:
#   TD (default widths: lengths 0, 2, 4, 6; S fixed at 1): the tests at 2 and 4 at the medians of P1 + P2 and P3 + P4,
#      the test at 6 at 1.19e-5 (P5's 70th percentile): the rule keeps 0, 2, 4 and 6, and at 6 often whatever the last
#      token's probability, so that C738's prepare runs
#   TW (every width; S fixed at 1): the tests at 1 to 4 at the medians of P1 to P4, the test at 5 at P4's 70th
#      percentile, so that =2's bound stops some drafts at 4 before the fifth pass that =1 drafts
#   TR (every width; S run from S0 = 0.905 with LLAMA_SPEC_ADAPT_EMA=1e-6, no drafting cost): TW's thresholds divided
#      by 0.905. The running speed-up then stays within a millionth of 0.905 (with its usual weight it drifts to where
#      the rule keeps nothing: a step that keeps tokens costs more and accepts none on this model), and the drafter
#      takes the S that the slot's state gives, as with S run on the rig
# Groups, each compared with its reference (the first arm):
#   d (TD)                                  dbase BASE; dbuilt BIN; don1 ON=1; don2 ON=2; dvp ON=2 VP=1;
#                                           dvpb VP=1 alone (as built the prepare refuses LLAMA_SPEC_ADAPT: none prepared)
#   w (TW)                                  wbase BASE; won1 ON=1; won2 ON=2
#   r (TR)                                  rbase BASE; ron1 ON=1; ron2 ON=2
#   b (TD, -bs: the backend sampler)        bbase BASE; bon2 ON=2; bvp ON=2 VP=1
#   p (TD, --spec-draft-sampling probabilistic) pbase BASE; pon1 ON=1; pon2 ON=2; pvp ON=2 VP=1;
#                                           pfault ON=2 LLAMA_SPEC_ADAPT_ONLINE_FAULT=1 (the control: the sampler draws
#                                           for the passes left out are skipped, so S1 and S2 must differ from pbase)
#   v (C797's arm settings: FN82's est. v4 table V(1..7), LLAMA_SPEC_ADAPT_DRAFT_MS=21.87, S run; here the rule keeps
#      nothing)                             vbase BASE; von2 ON=2
# Requests (token prompts, the top 3 probabilities of each token): G1 = 200 tokens, 96 greedy tokens; G2 = 600 tokens
# (a 150-token block four times), 128 greedy tokens; S1 = G1's prompt, 96 tokens at temperature 1.0 (top-k 20, top-p
# 0.95, min-p 0, seed 42); S2 = G2's prompt, the same with seed 7.
# Pass, for every arm but the controls against its group's reference:
#   - every request's tokens, top probabilities and draft counts (draft_n, draft_n_accepted: the verified drafts);
#   - the verify steps (LLAMA_SPEC_DUMP without run, step and task; each drafted position's candidates cut to the
#     verified draft, since the rule taken while drafting drafts fewer positions);
#   - the rule's steps (LLAMA_SPEC_ADAPT_LOG without run, step, task, n_draft and p): what it kept, its S, E, C, the
#     accepted count and the next S; and each step's p a prefix of the reference's, n_draft no larger;
#   - ON arms drafted fewer tokens in all (the log's n_draft) and their drafter's closing line counts the passes left
#     out; ON=2 drafted no more than ON=1 of its group, and fewer in w; VP arms prepared verify passes (dvpb none);
#     the references of d, w, r, b and p kept at least three different lengths; no error lines.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18815] [THREADS=8] [ARMS='dbase dbuilt ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,G1..S2.json,dump.jsonl,adapt.jsonl,steps.jsonl,rule.jsonl},
#   OUT/summary.tsv, OUT/passes.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18815}
THREADS=${THREADS:-8}
ARMS=${ARMS:-dbase dbuilt don1 don2 dvp dvpb wbase won1 won2 rbase ron1 ron2 bbase bon2 bvp pbase pon1 pon2 pvp pfault vbase von2}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0"
ADAPT="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_COST=87.597,94.940,105.432,117.985,139.227,152.703,166.181 LLAMA_SPEC_ADAPT_DRAFT_MS=21.87"
TD=100,105,111.3556,111.4,111.476328,111.477,111.477518
TW=100,110.3028,111.36485,111.474371,111.4856174,111.4970678,112.4970678
TR=100,111.38431,112.557846,112.6788637,112.6912907,112.7039431,113.7039431
REQS="G1 G2 S1 S2"
mkdir -p "$OUT"

ref() { echo "${1:0:1}base"; } # the group's reference
declare -A ENVS=(
    [dbase]="" [dbuilt]="" [don1]="LLAMA_SPEC_ADAPT_ONLINE=1" [don2]="LLAMA_SPEC_ADAPT_ONLINE=2"
    [dvp]="LLAMA_SPEC_ADAPT_ONLINE=2 LLAMA_VERIFY_PREPARE=1" [dvpb]="LLAMA_VERIFY_PREPARE=1"
    [wbase]="" [won1]="LLAMA_SPEC_ADAPT_ONLINE=1" [won2]="LLAMA_SPEC_ADAPT_ONLINE=2"
    [rbase]="" [ron1]="LLAMA_SPEC_ADAPT_ONLINE=1" [ron2]="LLAMA_SPEC_ADAPT_ONLINE=2"
    [bbase]="" [bon2]="LLAMA_SPEC_ADAPT_ONLINE=2" [bvp]="LLAMA_SPEC_ADAPT_ONLINE=2 LLAMA_VERIFY_PREPARE=1"
    [pbase]="" [pon1]="LLAMA_SPEC_ADAPT_ONLINE=1" [pon2]="LLAMA_SPEC_ADAPT_ONLINE=2"
    [pvp]="LLAMA_SPEC_ADAPT_ONLINE=2 LLAMA_VERIFY_PREPARE=1"
    [pfault]="LLAMA_SPEC_ADAPT_ONLINE=2 LLAMA_SPEC_ADAPT_ONLINE_FAULT=1"
    [vbase]="" [von2]="LLAMA_SPEC_ADAPT_ONLINE=2"
)
group_env() {
    case "$1" in
        d*|b*|p*) echo "LLAMA_SPEC_ADAPT_COST=$TD LLAMA_SPEC_ADAPT_S=1" ;;
        w*) echo "LLAMA_SPEC_ADAPT_COST=$TW LLAMA_SPEC_ADAPT_S=1 LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7" ;;
        r*) echo "LLAMA_SPEC_ADAPT_COST=$TR LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7 LLAMA_SPEC_ADAPT_S0=0.905 LLAMA_SPEC_ADAPT_EMA=1e-6 LLAMA_SPEC_ADAPT_DRAFT_MS=0" ;;
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
printf 'arm\tsteps\tdrafted\tdrafted_a_step\tkept\taccepted\tc815_drafts\tc815_stopped\tc815_left_out\tc815_draws\tc815_last\tvp_prepared\n' > "$OUT/passes.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    # shellcheck disable=SC2046,SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" LLAMA_SPEC_ADAPT_LOG="$d/adapt.jsonl" $ADAPT $(group_env "$arm") ${ENVS[$arm]:-} \
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
    # the verify steps without the run's start time, the step counter and the task id; the drafted positions'
    # candidates cut to the verified draft
    jq -c 'select(.draft != null) | del(.run, .step, .task) | (.draft | length) as $n
        | if has("cand") then (.cand |= .[:$n]) | (.p |= .[:$n]) | (.pick |= .[:$n]) else . end' \
        "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    # the rule's steps without run, step, task, n_draft and p
    jq -c 'select(.hdr == null) | del(.run, .step, .task, .n_draft, .p)' "$d/adapt.jsonl" > "$d/rule.jsonl" 2>/dev/null
    # the drafter's closing line (C815) and the target context's (C738)
    c8=$(grep -a -o -E 'C815: [0-9]+ drafts under the rule taken while drafting, [0-9]+ stopped early, [0-9]+ drafting passes left out, [0-9]+ sampler draws made for them, [0-9]+ last passes kept' "$d/server.log" | tail -n 1)
    nums=$(grep -o -E '[0-9]+' <<< "${c8:-}" | sed -n '2,6p' | tr '\n' '\t' | sed 's/\t$//')
    [ -n "$nums" ] || nums=$(printf '0\t0\t0\t0\t0')
    vp=$(grep -a -o -E 'C738: [0-9]+ decodes prepared ahead' "$d/server.log" | tail -n 1 | grep -o -E '[0-9]+' | sed -n 2p)
    st=$(jq -r -s 'map(select(.hdr == null)) |
        [length, (map(.n_draft) | add // 0), (if length > 0 then (map(.n_draft) | add) / length else 0 end),
         (map(.n_keep) | add // 0), (map(.n_acc) | add // 0)] | @tsv' "$d/adapt.jsonl" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\n' "$arm" "${st:-0	0	0	0	0}" "$nums" "${vp:-0}" >> "$OUT/passes.tsv"
    echo "$arm: done, $(wc -l < "$d/steps.jsonl") verify steps, $(wc -l < "$d/rule.jsonl") rule steps, ${c8:-no C815 line}, ${vp:-0} verify passes prepared; $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the verdict
fail=0
has() { [[ " $ARMS " == *" $1 "* ]]; }
col() { awk -F'\t' -v a="$1" -v c="$2" '$1 == a { print $c; exit }' "$OUT/passes.tsv"; }
same_reqs() { # <arm a> <arm b> [requests]: tokens, top probabilities and draft counts
    local r
    for r in ${3:-$REQS}; do
        if [ ! -s "$OUT/$1/$r.json" ] || [ ! -s "$OUT/$2/$r.json" ] || ! cmp -s \
                <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$1/$r.json") \
                <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$2/$r.json"); then
            echo "$r"; return 1
        fi
    done
}
same() { # <reference> <arm>
    local a=$1 b=$2 r pre
    if ! r=$(same_reqs "$a" "$b"); then echo "$b differs from $a at $r"; return 1; fi
    # (none on both sides only in group v, where the rule keeps nothing: the kept lengths are checked below)
    if [ ! -f "$OUT/$a/steps.jsonl" ] || { [ ! -s "$OUT/$a/steps.jsonl" ] && [ "${a:0:1}" != v ]; } ||
            ! cmp -s "$OUT/$a/steps.jsonl" "$OUT/$b/steps.jsonl"; then
        echo "$b's verify steps differ from $a's (or $a wrote none)"; return 1
    fi
    if [ ! -s "$OUT/$a/rule.jsonl" ] || ! cmp -s "$OUT/$a/rule.jsonl" "$OUT/$b/rule.jsonl"; then
        echo "$b's rule steps differ from $a's (or $a wrote none)"; return 1
    fi
    # each step's probabilities a prefix of the reference's, and no more drafted
    pre=$(jq -n -r --slurpfile x <(jq -c 'select(.hdr == null) | [.n_draft, .p]' "$OUT/$a/adapt.jsonl") \
        --slurpfile y <(jq -c 'select(.hdr == null) | [.n_draft, .p]' "$OUT/$b/adapt.jsonl") '
        if ($x | length) != ($y | length) then "steps \($x | length) and \($y | length)"
        else [range(0; $x | length) as $i | select(($y[$i][0] > $x[$i][0]) or ($y[$i][1] != $x[$i][1][:($y[$i][1] | length)]))]
            | if length == 0 then "ok" else "\(length) steps not a prefix" end end')
    if [ "$pre" != ok ]; then echo "$b's drafted probabilities against $a's: $pre"; return 1; fi
    echo "$b = $a (tokens, top probabilities and draft counts of $REQS; $(wc -l < "$OUT/$b/steps.jsonl") verify steps and $(wc -l < "$OUT/$b/rule.jsonl") rule steps the same; drafted $(col "$b" 3) tokens against $(col "$a" 3), each step's a prefix)"
}
{
    for b in $ARMS; do
        a=$(ref "$b")
        [ "$b" = "$a" ] && continue
        [ "$b" = pfault ] && continue
        has "$a" || continue
        same "$a" "$b" || fail=1
    done
    # the rule taken while drafting drafts less; =2 no more than =1
    for b in $ARMS; do
        [ "$b" != pfault ] && [[ "${ENVS[$b]:-}" == *LLAMA_SPEC_ADAPT_ONLINE=* ]] || continue
        a=$(ref "$b")
        has "$a" || continue
        da=$(col "$a" 3) db=$(col "$b" 3) st=$(col "$b" 8) lo=$(col "$b" 9)
        if [ "${db:-0}" -lt "${da:-0}" ] && [ "${st:-0}" -gt 0 ] && [ "${lo:-0}" -gt 0 ]; then
            echo "$b: drafted $db tokens against $a's $da ($(col "$b" 4) against $(col "$a" 4) a step); the drafter stopped $st drafts early and left out $lo passes, made $(col "$b" 10) sampler draws for them"
        else
            echo "$b: drafted ${db:-?} tokens against $a's ${da:-?}, stopped ${st:-?}, left out ${lo:-?}: the rule taken while drafting saved nothing"; fail=1
        fi
    done
    for g in d w r p; do
        has "${g}on1" && has "${g}on2" || continue
        d1=$(col "${g}on1" 3) d2=$(col "${g}on2" 3)
        if [ "${d2:-0}" -lt "${d1:-0}" ] || { [ "$g" != w ] && [ "${d2:-0}" -le "${d1:-0}" ]; }; then
            echo "${g}on2 drafted $d2 tokens, ${g}on1 $d1"
        else
            echo "${g}on2 drafted $d2 tokens, ${g}on1 $d1: $([ "$g" = w ] && echo "the bound left out no pass in w" || echo "more")"; fail=1
        fi
    done
    # the references' choices: kept lengths by count; a check needs at least three different ones
    for g in d w r b p v; do
        has "${g}base" || continue
        h=$(jq -r 'select(.hdr == null) | .n_keep' "$OUT/${g}base/adapt.jsonl" | sort -n | uniq -c | awk '{ printf "%s%s: %s", (NR > 1 ? ", " : ""), $2, $1 }')
        n=$(jq -r 'select(.hdr == null) | .n_keep' "$OUT/${g}base/adapt.jsonl" | sort -u | wc -l)
        if [ "$g" = v ] || [ "$n" -ge 3 ]; then
            echo "${g}base kept (length: steps) $h"
        else
            echo "${g}base kept (length: steps) $h: fewer than three lengths, the group checks little"; fail=1
        fi
    done
    for b in dvp bvp pvp; do
        has "$b" || continue
        n=$(col "$b" 12)
        if [ "${n:-0}" -gt 0 ]; then
            echo "$b: $n verify passes prepared during the last drafting pass ($(col "$b" 11) last passes the rule kept)"
        else
            echo "$b: no verify pass prepared"; fail=1
        fi
    done
    if has dvpb; then
        n=$(col dvpb 12)
        [ "${n:-0}" = 0 ] && echo "dvpb (as built with VP=1): no verify pass prepared, as expected" || { echo "dvpb prepared $n verify passes"; fail=1; }
    fi
    if has pfault && has pbase; then
        if g=$(same_reqs pbase pfault "G1 G2") && ! same_reqs pbase pfault "S1 S2" > /dev/null; then
            echo "pfault (control): G1 and G2 = pbase, S1 or S2 differ: without the draws a probabilistic draft's later steps change (AS-EXPECTED)"
        else
            echo "pfault (control): did not show the draws matter (G: ${g:-same}; S1 and S2 the same as pbase)"; fail=1
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
