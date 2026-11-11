#!/usr/bin/env bash
# C735 (Q1302): llama-server's context checkpoints inside a prompt's decode call (--ctx-checkpoints-incall, C217) with
# the MTP draft's catch-up queued behind the next call (LLAMA_MTP_CATCHUP_OVERLAP=1, C703), checked on the CPU with a
# random-weight qwen4exp model with one MTP block (test-llama-archs -o --mtp). Made from C217's server_check.sh
# (docs/data/2026-11-11_c217_incall_checkpoints): the same five requests, every arm with the MTP draft. BASE is the
# base branch's llama-server, BIN the branch's.
# Arms (all -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 8 --cache-ram 0, --spec-type draft-mtp
# --spec-draft-n-max 3 --spec-draft-p-min 0; OVL = LLAMA_MTP_CATCHUP_OVERLAP=1):
#   base     BASE, --ctx-checkpoints-vram 8: the reference
#   off      BIN, the same flags: both switches off
#   ovl      BIN, OVL: C703 alone
#   incall   BIN, + --ctx-checkpoints-incall: C217 alone
#   incall2  BIN, --ctx-checkpoints-vram 2 --ctx-checkpoints-incall: C217 alone, older checkpoints move to the host
#   both     BIN, + --ctx-checkpoints-incall, OVL
#   both2    BIN, --ctx-checkpoints-vram 2 --ctx-checkpoints-incall, OVL: a move to the host may meet a draft part that
#            waits for its catch-up
#   base4    BASE, --ctx-checkpoints 4 --ctx-checkpoints-vram 4 (the release preset's counts)
#   incall4  BIN, the same + --ctx-checkpoints-incall
#   both4    BIN, the same + --ctx-checkpoints-incall, OVL
#   sbase    BASE, --ctx-checkpoints-vram 8 --ctx-checkpoints-spread 200: checkpoints every 200 tokens inside a prompt
#   sincall  BIN, the same + --ctx-checkpoints-incall
#   sboth    BIN, the same + --ctx-checkpoints-incall, OVL
#   bboth    BASE, --ctx-checkpoints-vram 8 --ctx-checkpoints-incall, OVL: the control, the base with both switches
# Requests (token prompts, 16 greedy tokens and the top 3 probabilities each): Q1 = 3000 tokens; Q2 = Q1's first 2950
# and 200 new ones (resumes from a checkpoint below 2950); Q3 = Q2, its reply and 400 new ones (the live state); Q4 = Q1
# again (resumes from a checkpoint below 2950); Q5 = Q3 with the token at 3100 changed (resumes from the checkpoint
# Q4 left below it).
# Every arm writes LLAMA_SPEC_DUMP: one line per verify step with the draft, the drafter's candidates and their
# probabilities and the target's top tokens, so that a draft state that differs (a checkpoint's draft part or the
# driver's state at it, restored by Q2, Q4 and Q5) shows even when no draft is accepted.
# Pass: off, ovl, incall, incall2, both and both2 give base's tokens, top probabilities, prompt_n, cache_n, restored
# checkpoint (cache_ckpt_n), draft counts (draft_n, draft_n_accepted) and verify steps (the dump without its run, step and
# task fields) on every request, and their servers take checkpoints at base's token counts, in its order (one taken
# inside a call counts when it joins its batch); incall4 and both4 the same against base4, sincall and sboth against
# sbase; the incall and both arms took checkpoints inside calls; the OVL arms say the overlap is on; no arm of BIN logs
# an error or erases an in-call checkpoint. bboth is reported, not judged: on the base its draft parts were read before
# the draft's pass, so it is expected to erase in-call checkpoints or to differ.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   OUT=<dir> [PORT=18735] [THREADS=8] [ARMS='base off ...'] [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,Q1..Q5.json,ckpts.txt,dump.jsonl,steps.jsonl}, OUT/summary.tsv,
#   OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18735}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off ovl incall incall2 both both2 base4 incall4 both4 sbase sincall sboth bboth}
SARGS=${SARGS:-}
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"
OVL=LLAMA_MTP_CATCHUP_OVERLAP=1
mkdir -p "$OUT"

declare -A EXE=(
    [base]=$BASE [off]=$BIN [ovl]=$BIN [incall]=$BIN [incall2]=$BIN [both]=$BIN [both2]=$BIN
    [base4]=$BASE [incall4]=$BIN [both4]=$BIN [sbase]=$BASE [sincall]=$BIN [sboth]=$BIN [bboth]=$BASE
)
declare -A FLAGS=(
    [base]="--ctx-checkpoints-vram 8"
    [off]="--ctx-checkpoints-vram 8"
    [ovl]="--ctx-checkpoints-vram 8"
    [incall]="--ctx-checkpoints-vram 8 --ctx-checkpoints-incall"
    [incall2]="--ctx-checkpoints-vram 2 --ctx-checkpoints-incall"
    [both]="--ctx-checkpoints-vram 8 --ctx-checkpoints-incall"
    [both2]="--ctx-checkpoints-vram 2 --ctx-checkpoints-incall"
    [base4]="--ctx-checkpoints 4 --ctx-checkpoints-vram 4"
    [incall4]="--ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
    [both4]="--ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
    [sbase]="--ctx-checkpoints-vram 8 --ctx-checkpoints-spread 200"
    [sincall]="--ctx-checkpoints-vram 8 --ctx-checkpoints-spread 200 --ctx-checkpoints-incall"
    [sboth]="--ctx-checkpoints-vram 8 --ctx-checkpoints-spread 200 --ctx-checkpoints-incall"
    [bboth]="--ctx-checkpoints-vram 8 --ctx-checkpoints-incall"
)
declare -A ENVS=(
    [ovl]=$OVL [both]=$OVL [both2]=$OVL [both4]=$OVL [sboth]=$OVL [bboth]=$OVL
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (a Park-Miller generator, exact in awk's
# doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }
first() { echo "$1" | cut -d, -f"1-$2"; }

complete() { # <arm dir> <name> <comma-separated tokens>
    printf '{"prompt":[%s],"n_predict":16,"top_k":1,"temperature":0,"n_probs":3,"return_tokens":true,"cache_prompt":true}' "$3" \
        > "$1/$2.req.json"
    curl -s --max-time 1800 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

reply() { jq -r '.tokens | map(tostring) | join(",")' "$1"; }

printf 'arm\treq\tprompt_n\tcache_n\tcache_src\tcache_ckpt_n\tckpt_created_n\tckpt_dev_n\tckpt_incall_n\tdraft_n\tdraft_n_accepted\tckpt_save_ms\tcache_restore_ms\tprompt_ms\ttokens\n' > "$OUT/summary.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d" && mkdir -p "$d"
    # shellcheck disable=SC2086
    env LLAMA_SPEC_DUMP="$d/dump.jsonl" ${ENVS[$arm]:-} "${EXE[$arm]}/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" \
        -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 8 --cache-ram 0 --no-webui -lv 4 --host 127.0.0.1 --port "$PORT" \
        $MTP $SARGS ${FLAGS[$arm]} > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -n 20 "$d/server.log"; kill "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')

    q1=$(toks 3000 1 "$nv")
    complete "$d" Q1 "$q1"
    q2="$(first "$q1" 2950),$(toks 200 2 "$nv")"
    complete "$d" Q2 "$q2"
    q3="$q2,$(reply "$d/Q2.json"),$(toks 400 3 "$nv")"
    complete "$d" Q3 "$q3"
    complete "$d" Q4 "$q1"
    # the token at 3100 (field 3101) changed
    q5=$(echo "$q3" | awk -F, -v OFS=, -v v="$nv" '{ $3101 = ($3101 % (v - 1)) + 1; print }')
    complete "$d" Q5 "$q5"

    # SIGTERM and wait: the driver's closing lines and the dump are whole
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    for r in Q1 Q2 Q3 Q4 Q5; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.cache_n, .timings.cache_src,
            .timings.cache_ckpt_n, .timings.ckpt_created_n, .timings.ckpt_dev_n, (.timings.ckpt_incall_n // 0),
            (.timings.draft_n // 0), (.timings.draft_n_accepted // 0),
            .timings.ckpt_save_ms, .timings.cache_restore_ms, .timings.prompt_ms, (.tokens | map(tostring) | join(" "))] | @tsv' \
            "$d/$r.json" >> "$OUT/summary.tsv"
    done
    # the token counts of the checkpoints the server took, in the order it took them: those before a call when created,
    # those inside a call when they join the batch (one that a later one of the same batch evicts is never filled)
    grep -o -E 'created context checkpoint [0-9]+ of [0-9]+ \([^)]*n_tokens = [0-9]+|context checkpoint at n_tokens = [0-9]+ goes inside the call' \
        "$d/server.log" | grep -o -E 'n_tokens = [0-9]+' | sed 's/n_tokens = //' > "$d/ckpts.txt"
    # the verify steps without the run's start time, the step counter and the task id (it counts the server's own
    # tasks too, and a prompt in fewer calls makes fewer of them)
    jq -c 'select(.draft != null) | del(.run, .step, .task)' "$d/dump.jsonl" > "$d/steps.jsonl" 2>/dev/null
    echo "$arm: done, $(wc -l < "$d/ckpts.txt") checkpoints ($(grep -c 'goes inside the call' "$d/server.log") inside a call, $(grep -c 'created context checkpoint [0-9]* of [0-9]* inside the call' "$d/server.log") filled), $(wc -l < "$d/steps.jsonl") verify steps, $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the verdict
fail=0
same() { # <arm a> <arm b>: tokens, top probabilities, prompt_n, cache_n, restored checkpoint and draft counts of every
         # request, the checkpoints' token counts and the verify steps
    local r
    for r in Q1 Q2 Q3 Q4 Q5; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities, .timings.prompt_n, .timings.cache_n, .timings.cache_ckpt_n, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities, .timings.prompt_n, .timings.cache_n, .timings.cache_ckpt_n, .timings.draft_n, .timings.draft_n_accepted]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    if ! cmp -s "$OUT/$1/ckpts.txt" "$OUT/$2/ckpts.txt"; then
        echo "$2 takes checkpoints at other token counts than $1"; return 1
    fi
    if [ ! -s "$OUT/$1/steps.jsonl" ] || ! cmp -s "$OUT/$1/steps.jsonl" "$OUT/$2/steps.jsonl"; then
        echo "$2's verify steps differ from $1's (or $1 wrote none)"; return 1
    fi
    echo "$2 = $1 (tokens, top probabilities, prompt_n, cache_n, restored checkpoint and draft counts of Q1-Q5; $(wc -l < "$OUT/$2/ckpts.txt") checkpoints at the same token counts; $(wc -l < "$OUT/$2/steps.jsonl") verify steps the same)"
}
has() { [[ " $ARMS " == *" $1 "* ]]; }
{
    for a in off ovl incall incall2 both both2; do
        has base && has "$a" && { same base "$a" || fail=1; }
    done
    for a in incall4 both4; do
        has base4 && has "$a" && { same base4 "$a" || fail=1; }
    done
    for a in sincall sboth; do
        has sbase && has "$a" && { same sbase "$a" || fail=1; }
    done
    for a in incall incall2 both both2 incall4 both4 sincall sboth; do
        has "$a" || continue
        n=$(awk -F'\t' -v a="$a" '$1 == a { s += $9 } END { print s + 0 }' "$OUT/summary.tsv")
        if [ "$n" -gt 0 ]; then echo "$a: $n checkpoints taken inside a call"; else echo "$a: no checkpoint taken inside a call"; fail=1; fi
    done
    for a in ovl both both2 both4 sboth; do
        has "$a" || continue
        if grep -q 'LLAMA_MTP_CATCHUP_OVERLAP: a prompt call' "$OUT/$a/server.log"; then
            echo "$a: the overlap is on; $(grep -o -E '[0-9]+ catch-ups ran after the next call was queued' "$OUT/$a/server.log" | tail -n 1)"
        else
            echo "$a: the overlap is not on"; fail=1
        fi
    done
    for a in $ARMS; do
        [ "$a" = bboth ] && continue
        e=$(grep -ciE 'error|abort|failed to restore' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
        grep -q 'moved to the host' "$OUT/$a/server.log" && echo "$a: $(grep -c 'moved to the host' "$OUT/$a/server.log") checkpoint(s) moved to the host"
        grep -q 'erased: ' "$OUT/$a/server.log" && echo "$a: $(grep -c 'erased: ' "$OUT/$a/server.log") in-call checkpoint(s) dropped"
        if grep -q 'was not taken inside the call' "$OUT/$a/server.log"; then
            echo "$a: $(grep -c 'was not taken inside the call' "$OUT/$a/server.log") in-call checkpoint(s) not taken inside the call"; fail=1
        fi
    done
    if has bboth; then
        ne=$(grep -c 'was not taken inside the call' "$OUT/bboth/server.log")
        nb=$(awk -F'\t' '$1 == "bboth" { s += $9 } END { print s + 0 }' "$OUT/summary.tsv")
        if has base && same base bboth > /dev/null; then
            echo "bboth (the control, the base with both switches): = base, $nb checkpoints inside a call, $ne erased: the check does not see the base's defect"
        else
            echo "bboth (the control, the base with both switches): $(same base bboth | head -n 1); $nb checkpoints inside a call, $ne erased (expected)"
        fi
    fi
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
