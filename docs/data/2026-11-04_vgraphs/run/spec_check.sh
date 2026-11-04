#!/usr/bin/env bash
# C151 (Q701): llama-server's speculative decoding on the CPU with the per-size graphs on and off. A random-weight model
# from test-llama-archs drafts for itself (--spec-type draft-simple), so every verify batch is 1 + 0..3 tokens wide and
# the widths alternate. The same requests go to one server per arm; the arms differ only in the switches of the
# per-size graph port (upstream llama.cpp #31156 and #31274):
#   base     the defaults: one graph arena per batch size up to 8 tokens, one shared scheduler, calloc'd graph metadata
#   gps0     LLAMA_GRAPH_PER_SIZE=0: the two arenas of before #31156 (with and without outputs), as -ladder2
#   sps1     LLAMA_SCHED_PER_SIZE=1: a scheduler per arena of up to 8 tokens
#   calloc0  LLAMA_GRAPH_META_CALLOC=0: every page of the graph metadata written once, as before #31274
#   noreuse  LLAMA_GRAPH_REUSE_DISABLE=1: every graph rebuilt (the reference without any reuse)
# Requests (token prompts of 40-90 tokens, 64-96 tokens generated, seed 42; their speculative.n_max and p_min fields):
# R1 greedy, n_max 3; R2 temperature 1.0, n_max 3; R3 temperature 1.0, n_max 1; R4 greedy, n_max 2, p_min 0.3; R5
# temperature 1.0, n_max 3, p_min 0.3. The widths the steps actually took are counted from the log (widths.tsv).
# Pass: every arm gives base's tokens and top probabilities for every request (on the CPU the same batches compute the
# same numbers wherever their graphs live), the requests drafted, base's log shows at least two verify widths, sps1 logs
# its per-size schedulers, and no log has an error line.
# Usage: BIN=<dir with llama-server> MODEL=<gguf> OUT=<dir> [PORT=18090] [THREADS=8] [ARMS='base gps0 ...'] \
#   [SARGS='<more server arguments>'] bash spec_check.sh
# Needs curl and jq. Output: OUT/<arm>/{server.log,R1..R5.json,status.txt}, OUT/summary.tsv, OUT/widths.tsv,
# OUT/verdict.txt
set -uo pipefail
BIN=${BIN:?BIN=<dir with llama-server>}
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18090}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base gps0 sps1 calloc0 noreuse}
SARGS=${SARGS:-}
mkdir -p "$OUT"

declare -A ENVS=(
    [base]=""
    [gps0]="LLAMA_GRAPH_PER_SIZE=0"
    [sps1]="LLAMA_SCHED_PER_SIZE=1"
    [calloc0]="LLAMA_GRAPH_META_CALLOC=0"
    [noreuse]="LLAMA_GRAPH_REUSE_DISABLE=1"
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (Park-Miller, exact in awk's doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <tokens> <n_predict> <temperature> <n_max> <p_min>
    printf '{"prompt":[%s],"n_predict":%d,"temperature":%s,"top_k":%s,"seed":42,"n_probs":3,"return_tokens":true,"cache_prompt":false,"speculative.n_max":%d,"speculative.n_min":0,"speculative.p_min":%s}' \
        "$3" "$4" "$5" "$([ "$5" = 0 ] && echo 1 || echo 40)" "$6" "$7" > "$1/$2.req.json"
    curl -s --max-time 900 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

# error lines of a server log: an error-level line, or one that says error or abort (not the debug log's
# set_abort_callback calls)
errors() { sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E ' E |[Ee]rror|[Aa]bort' | grep -vc 'set_abort_callback'; }

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\tpredicted_ms\ttokens\n' > "$OUT/summary.tsv"
printf 'arm\twidth\tsteps\n' > "$OUT/widths.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    mkdir -p "$d"
    # shellcheck disable=SC2086
    env ${ENVS[$arm]} "$BIN/llama-server" -m "$MODEL" -md "$MODEL" --spec-type draft-simple --spec-draft-n-max 3 \
        -ngl 0 -ngld 0 -t "$THREADS" -c 4096 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 --cache-ram 0 --no-webui -lv 5 \
        --host 127.0.0.1 --port "$PORT" $SARGS > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -20 "$d/server.log"; kill "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')

    complete "$d" R1 "$(toks 40 11 "$nv")" 96 0   3 0.0
    complete "$d" R2 "$(toks 60 12 "$nv")" 96 1.0 3 0.0
    complete "$d" R3 "$(toks 50 13 "$nv")" 64 1.0 1 0.0
    complete "$d" R4 "$(toks 90 14 "$nv")" 96 0   2 0.3
    complete "$d" R5 "$(toks 70 15 "$nv")" 96 1.0 3 0.3
    grep -E '^(VmRSS|RssAnon|RssFile|VmHWM)' "/proc/$pid/status" > "$d/status.txt"

    kill "$pid"; wait "$pid" 2>/dev/null
    for r in R1 R2 R3 R4 R5; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, .timings.draft_n,
            .timings.draft_n_accepted, .timings.predicted_ms, (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" \
            >> "$OUT/summary.tsv"
    done
    # verify widths: a step without a draft decodes 1 token, a step with a draft of n tokens decodes 1 + n
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | awk -v a="$arm" '
        /slot decode token, id=/ { w[1]++ }
        /generate_draft: id=/ { if (match($0, /#draft=[0-9]+/)) w[1 + substr($0, RSTART + 7, RLENGTH - 7)]++ }
        END { for (k in w) print a "\t" k "\t" w[k] }' | sort -t$'\t' -k2,2n >> "$OUT/widths.tsv"
    echo "$arm: done, $(errors "$d/server.log") error lines in its log, $(tr '\n' ' ' < "$d/status.txt")"
done

fail=0
same() { # <arm a> <arm b>: tokens and top probabilities of every request, exactly
    for r in R1 R2 R3 R4 R5; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    echo "$2 = $1 (tokens and top probabilities of R1-R5)"
}
{
    for a in $ARMS; do
        [ "$a" = base ] && continue
        same base "$a" || fail=1
    done
    nd=$(awk -F'\t' '$1 == "base" { s += $5 } END { print s + 0 }' "$OUT/summary.tsv")
    na=$(awk -F'\t' '$1 == "base" { s += $6 } END { print s + 0 }' "$OUT/summary.tsv")
    nw=$(awk -F'\t' '$1 == "base" && $3 > 0' "$OUT/widths.tsv" | wc -l)
    echo "base: $nd drafted, $na accepted, $nw verify widths ($(awk -F'\t' '$1 == "base" { printf "%s:%s ", $2, $3 }' "$OUT/widths.tsv"))"
    [ "$nd" -gt 0 ] && [ "$nw" -ge 2 ] || { echo "base did not draft at two widths"; fail=1; }
    if [[ " $ARMS " == *" sps1 "* ]]; then
        n=$(grep -c 'per-size schedulers' "$OUT/sps1/server.log")
        echo "sps1: $n per-size schedulers created ($(grep -o 'scheduler for [0-9]*-token batches ([a-z]* outputs)' "$OUT/sps1/server.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';'))"
        [ "$n" -ge 2 ] || fail=1
    fi
    for a in $ARMS; do
        e=$(errors "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
        grep -m 1 'graphs per batch size' "$OUT/$a/server.log" | sed "s/^/$a: /"
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
} | tee "$OUT/verdict.txt"
# the block above runs in a subshell of the pipe: its last line is the verdict
grep -qx PASS "$OUT/verdict.txt"
