#!/usr/bin/env bash
# C218 (Q771) on the CPU: llama-server with the micro-batch taper against the build it was added to, on a random-weight
# qwen4exp model (Flash-Next's architecture, from test-llama-archs), at the rig's batch settings (-b 2048 -ub 512) with
# context checkpoints, so that the server cuts every prompt 516 and 4 tokens before its end and reads a checkpoint
# there. The same requests go to one server per arm:
#   base     BASE_BIN (the build before the taper, -ladder2)
#   off      BIN with no switch: must give base's tokens, probabilities, checkpoints and restores, and log no taper line
#   on       BIN with LLAMA_UBATCH_TAPER=on (head=256:64,tail=512:128,short=600:64)
#   flag     BIN with --ubatch-taper head=512:16,tail=512:16,short=600:16
#   req      BIN with no switch; every request carries "ubatch_taper": "head=100:7,tail=50:13,short=300:5"
#   badreq   BIN with no switch; every request carries "ubatch_taper": "bogus" (the server keeps its own, off, and warns)
# Requests (token prompts, greedy, 8 tokens out, top-3 probabilities, cache_prompt on), for each prompt length L in LENS
# (default 100 600 3000 9000):
#   A<L>  a fresh prompt of L tokens (nothing in common with the slot's cache: processed from its first token)
#   C<L>  A<L>'s prompt, its 8 output tokens and 40 new ones (an agent's next turn: the live state continues)
#   B<L>  A<L>'s prompt with its last 10 tokens replaced by 30 others (the server restores a checkpoint before them)
# Pass: every arm gives base's tokens, top-3 probabilities, prompt_n and cache_n for every request; the same checkpoint
# creations (pos_min, pos_max, n_tokens, in order) and restores (pos_min, pos_max, n_tokens, n_past) in its log; on,
# flag and req log taper lines (one per prompt call above 8 tokens) and off, base and badreq none; badreq logs the
# warning; no log has an error line other than badreq's expected one.
# Usage: BASE_BIN=<dir> BIN=<dir> MODEL=<gguf> OUT=<dir> [PORT=18093] [THREADS=8] [ARMS='base off ...'] [LENS='100 ...']
#   bash server_check.sh
# Needs curl and jq. Output: OUT/<arm>/{server.log,*.json,ckpt.txt,taper.txt}, OUT/summary.tsv, OUT/verdict.txt
set -uo pipefail
BASE_BIN=${BASE_BIN:?BASE_BIN=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with llama-server>}
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18093}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off on flag req badreq}
LENS=${LENS:-100 600 3000 9000}
mkdir -p "$OUT"

declare -A ENVS=([base]="" [off]="" [on]="LLAMA_UBATCH_TAPER=on" [flag]="" [req]="" [badreq]="")
declare -A FLAGS=([flag]="--ubatch-taper head=512:16,tail=512:16,short=600:16")
declare -A REQ=([req]="head=100:7,tail=50:13,short=300:5" [badreq]="bogus")

# deterministic token lists: n tokens below the vocabulary size, from a seed (Park-Miller, exact in awk's doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <tokens> <taper or empty>
    local extra=""
    [ -n "$4" ] && extra=",\"ubatch_taper\":\"$4\""
    printf '{"prompt":[%s],"n_predict":8,"temperature":0,"top_k":1,"seed":42,"n_probs":3,"return_tokens":true,"cache_prompt":true%s}' \
        "$3" "$extra" > "$1/$2.req.json"
    curl -s --max-time 1800 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

errors() { sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E ' E |[Ee]rror|[Aa]bort' | grep -v 'set_abort_callback' | grep -vc "bad micro-batch taper 'bogus'"; }

printf 'arm\treq\tprompt_n\tcache_n\ttokens\n' > "$OUT/summary.tsv"
NAMES=""
for L in $LENS; do NAMES="$NAMES A$L C$L B$L"; done

for arm in $ARMS; do
    d="$OUT/$arm"
    rm -rf "$d"; mkdir -p "$d"
    b=$BIN; [ "$arm" = base ] && b=$BASE_BIN
    # shellcheck disable=SC2086
    env ${ENVS[$arm]:-} "$b/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" -c 16384 -b 2048 -ub 512 -np 1 \
        --ctx-checkpoints 4 --cache-ram 0 --no-webui -lv 5 ${FLAGS[$arm]:-} \
        --override-kv qwen4exp.context_length=int:16384 --host 127.0.0.1 --port "$PORT" > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -20 "$d/server.log"; kill "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')
    for L in $LENS; do
        a=$(toks "$L" "$L" "$nv")
        complete "$d" "A$L" "$a" "${REQ[$arm]:-}"
        gen=$(jq -r '.tokens | map(tostring) | join(",")' "$d/A$L.json")
        complete "$d" "C$L" "$a,$gen,$(toks 40 $((L + 7)) "$nv")" "${REQ[$arm]:-}"
        head=$(echo "$a" | cut -d, -f1-$((L - 10)))
        complete "$d" "B$L" "$head,$(toks 30 $((L + 3)) "$nv")" "${REQ[$arm]:-}"
    done
    kill "$pid"; wait "$pid" 2>/dev/null
    for r in $NAMES; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.cache_n,
            (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | sed -n \
        -e 's/.*created context checkpoint [0-9]* of [0-9]* (\(pos_min = [0-9-]*, pos_max = [0-9-]*, n_tokens = [0-9]*\).*/created \1/p' \
        -e 's/.*restored context checkpoint (\(pos_min = [0-9-]*, pos_max = [0-9-]*, n_tokens = [0-9]*, n_past = [0-9]*\).*/restored \1/p' \
        -e 's/.*\(forcing full prompt re-processing.*\)/\1/p' > "$d/ckpt.txt"
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -o 'ubatch taper: .*' > "$d/taper.txt"
    echo "$arm: done, $(wc -l < "$d/ckpt.txt") checkpoint lines, $(wc -l < "$d/taper.txt") taper lines, $(errors "$d/server.log") error lines"
done

fail=0
same() { # <arm a> <arm b>: tokens, top probabilities, prompt_n and cache_n of every request, and the checkpoint lines
    for r in $NAMES; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities, .timings.prompt_n, .timings.cache_n]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities, .timings.prompt_n, .timings.cache_n]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    if ! cmp -s "$OUT/$1/ckpt.txt" "$OUT/$2/ckpt.txt"; then
        echo "$2: checkpoint lines differ from $1's"; diff "$OUT/$1/ckpt.txt" "$OUT/$2/ckpt.txt" | head -10; return 1
    fi
    echo "$2 = $1 (tokens, top-3 probabilities, prompt_n, cache_n of $(echo $NAMES | wc -w) requests; $(wc -l < "$OUT/$2/ckpt.txt") checkpoint lines)"
}
{
    for a in $ARMS; do
        [ "$a" = base ] && continue
        same base "$a" || fail=1
        n=$(wc -l < "$OUT/$a/taper.txt")
        case $a in
            on|flag|req) [ "$n" -gt 0 ] || { echo "$a: no taper line"; fail=1; } ;;
            *) [ "$n" = 0 ] || { echo "$a: $n taper lines with the taper off"; fail=1; } ;;
        esac
        if [ "$a" = badreq ]; then
            grep -q "bad ubatch_taper 'bogus'" "$OUT/$a/server.log" || { echo "badreq: no warning"; fail=1; }
        fi
    done
    for a in $ARMS; do
        e=$(errors "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
} | tee "$OUT/verdict.txt"
grep -qx PASS "$OUT/verdict.txt"
