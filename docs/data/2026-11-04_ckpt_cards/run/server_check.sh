#!/usr/bin/env bash
# C22 / C28 (Q440, Q448, Q548): llama-server's context checkpoints kept on the devices (--ctx-checkpoints-vram) and
# taken inside long prompts (--ctx-checkpoints-spread), checked on the CPU with a random-weight recurrent model
# (test-llama-archs -o). The same five requests go to six servers that differ only in where the checkpoints live and
# where they are taken:
#   base       the release line: checkpoints on the host, at a prompt's end (and after the prefix, with a chat template)
#   vram       --ctx-checkpoints-vram 8: all of them in device stores (on the CPU the stores are CPU buffers: the same
#              code path, a memcpy instead of a copy inside the GPU)
#   vram2      --ctx-checkpoints-vram 2: the newest two in stores, older ones moved to the host in the background
#   vram2s     as vram2, moved at once (--ctx-checkpoints-vram-sync)
#   spread     --ctx-checkpoints-spread 512: checkpoints inside the prompt (here every 512 tokens: token prompts have
#              no message starts), a full list drops the one that leaves the smallest gap
#   spread_v2  spread + --ctx-checkpoints-vram 2
# Requests (greedy, token prompts, 16 tokens each): P1 = 3000 tokens; P2 = P1 + its reply + 500 tokens (continues the
# live state); P3 = P2 with one token changed at 2200 (history edited far back); P4 = P2 again; P5 = P3 + its reply +
# 300 tokens.
# Pass: vram, vram2 and vram2s give base's tokens and top probabilities exactly (where a checkpoint lives changes no
# number), spread_v2 gives spread's exactly, spread's P3 resumes from a checkpoint at or after 2048 tokens, and no server
# logs an error. spread against base is reported, not required: it splits the prompt into other batches.
# Usage: BIN=<dir with llama-server> MODEL=<gguf> OUT=<dir> [PORT=18080] [THREADS=8] [ARMS='base vram ...'] \
#   [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,P1..P5.json}, OUT/summary.tsv, OUT/verdict.txt
set -uo pipefail
BIN=${BIN:?BIN=<dir with llama-server>}
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18080}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base vram vram2 vram2s spread spread_v2}
SARGS=${SARGS:-}
mkdir -p "$OUT"

declare -A FLAGS=(
    [base]=""
    [vram]="--ctx-checkpoints-vram 8"
    [vram2]="--ctx-checkpoints-vram 2"
    [vram2s]="--ctx-checkpoints-vram 2 --ctx-checkpoints-vram-sync"
    [spread]="--ctx-checkpoints-spread 512"
    [spread_v2]="--ctx-checkpoints-spread 512 --ctx-checkpoints-vram 2"
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (a Park-Miller generator, exact in awk's
# doubles: mawk's srand(seed) does not repeat its sequence)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <comma-separated tokens>
    printf '{"prompt":[%s],"n_predict":16,"top_k":1,"temperature":0,"n_probs":3,"return_tokens":true,"cache_prompt":true}' "$3" \
        > "$1/$2.req.json"
    curl -s --max-time 900 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

reply() { jq -r '.tokens | map(tostring) | join(",")' "$1"; }

printf 'arm\treq\tprompt_n\tcache_n\tcache_src\tcache_ckpt_n\tckpt_created_n\tckpt_dev_n\tckpt_save_ms\tcache_restore_ms\tprompt_ms\ttokens\n' > "$OUT/summary.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    mkdir -p "$d"
    "$BIN/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 8 --cache-ram 0 \
        --no-webui -lv 4 --host 127.0.0.1 --port "$PORT" $SARGS ${FLAGS[$arm]} > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -20 "$d/server.log"; kill "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')

    p1=$(toks 3000 1 "$nv")
    complete "$d" P1 "$p1"
    p2="$p1,$(reply "$d/P1.json"),$(toks 500 2 "$nv")"
    complete "$d" P2 "$p2"
    # one token changed at 2200
    p3=$(echo "$p2" | awk -F, -v OFS=, -v v="$nv" '{ $2201 = ($2201 % (v - 1)) + 1; print }')
    complete "$d" P3 "$p3"
    complete "$d" P4 "$p2"
    p5="$p3,$(reply "$d/P3.json"),$(toks 300 3 "$nv")"
    complete "$d" P5 "$p5"

    kill "$pid"; wait "$pid" 2>/dev/null
    for r in P1 P2 P3 P4 P5; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.cache_n, .timings.cache_src,
            .timings.cache_ckpt_n, .timings.ckpt_created_n, .timings.ckpt_dev_n, .timings.ckpt_save_ms,
            .timings.cache_restore_ms, .timings.prompt_ms, (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" \
            >> "$OUT/summary.tsv"
    done
    echo "$arm: done, $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
done

# the verdict
fail=0
same() { # <arm a> <arm b>: tokens and top probabilities of every request, exactly
    for r in P1 P2 P3 P4 P5; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    echo "$2 = $1 (tokens and top probabilities of P1-P5)"
}
{
    for a in vram vram2 vram2s; do
        [[ " $ARMS " == *" $a "* ]] && { same base "$a" || fail=1; }
    done
    if [[ " $ARMS " == *" spread_v2 "* ]]; then same spread spread_v2 || fail=1; fi
    if [[ " $ARMS " == *" spread "* ]]; then
        n=$(jq -r '.timings.cache_ckpt_n' "$OUT/spread/P3.json")
        if [ "$n" -ge 2048 ] 2>/dev/null; then echo "spread P3 resumed from the checkpoint at $n tokens"; else echo "spread P3 resumed from $n, not >= 2048"; fail=1; fi
        same base spread > /dev/null && echo "spread = base as well" || echo "spread differs from base (other batches; reported only)"
    fi
    for a in $ARMS; do
        e=$(grep -ciE 'error|abort|failed to restore' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
        grep -q 'moved to the host' "$OUT/$a/server.log" && echo "$a: $(grep -c 'moved to the host' "$OUT/$a/server.log") checkpoint(s) moved to the host"
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
} | tee "$OUT/verdict.txt"
exit $fail
