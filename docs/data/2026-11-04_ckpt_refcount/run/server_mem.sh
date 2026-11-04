#!/usr/bin/env bash
# C154 (Q704): llama-server's host memory with and without reference-counted context checkpoints
# (--ctx-checkpoints-share, upstream #30137), checked on the CPU. The same eight requests go to servers that differ
# only in the switch, the prompt cache (--cache-ram) and where the checkpoints live or are taken; optionally the same
# arms on a build without the change (OLD_BIN). Each server's peak resident memory (VmHWM) and its resident memory
# after each request (VmRSS, RssAnon) come from /proc/<pid>/status.
# Requests (greedy, token prompts, 16 tokens each): P1 = 3000 tokens; P2 = P1 + its reply + 500 tokens; P3 = P2 with
# one token changed at 2200; P4 = P2 again; P5 = P3 + its reply + 300 tokens; P6 = another 3000-token conversation
# (with a prompt cache, the slot's prompt and checkpoints go into it); P7 = P5 + its reply + 200 tokens (back to the
# first conversation: restored from the cache, the second one saved); P8 = P6 + its reply + 200 tokens.
# Arms (the base flags, then):
#   c0_copy  c0_share   --cache-ram 0 (the rig's lines), without / with --ctx-checkpoints-share
#   cr_copy  cr_share   --cache-ram 4096: a prompt cache
#   cr2_copy cr2_share  as cr, --ctx-checkpoints 2
#   crv_copy crv_share  as cr, --ctx-checkpoints-vram 2 (device stores; on the CPU, CPU buffers: the same code path)
#   crs_copy crs_share  as cr, --ctx-checkpoints-spread 512 (checkpoints inside the prompt)
#   c0_old   cr_old     c0 and cr on OLD_BIN (a build without the change; no switch)
# Pass: every *_share arm gives its *_copy arm's tokens and top probabilities exactly, every *_old arm too, and no
# server logs an error. The memory is reported: with a prompt cache, a copy arm holds each saved checkpoint twice
# from the save until the slot drops its own (est. the checkpoints' size more at the peak); with --cache-ram 0 nothing
# copies a checkpoint, so the switch changes nothing.
# Usage: BIN=<dir with llama-server> MODEL=<gguf> OUT=<dir> [OLD_BIN=<dir>] [PORT=18090] [THREADS=8] [CKPT=4]
#   [CTX=16384] [ARMS='c0_copy c0_share ...'] [SARGS='<more server arguments>'] bash server_mem.sh
# A random test model (test-llama-archs -o) trains on 256 tokens: give it a longer context, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl and jq. Output: OUT/<arm>/{server.log,P1..P8.json}, OUT/mem.tsv, OUT/requests.tsv, OUT/verdict.txt
set -uo pipefail
BIN=${BIN:?BIN=<dir with llama-server>}
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
OLD_BIN=${OLD_BIN:-}
PORT=${PORT:-18090}
THREADS=${THREADS:-8}
CKPT=${CKPT:-4}
CTX=${CTX:-16384}
ARMS=${ARMS:-c0_copy c0_share cr_copy cr_share cr2_copy cr2_share crv_copy crv_share crs_copy crs_share${OLD_BIN:+ c0_old cr_old}}
SARGS=${SARGS:-}
mkdir -p "$OUT"

CR="--cache-ram 4096"
SH="--ctx-checkpoints-share"
declare -A FLAGS=(
    [c0_copy]="--cache-ram 0"            [c0_share]="--cache-ram 0 $SH"
    [cr_copy]="$CR"                       [cr_share]="$CR $SH"
    [cr2_copy]="$CR --ctx-checkpoints 2"  [cr2_share]="$CR --ctx-checkpoints 2 $SH"
    [crv_copy]="$CR --ctx-checkpoints-vram 2"     [crv_share]="$CR --ctx-checkpoints-vram 2 $SH"
    [crs_copy]="$CR --ctx-checkpoints-spread 512" [crs_share]="$CR --ctx-checkpoints-spread 512 $SH"
    [c0_old]="--cache-ram 0"             [cr_old]="$CR"
)

# deterministic token lists: n tokens below the vocabulary size, from a seed (Park-Miller, exact in awk's doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <comma-separated tokens>
    printf '{"prompt":[%s],"n_predict":16,"top_k":1,"temperature":0,"n_probs":3,"return_tokens":true,"cache_prompt":true}' "$3" \
        > "$1/$2.req.json"
    curl -s --max-time 1800 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

reply() { jq -r '.tokens | map(tostring) | join(",")' "$1"; }

kb() { awk -v k="$2" '$1 == k":" { print $2 }' "/proc/$1/status"; } # <pid> <VmRSS|VmHWM|RssAnon>

printf 'arm\treq\tVmRSS_kB\tVmHWM_kB\tRssAnon_kB\n' > "$OUT/mem.tsv"
printf 'arm\treq\tprompt_n\tcache_n\tcache_src\tcache_ckpt_n\tckpt_created_n\ttokens\n' > "$OUT/requests.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    mkdir -p "$d"
    b="$BIN"
    [[ $arm == *_old ]] && b="$OLD_BIN"
    "$b/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" -c "$CTX" -b 512 -ub 128 -np 1 --ctx-checkpoints "$CKPT" \
        --no-webui -lv 4 --host 127.0.0.1 --port "$PORT" $SARGS ${FLAGS[$arm]} > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 600); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        echo "$arm: server did not start"; tail -20 "$d/server.log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')
    printf '%s\tload\t%s\t%s\t%s\n' "$arm" "$(kb $pid VmRSS)" "$(kb $pid VmHWM)" "$(kb $pid RssAnon)" >> "$OUT/mem.tsv"

    run() { # <name> <tokens>
        complete "$d" "$1" "$2"
        printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$1" "$(kb $pid VmRSS)" "$(kb $pid VmHWM)" "$(kb $pid RssAnon)" >> "$OUT/mem.tsv"
    }
    p1=$(toks 3000 1 "$nv");                           run P1 "$p1"
    p2="$p1,$(reply "$d/P1.json"),$(toks 500 2 "$nv")"; run P2 "$p2"
    p3=$(echo "$p2" | awk -F, -v OFS=, -v v="$nv" '{ $2201 = ($2201 % (v - 1)) + 1; print }')
                                                        run P3 "$p3"
                                                        run P4 "$p2"
    p5="$p3,$(reply "$d/P3.json"),$(toks 300 3 "$nv")"; run P5 "$p5"
    p6=$(toks 3000 4 "$nv");                           run P6 "$p6"
    p7="$p5,$(reply "$d/P5.json"),$(toks 200 5 "$nv")"; run P7 "$p7"
    p8="$p6,$(reply "$d/P6.json"),$(toks 200 6 "$nv")"; run P8 "$p8"

    kill "$pid"; wait "$pid" 2>/dev/null
    for r in P1 P2 P3 P4 P5 P6 P7 P8; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.cache_n, .timings.cache_src,
            .timings.cache_ckpt_n, .timings.ckpt_created_n, (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" \
            >> "$OUT/requests.tsv"
    done
    echo "$arm: done, peak $(awk -v a="$arm" '$1 == a { h = $4 } END { print h }' "$OUT/mem.tsv") kB," \
        "$(grep -c 'created context checkpoint' "$d/server.log") checkpoints created," \
        "checkpoint size $(grep -m1 -o 'created context checkpoint.*size = [0-9.]* MiB' "$d/server.log" | grep -o '[0-9.]* MiB$')"
done

# the verdict
fail=0
same() { # <arm a> <arm b>: tokens and top probabilities of every request, exactly
    for r in P1 P2 P3 P4 P5 P6 P7 P8; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$1/$r.json" 2>/dev/null) \
                    <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$2/$r.json" 2>/dev/null) ||
           [ ! -s "$OUT/$1/$r.json" ]; then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    echo "$2 = $1 (tokens and top probabilities of P1-P8)"
}
peak() { awk -v a="$1" '$1 == a { h = $4 } END { print h + 0 }' "$OUT/mem.tsv"; }
{
    for a in $ARMS; do
        case $a in
            *_share) c=${a%_share}_copy ;;
            *_old)   c=${a%_old}_copy ;;
            *)       continue ;;
        esac
        [[ " $ARMS " == *" $c "* ]] || continue
        same "$c" "$a" || fail=1
        echo "  peak: $c $(peak "$c") kB, $a $(peak "$a") kB, $c - $a = $(( $(peak "$c") - $(peak "$a") )) kB"
    done
    for a in $ARMS; do
        e=$(grep -ciE 'error|abort|failed to restore' "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
} | tee "$OUT/verdict.txt"
exit $fail
