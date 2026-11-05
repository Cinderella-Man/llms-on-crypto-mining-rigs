#!/usr/bin/env bash
# C536 (Q1025): llama-server's sliding-window checkpoint switches on the CPU, one server per arm, with a random-weight
# sliding-window model (test-llama-archs -o, e.g. gemma3, its window 1/8 of LLAMA_ARCHS_N_CTX). The same five greedy
# requests go to every arm, so that each checkpoint rule is hit: P1 = 3000 tokens (its checkpoint C1 near the end);
# P2 = P1 + its reply + 500 tokens; P3 = P1 + its reply + 500 other tokens (diverges just after C1: C1 is restored, and
# the prompt resumes after or at its last position, LLAMA_SERVER_CKPT_RESUME_NEXT); P4 = P2 with the token at 2900
# changed (diverges inside C1's window, before its end: only a ring checkpoint past the divergence serves it,
# LLAMA_SERVER_CKPT_RING and _RING_PAST); P5 = P2 again.
# Arms come from $OUT/arms.txt, one per line: <name> <bin dir with llama-server> [VAR=value ...]. The arm "default" is
# the reference: "base" (another build at its defaults) must give its tokens, an arm whose name ends in "noinit0" its
# tokens and restore points; every other arm is reported: tokens, restore points (pos_min, pos_max, n_past) and the
# checkpoints' sizes against the default's.
# Usage: MODEL=<gguf> OUT=<dir> [PORT=18090] [THREADS=8] [SARGS='<more server arguments>'] bash server_arms.sh
# Needs curl, jq and gawk. Output: OUT/<arm>/{server.log,P1..P5.json,tokens.txt,restores.txt,sizes.txt},
# OUT/verdicts.tsv (check, verdict, detail)
set -uo pipefail
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18090}
THREADS=${THREADS:-8}
ARCH=$(basename "$MODEL" | sed 's/-\(dense\|moe\)\.gguf$//')
SARGS=${SARGS:---override-kv $ARCH.context_length=int:16384}
: > "$OUT/verdicts.tsv"

toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <comma-separated tokens>
    printf '{"prompt":[%s],"n_predict":16,"top_k":1,"temperature":0,"n_probs":3,"return_tokens":true,"cache_prompt":true}' "$3" \
        > "$1/$2.req.json"
    curl -s --max-time 900 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

reply() { jq -r '.tokens | map(tostring) | join(",")' "$1"; }

while read -r arm bin envs; do
    [ -n "$arm" ] || continue
    d="$OUT/$arm"
    mkdir -p "$d"
    # shellcheck disable=SC2086
    env $envs "$bin/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 8 \
        --cache-ram 0 --no-webui -lv 4 --host 127.0.0.1 --port "$PORT" $SARGS > "$d/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
        sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
        printf 'server %s\tFAIL\tthe server did not start (%s)\n' "$arm" "$(tail -n 1 "$d/server.log")" >> "$OUT/verdicts.tsv"
        kill "$pid" 2>/dev/null
        continue
    fi
    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')
    p1=$(toks 3000 1 "$nv")
    complete "$d" P1 "$p1"
    p2="$p1,$(reply "$d/P1.json"),$(toks 500 2 "$nv")"
    complete "$d" P2 "$p2"
    p3="$p1,$(reply "$d/P1.json"),$(toks 500 4 "$nv")"
    complete "$d" P3 "$p3"
    p4=$(echo "$p2" | awk -F, -v OFS=, -v v="$nv" '{ $2901 = ($2901 % (v - 1)) + 1; print }')
    complete "$d" P4 "$p4"
    complete "$d" P5 "$p2"
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    for r in P1 P2 P3 P4 P5; do
        printf '%s %s\n' "$r" "$(reply "$d/$r.json")"
    done > "$d/tokens.txt"
    for r in P1 P2 P3 P4 P5; do
        printf '%s %s\n' "$r" "$(jq -r '"cached \(.timings.cache_n) read \(.timings.prompt_n)"' "$d/$r.json")"
    done > "$d/cache.txt"
    grep -o 'restored context checkpoint (pos_min = [0-9]*, pos_max = [0-9]*, n_tokens = [0-9]*, n_past = [0-9]*' \
        "$d/server.log" | sed 's/restored context checkpoint (//' > "$d/restores.txt"
    grep -o 'created context checkpoint .*size = [0-9.]* MiB' "$d/server.log" | grep -o 'size = [0-9.]* MiB' \
        | sort | uniq -c | awk '{ printf "%s x %s %s; ", $1, $4, $5 }' > "$d/sizes.txt"
    grep -ci 'error\|abort' "$d/server.log" > "$d/errors.txt"
done < "$OUT/arms.txt"

R=$OUT/default
while read -r arm _; do
    [ "$arm" = default ] && continue
    d="$OUT/$arm"
    [ -s "$d/tokens.txt" ] || continue
    same_t=$(cmp -s "$R/tokens.txt" "$d/tokens.txt" && echo yes || echo no)
    same_r=$(cmp -s "$R/restores.txt" "$d/restores.txt" && echo yes || echo no)
    same_s=$(cmp -s "$R/sizes.txt" "$d/sizes.txt" && echo yes || echo no)
    same_c=$(cmp -s "$R/cache.txt" "$d/cache.txt" && echo yes || echo no)
    detail="tokens as default: $same_t; cached/read: $(tr '\n' ';' < "$d/cache.txt"); restores ($(wc -l < "$d/restores.txt")): $(tr '\n' ';' < "$d/restores.txt" | cut -c1-240); checkpoints: $(cat "$d/sizes.txt"); error lines $(cat "$d/errors.txt")"
    case $arm in
        base|*noinit0)
            v=$([ "$same_t$same_r$same_c" = yesyesyes ] && echo OK || echo FAIL) ;;
        *)
            v=$([ "$same_r$same_s$same_c" = yesyesyes ] && echo "NO EFFECT" || echo ACTS) ;;
    esac
    printf 'server %s\t%s\t%s\n' "$arm" "$v" "$detail" >> "$OUT/verdicts.tsv"
done < "$OUT/arms.txt"
printf 'server default\t%s\tcached/read: %s; restores (%s): %s; checkpoints: %s; error lines %s\n' \
    "$([ -s "$R/tokens.txt" ] && echo OK || echo FAIL)" "$(tr '\n' ';' < "$R/cache.txt")" "$(wc -l < "$R/restores.txt")" \
    "$(tr '\n' ';' < "$R/restores.txt" | cut -c1-240)" "$(cat "$R/sizes.txt")" "$(cat "$R/errors.txt")" >> "$OUT/verdicts.tsv"
