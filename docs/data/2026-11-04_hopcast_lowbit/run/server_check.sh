#!/usr/bin/env bash
# C408 (Q963) on the CPU, from C200's server check: llama-server's speculative decoding with the card-hop switches,
# against the build below the new types (-kshape). A random-weight qwen4exp model (Flash-Next's architecture, from
# test-llama-archs) drafts for itself (--spec-type draft-simple), so every verify pass is 1 + 0..3 tokens wide, and the
# prompts of 40-90 tokens run in micro-batches of 32 tokens. The same requests go to one server per arm:
#   base       BASE_BIN (the build before the new types, -kshape)
#   off        BIN with no switch set: must give base's tokens and top probabilities exactly
#   staged     BIN with GGML_CUDA_HOP_STAGED=1 (a CUDA path; on the CPU nothing changes): exactly base's
#   q8auto     BIN with LLAMA_HOP_CAST=q8_0 and no LLAMA_HOP_CAST_AT: on the CPU there is no boundary between two GPUs,
#              so no hop and exactly base's
#   q8at1      BIN with LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1 (the round trip before layer 1): runs, logs the hop,
#              and may differ from base (how much is in summary.tsv; test-hop-cast measures it)
#   f16at1     the same with LLAMA_HOP_CAST=f16
#   q6at1, q5at1, q4at1, nlat1  the same with LLAMA_HOP_CAST=q6_k, q5_0, q4_0, iq4_nl (the types below 8 bits)
#   q4auto     LLAMA_HOP_CAST=q4_0 without LLAMA_HOP_CAST_AT: no boundary on the CPU, exactly base's
# Requests (token prompts, 64-96 tokens generated, seed 42): R1 greedy, n_max 3; R2 temperature 1.0, n_max 3; R3
# greedy, n_max 1; R4 temperature 1.0, n_max 2, p_min 0.3.
# Pass: off, staged, q8auto and q4auto give base's tokens and top-3 probabilities for every request; every
# *at1 arm logs (-lv 5) "card hops in <type> before layers 1" and complete every request; no log has an error line.
# Usage: BASE_BIN=<dir> BIN=<dir> MODEL=<gguf> OUT=<dir> [PORT=18091] [THREADS=8] [ARMS='base off ...'] bash server_check.sh
# Needs curl and jq. Output: OUT/<arm>/{server.log,R1..R4.json}, OUT/summary.tsv, OUT/verdict.txt
set -uo pipefail
BASE_BIN=${BASE_BIN:?BASE_BIN=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with llama-server>}
MODEL=${MODEL:?MODEL=<gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18091}
THREADS=${THREADS:-8}
ARMS=${ARMS:-base off staged q8auto q4auto q8at1 f16at1 q6at1 q5at1 q4at1 nlat1}
mkdir -p "$OUT"

declare -A ENVS=(
    [base]="" [off]="" [staged]="GGML_CUDA_HOP_STAGED=1" [q8auto]="LLAMA_HOP_CAST=q8_0"
    [q8at1]="LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1" [f16at1]="LLAMA_HOP_CAST=f16 LLAMA_HOP_CAST_AT=1"
    [q4auto]="LLAMA_HOP_CAST=q4_0" [q6at1]="LLAMA_HOP_CAST=q6_k LLAMA_HOP_CAST_AT=1" [q5at1]="LLAMA_HOP_CAST=q5_0 LLAMA_HOP_CAST_AT=1"
    [q4at1]="LLAMA_HOP_CAST=q4_0 LLAMA_HOP_CAST_AT=1" [nlat1]="LLAMA_HOP_CAST=iq4_nl LLAMA_HOP_CAST_AT=1"
)
# the type as the log names it (ggml_type_name)
declare -A HOPTYPE=([q8at1]=q8_0 [f16at1]=f16 [q6at1]=q6_K [q5at1]=q5_0 [q4at1]=q4_0 [nlat1]=iq4_nl)

# deterministic token lists: n tokens below the vocabulary size, from a seed (Park-Miller, exact in awk's doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <arm dir> <name> <tokens> <n_predict> <temperature> <n_max> <p_min>
    printf '{"prompt":[%s],"n_predict":%d,"temperature":%s,"top_k":%s,"seed":42,"n_probs":3,"return_tokens":true,"cache_prompt":false,"speculative.n_max":%d,"speculative.n_min":0,"speculative.p_min":%s}' \
        "$3" "$4" "$5" "$([ "$5" = 0 ] && echo 1 || echo 40)" "$6" "$7" > "$1/$2.req.json"
    curl -s --max-time 900 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

errors() { sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E ' E |[Ee]rror|[Aa]bort' | grep -vc 'set_abort_callback'; }

printf 'arm\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\ttokens\n' > "$OUT/summary.tsv"

for arm in $ARMS; do
    d="$OUT/$arm"
    mkdir -p "$d"
    b=$BIN; [ "$arm" = base ] && b=$BASE_BIN
    # shellcheck disable=SC2086
    env ${ENVS[$arm]} "$b/llama-server" -m "$MODEL" -md "$MODEL" --spec-type draft-simple --spec-draft-n-max 3 \
        -ngl 0 -ngld 0 -t "$THREADS" -c 4096 -b 512 -ub 32 -np 1 --ctx-checkpoints 4 --cache-ram 0 --no-webui -lv 5 \
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
    complete "$d" R1 "$(toks 40 11 "$nv")" 96 0   3 0.0
    complete "$d" R2 "$(toks 60 12 "$nv")" 96 1.0 3 0.0
    complete "$d" R3 "$(toks 90 13 "$nv")" 64 0   1 0.0
    complete "$d" R4 "$(toks 70 14 "$nv")" 96 1.0 2 0.3
    kill "$pid"; wait "$pid" 2>/dev/null
    for r in R1 R2 R3 R4; do
        jq -r --arg a "$arm" --arg r "$r" '[$a, $r, .timings.prompt_n, .timings.predicted_n, .timings.draft_n,
            .timings.draft_n_accepted, (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
    done
    echo "$arm: done, $(errors "$d/server.log") error lines in its log"
done

fail=0
same() { # <arm a> <arm b>: tokens and top probabilities of every request, exactly
    for r in R1 R2 R3 R4; do
        if ! cmp -s <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$1/$r.json") \
                    <(jq -c '[.tokens, .completion_probabilities]' "$OUT/$2/$r.json"); then
            echo "$2 differs from $1 at $r"; return 1
        fi
    done
    echo "$2 = $1 (tokens and top probabilities of R1-R4)"
}
{
    for a in $ARMS; do
        case $a in
            base) ;;
            off|staged|q8auto|q4auto) same base "$a" || fail=1 ;;
            *)
                n=$(grep -c "card hops in ${HOPTYPE[$a]} before layers 1 " "$OUT/$a/server.log")
                k=$(jq -s '[.[] | .tokens | length] | add' "$OUT/$a"/R?.json)
                echo "$a: $n contexts logged the hop before layer 1, $k tokens generated; $(same base "$a" 2>&1 | tail -n 1)"
                [ "$n" -ge 1 ] && [ "${k:-0}" -gt 0 ] || fail=1
                ;;
        esac
    done
    nd=$(awk -F'\t' '$1 == "base" { s += $5 } END { print s + 0 }' "$OUT/summary.tsv")
    echo "base: $nd tokens drafted"
    [ "$nd" -gt 0 ] || { echo "base did not draft"; fail=1; }
    for a in $ARMS; do
        e=$(errors "$OUT/$a/server.log")
        [ "$e" = 0 ] || { echo "$a: $e error lines in its log"; fail=1; }
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
} | tee "$OUT/verdict.txt"
grep -qx PASS "$OUT/verdict.txt"
