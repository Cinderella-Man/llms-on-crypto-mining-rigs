#!/usr/bin/env bash
# C770 (Q1355): llama-server's switch line at its default verbosity, on the CPU. BASE is the base branch's llama-server,
# BIN the branch's; both run every arm with the same model and MTP draft (as C771's server_check.sh,
# docs/data/2026-11-11_c771_draft_kpool_split: -m MODEL -md DRAFT -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4
# --cache-ram 0 -fit off --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64),
# arguments and environment, at the server's default verbosity (no -lv). Each arm's environment holds only PATH, HOME
# and the arm's variables (env -i), so the line lists exactly those.
# Arms:
#   none      no LLAMA_ or GGML_ variable
#   several   five that the branch reads: LLAMA_SAMPLING_TOP_K_ROWS=1 (C691), LLAMA_GPU_KQ_MASK=1, GGML_CUDA_KEEPALIVE=decode,
#             GGML_CUDA_MMVQ_PASCAL_SKIP= (empty), LLAMA_ARG_NO_UI=1 (the negative form of LLAMA_ARG_UI)
#   unknown   the five, plus three that no source of the branch reads: LLAMA_SAMPLNG_TOP_K_ROWS=1 (a typo),
#             LLAMA_C770_NOTE='two words' and LLAMA_C770_TOKEN=c770-secret-value (a name with TOKEN: its value hidden)
# Requests (token prompts, the top 3 probabilities of each token): G1 = 100 tokens, 64 greedy tokens; S1 = the same
# prompt, 64 tokens at temperature 1.0 (top-k 20, top-p 0.95, min-p 0, seed 42).
# Pass: in every arm BIN's tokens, top probabilities and draft counts (draft_n, draft_n_accepted) = BASE's on both
# requests; line_check.py's verdict on every arm's two logs (BIN's log has one switch line with each variable in its
# list, the hidden value nowhere; BASE's none); no log has an error line.
# Usage: BASE=<dir with the base llama-server> BIN=<dir with the branch's llama-server> MODEL=<gguf with an MTP block>
#   DRAFT=<its MTP-only draft gguf> OUT=<dir> [PORT=18770] [THREADS=8] [ARMS='none several unknown']
#   [SARGS='<more server arguments>'] bash server_check.sh
# A random test model trains on 256 tokens, and llama-server caps a slot's context there: give it a longer one, e.g.
#   SARGS='--override-kv qwen4exp.context_length=int:16384'
# Needs curl, jq and python3. Output: OUT/<arm>-{base,tip}/{server.log,G1.json,S1.json}, OUT/summary.tsv,
#   OUT/lines.tsv, OUT/verdict.txt
set -uo pipefail
BASE=${BASE:?BASE=<dir with the base llama-server>}
BIN=${BIN:?BIN=<dir with the branch llama-server>}
MODEL=${MODEL:?MODEL=<gguf with an MTP block>}
DRAFT=${DRAFT:?DRAFT=<its MTP-only draft gguf>}
OUT=${OUT:?OUT=<dir>}
PORT=${PORT:-18770}
THREADS=${THREADS:-8}
ARMS=${ARMS:-none several unknown}
SARGS=${SARGS:-}
RUN=$(dirname "$(realpath "$0")")
REQS="G1 S1"
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 64"
mkdir -p "$OUT"

SEVERAL=(LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_GPU_KQ_MASK=1 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_MMVQ_PASCAL_SKIP= LLAMA_ARG_NO_UI=1)
UNREAD=(LLAMA_SAMPLNG_TOP_K_ROWS=1 "LLAMA_C770_NOTE=two words" LLAMA_C770_TOKEN=c770-secret-value)
arm_env() { # <arm>: the arm's variables, one per line, each with "read" or "unread" (what the branch's line should say)
    case $1 in
        several) printf '%s\tread\n' "${SEVERAL[@]}" ;;
        unknown) printf '%s\tread\n' "${SEVERAL[@]}"; printf '%s\tunread\n' "${UNREAD[@]}" ;;
    esac
}

# deterministic token lists: n tokens below the vocabulary size, from a seed (a Park-Miller generator, exact in awk's
# doubles)
toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }

complete() { # <run dir> <name> <comma-separated tokens> <n_predict> <temperature> <seed>
    printf '{"prompt":[%s],"n_predict":%s,"temperature":%s,"top_k":20,"top_p":0.95,"min_p":0,"seed":%s,"n_probs":3,"return_tokens":true,"cache_prompt":false}' \
        "$3" "$4" "$5" "$6" > "$1/$2.req.json"
    curl -s --max-time 900 -H 'Content-Type: application/json' --data @"$1/$2.req.json" \
        "http://127.0.0.1:$PORT/completion" > "$1/$2.json"
    rm -f "$1/$2.req.json"
}

printf 'arm\tbuild\treq\tprompt_n\tpredicted_n\tdraft_n\tdraft_n_accepted\ttokens\n' > "$OUT/summary.tsv"
for arm in $ARMS; do
    for build in base tip; do
        d="$OUT/$arm-$build"
        rm -rf "$d" && mkdir -p "$d"
        exe=$BASE
        [ "$build" = tip ] && exe=$BIN
        arm_env "$arm" > "$d/env.tsv"
        vars=()
        while IFS=$'\t' read -r v _; do vars+=("$v"); done < "$d/env.tsv"
        # shellcheck disable=SC2086
        env -i PATH="$PATH" HOME="${HOME:-/root}" "${vars[@]}" \
            "$exe/llama-server" -m "$MODEL" -md "$DRAFT" -t "$THREADS" -c 8192 -b 512 -ub 128 -np 1 --ctx-checkpoints 4 \
            --cache-ram 0 -fit off --no-webui --host 127.0.0.1 --port "$PORT" $MTP $SARGS > "$d/server.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 180); do
            curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if ! curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
            echo "$arm-$build: server did not start"; tail -n 20 "$d/server.log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; continue
        fi
        nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')
        p1=$(toks 100 11 "$nv")
        complete "$d" G1 "$p1" 64 0 1
        complete "$d" S1 "$p1" 64 1.0 42
        kill "$pid"
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        for r in $REQS; do
            jq -r --arg a "$arm" --arg b "$build" --arg r "$r" '[$a, $b, $r, .timings.prompt_n, .timings.predicted_n,
                (.timings.draft_n // 0), (.timings.draft_n_accepted // 0),
                (.tokens | map(tostring) | join(" "))] | @tsv' "$d/$r.json" >> "$OUT/summary.tsv"
        done
        echo "$arm-$build: done, $(grep -ac 'switches:' "$d/server.log") switch lines, $(grep -c -iE 'error|abort' "$d/server.log") error lines in its log"
    done
done

# the verdict
fail=0
{
    for arm in $ARMS; do
        for r in $REQS; do
            a="$OUT/$arm-base/$r.json" b="$OUT/$arm-tip/$r.json"
            if [ -s "$a" ] && [ -s "$b" ] && cmp -s <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$a") \
                                                  <(jq -c '[.tokens, .completion_probabilities, .timings.draft_n, .timings.draft_n_accepted]' "$b"); then
                echo "$arm $r: tip = base ($(jq '.tokens | length' "$b") tokens, their top probabilities, $(jq '.timings.draft_n_accepted' "$b") of $(jq '.timings.draft_n' "$b") drafts accepted)"
            else
                echo "$arm $r: tip differs from base"; fail=1
            fi
        done
    done
    largs=
    for arm in $ARMS; do
        largs="$largs $arm:$OUT/$arm-tip/env.tsv:$OUT/$arm-base/server.log:$OUT/$arm-tip/server.log"
    done
    # shellcheck disable=SC2086
    python3 -I "$RUN/line_check.py" --tsv "$OUT/lines.tsv" $largs || fail=1
    for arm in $ARMS; do
        for build in base tip; do
            f="$OUT/$arm-$build/server.log"
            [ -f "$f" ] || continue
            e=$(grep -ciE 'error|abort' "$f")
            [ "$e" = 0 ] || { echo "$arm-$build: $e error lines in its log"; fail=1; }
        done
    done
    [ $fail = 0 ] && echo "PASS" || echo "FAIL"
    exit $fail
} | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
