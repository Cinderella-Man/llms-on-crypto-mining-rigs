#!/usr/bin/env bash
# The draft-memory switches on the real files, on a CPU (no GPU needed): one llama-server load per arm, two greedy
# requests each. Drafts never change a greedy output, so every arm must return the same text as the arm without them;
# what may differ is the acceptance and the draft's buffers.
#
# Env: BIN (llama.cpp bin/ with llama-server), MODEL (target, first shard), DRAFT (the MTP draft), OUT (results),
#      THREADS (default 16), PORT (default 18181), NPRED (default 96), ARMS (default: all below), N_MAX (default 3),
#      SHORT (LLAMA_DRAFT_SHORTLIST, default 32768,8192,2), BODIES (an agent check's proxy.jsonl.bodies.gz: the nodraft
#      arm then writes token_freq.tsv with run/token_freq.sh)
# Arms: nodraft, base (draft, no switch), embd (LLAMA_DRAFT_SHARE=embd), share (embd,head), short (LLAMA_DRAFT_SHORTLIST
#       on the draft's own head), shareshort (both), shortfile (both, with token_freq.tsv as the frequency list),
#       dio (-lm dio, LLAMA_DIO_STAGING=2x16, the switches off)
# Needs curl and jq. Writes $OUT/<arm>.server.log, $OUT/<arm>.p<i>.json and $OUT/summary.tsv.
set -uo pipefail

: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}"
THREADS=${THREADS:-16}
PORT=${PORT:-18181}
NPRED=${NPRED:-96}
N_MAX=${N_MAX:-3}
ARMS=${ARMS:-"nodraft base embd share short shareshort shortfile dio"}
SHORT=${SHORT:-32768,8192,2}
mkdir -p "$OUT"

PROMPTS=(
'<|im_start|>user
Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals and returns them sorted. Add a short docstring and three asserts.<|im_end|>
<|im_start|>assistant
<think>

</think>

'
'<|im_start|>user
The file src/server.c fails to compile with "error: implicit declaration of function strndup". Explain the cause in two sentences, then give the one-line fix.<|im_end|>
<|im_start|>assistant
<think>

</think>

'
)

arm_env() {
    case $1 in
        nodraft|base) echo "" ;;
        embd)         echo "LLAMA_DRAFT_SHARE=embd" ;;
        share)        echo "LLAMA_DRAFT_SHARE=embd,head" ;;
        short)        echo "LLAMA_DRAFT_SHORTLIST=$SHORT" ;;
        shareshort)   echo "LLAMA_DRAFT_SHARE=embd,head LLAMA_DRAFT_SHORTLIST=$SHORT" ;;
        shortfile)    echo "LLAMA_DRAFT_SHARE=embd,head LLAMA_DRAFT_SHORTLIST=$SHORT LLAMA_DRAFT_SHORTLIST_FILE=$OUT/token_freq.tsv" ;;
        dio)          echo "LLAMA_DIO_STAGING=2x16" ;;
    esac
}

arm_args() {
    local a=(-m "$MODEL" -c 4096 -np 1 -t "$THREADS" -tb "$THREADS" --port "$PORT" --no-webui --metrics)
    case $1 in
        nodraft) ;;
        dio)     a+=(-md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max "$N_MAX" -lm dio) ;;
        *)       a+=(-md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max "$N_MAX") ;;
    esac
    printf '%s\n' "${a[@]}"
}

printf 'arm\tprompt\tn_predicted\tdraft_n\tdraft_accepted\tacceptance\tgen_tok_s\tsame_text_as_nodraft\n' > "$OUT/summary.tsv"

for arm in $ARMS; do
    mapfile -t args < <(arm_args "$arm")
    echo "== $arm: $(arm_env "$arm") ${args[*]}"
    # shellcheck disable=SC2046
    env $(arm_env "$arm") "$BIN/llama-server" "${args[@]}" > "$OUT/$arm.server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 1800); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
        kill -0 $pid 2> /dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "$arm: the server did not come up (see $arm.server.log)"
        kill $pid 2> /dev/null; wait $pid 2> /dev/null
        continue
    fi
    for i in "${!PROMPTS[@]}"; do
        jq -n --arg p "${PROMPTS[$i]}" --argjson n "$NPRED" \
            '{prompt: $p, n_predict: $n, temperature: 0, top_k: 1, cache_prompt: false, seed: 1}' |
            curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- > "$OUT/$arm.p$i.json"
        same=-
        if [ "$arm" != nodraft ] && [ -f "$OUT/nodraft.p$i.json" ]; then
            [ "$(jq -r .content "$OUT/$arm.p$i.json")" = "$(jq -r .content "$OUT/nodraft.p$i.json")" ] && same=yes || same=NO
        fi
        jq -r --arg arm "$arm" --arg i "$i" --arg same "$same" '[$arm, $i, .timings.predicted_n, (.timings.draft_n // 0),
            (.timings.draft_n_accepted // 0),
            (if (.timings.draft_n // 0) > 0 then ((.timings.draft_n_accepted / .timings.draft_n) * 1000 | round / 1000) else "-" end),
            (.timings.predicted_per_second * 100 | round / 100), $same] | @tsv' "$OUT/$arm.p$i.json" >> "$OUT/summary.tsv"
    done
    # the frequency list for the shortfile arm, from an agent check's logged requests (BODIES), tokenized by this server
    if [ "$arm" = nodraft ] && [ -n "${BODIES:-}" ]; then
        BODIES=$BODIES URL=http://127.0.0.1:$PORT OUT=$OUT/token_freq.tsv bash "$(dirname "$0")/token_freq.sh"
    fi
    kill $pid 2> /dev/null
    wait $pid 2> /dev/null
done

if command -v column > /dev/null; then column -t -s $'\t' "$OUT/summary.tsv"; else cat "$OUT/summary.tsv"; fi
