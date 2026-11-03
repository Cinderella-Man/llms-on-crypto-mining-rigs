#!/usr/bin/env bash
# The bytes-per-token switches in llama-server on a real model: one server load per arm, four coding prompts, each
# greedy, with the server's default sampling (top-k 40, top-p 0.95, min-p 0.05, temperature 0.8) and with the agent
# check's (the makers': top-k 20, top-p 0.95, temperature 1), each sampled from a fixed seed. The text must equal
# the "off" arm's for "head" (LLAMA_HEAD_BOUND) and for "tail:p=1"; the server log ends with the head bound's counts
# (rows kept, full-head fallbacks, bound against the top logit) and the experts kept (llama's info lines, which the
# server shows at -lv 4).
# The same script runs on a CPU (SERVER_ARGS with -t) and on the rig (SERVER_ARGS: the release line's layout).
# Env: BIN (bin/ with llama-server), MODEL, OUT, ARMS (default "off head tail:p=1 tail:p=0.9,renorm=1"; "head" uses
#      HEAD, default 32768,8192, and FREQ as LLAMA_HEAD_BOUND_FILE if set), SERVER_ARGS (default "-c 4096 -np 1"),
#      PORT (18191), NPRED (64), UP_S (seconds to wait for a load, default 1800)
# Writes $OUT/<arm>.server.log, $OUT/<arm>.p<i>.<mode>.json and $OUT/summary.tsv. Needs curl and jq.
set -uo pipefail

: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
ARMS=${ARMS:-"off head tail:p=1 tail:p=0.9,renorm=1"}
HEAD=${HEAD:-32768,8192}
SERVER_ARGS=${SERVER_ARGS:-"-c 4096 -np 1"}
PORT=${PORT:-18191}
NPRED=${NPRED:-64}
UP_S=${UP_S:-1800}
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
'<|im_start|>user
List the files a typical Rust crate with one binary and one library has, one per line, with a few words on each.<|im_end|>
<|im_start|>assistant
<think>

</think>

'
'<|im_start|>user
Translate this shell pipeline into Python with subprocess and explain each stage: `grep -rn TODO src | sort | uniq -c | sort -rn | head`.<|im_end|>
<|im_start|>assistant
<think>

</think>

'
)
modes=(greedy sampled agentic)
mode_json() {
    case $1 in
        greedy)  echo '{"temperature": 0, "top_k": 1}' ;;
        sampled) echo '{"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}' ;;
        agentic) echo '{"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0}' ;;
    esac
}

[ -f "$OUT/summary.tsv" ] || printf 'arm\tswitch\tprompt\tmode\tn_predicted\tgen_tok_s\tsame_text_as_off\n' > "$OUT/summary.tsv"
for arm in $ARMS; do
    envs=""
    case $arm in
        off)    ;;
        head)   envs="LLAMA_HEAD_BOUND=$HEAD${FREQ:+ LLAMA_HEAD_BOUND_FILE=$FREQ}" ;;
        tail:*) envs="LLAMA_MOE_TAIL=${arm#tail:}" ;;
        *)      echo "unknown arm $arm"; continue ;;
    esac
    name=${arm//[:=,]/_}
    echo "== $arm: $envs"
    # shellcheck disable=SC2086
    env $envs "$BIN/llama-server" -m "$MODEL" $SERVER_ARGS -lv 4 --port "$PORT" --no-webui > "$OUT/$name.server.log" 2>&1 &
    pid=$!
    t0=$(date +%s)
    for _ in $(seq 1 "$UP_S"); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
        kill -0 $pid 2> /dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "$arm: the server did not come up (see $name.server.log)"
        kill $pid 2> /dev/null; wait $pid 2> /dev/null
        continue
    fi
    echo "$arm: up after $(( $(date +%s) - t0 )) s"
    for i in "${!PROMPTS[@]}"; do
        for mode in "${modes[@]}"; do
            f=$OUT/$name.p$i.$mode.json
            jq -n --arg p "${PROMPTS[$i]}" --argjson n "$NPRED" --argjson s "$(mode_json "$mode")" \
                '{prompt: $p, n_predict: $n, cache_prompt: false, seed: 7} + $s' |
                curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- > "$f"
            same=-
            if [ "$arm" != off ] && [ -f "$OUT/off.p$i.$mode.json" ]; then
                [ "$(jq -r .content "$f")" = "$(jq -r .content "$OUT/off.p$i.$mode.json")" ] && same=yes || same=NO
            fi
            jq -r --arg arm "$arm" --arg sw "$envs" --arg i "$i" --arg m "$mode" --arg same "$same" \
                '[$arm, $sw, $i, $m, .timings.predicted_n, (.timings.predicted_per_second * 100 | round / 100), $same] | @tsv' \
                "$f" >> "$OUT/summary.tsv"
        done
    done
    kill $pid 2> /dev/null
    wait $pid 2> /dev/null
    # what the server logged at the end: the head bound's counts, the experts kept
    grep -E 'head bound:|moe tail' "$OUT/$name.server.log" | tail -n 4
done

if command -v column > /dev/null; then column -t -s $'\t' "$OUT/summary.tsv"; else cat "$OUT/summary.tsv"; fi
