#!/usr/bin/env bash
# Q92 (2026-10-24): llama-server with backend sampling (-bs), the screened head requested (LLAMA_SCREEN_HEAD=1) against
# not requested, same requests (the 8 prompts in prompts.txt, each greedy, sampled with a seed, and sampled with n_probs,
# which needs the full head); every response must be byte-identical. Runs inside the CUDA container (installs curl and
# python3); <out dir> must be mounted too:
#   docker run --rm --gpus "\"device=$GPUS\"" -e LLAMA_NO_MOE_DOWN_RESCALE=1 -e BIN=$BIN -v $PWD:$PWD -v $MODELS:$MODELS \
#     -v $BIN:$BIN nvidia/cuda:12.9.2-devel-ubuntu24.04 bash $PWD/quants/gptq_head/screen_server_ab.sh <model> <out dir>
# Env: BIN (bin/ of a build of the screened-head branch), PROMPTS (default prompts.txt next to this script).
set -euo pipefail
M=$1; OUT=$2; B=$BIN; PROMPTS=${PROMPTS:-$(dirname "$0")/prompts.txt}
mkdir -p $OUT
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3 >/dev/null; }
for scr in 0 1; do
    LLAMA_SCREEN_HEAD=$scr $B/llama-server -m $M -ngl 99 -fa on -c 8192 -np 1 -bs --port 8092 --host 127.0.0.1 \
        > $OUT/server-$scr.log 2>&1 &
    pid=$!
    until curl -sf http://127.0.0.1:8092/health > /dev/null; do sleep 2; kill -0 $pid; done
    i=0
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        i=$((i+1))
        for mode in greedy sampled probs; do
            case $mode in
                greedy)  s='"temperature": 0';;
                sampled) s='"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0, "seed": 1234';;
                probs)   s='"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0, "seed": 1234, "n_probs": 3';;
            esac
            body=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$(printf '%b' "$p")")
            curl -s http://127.0.0.1:8092/completion -H 'Content-Type: application/json' \
                -d "{\"prompt\": $body, \"n_predict\": 128, \"cache_prompt\": false, $s}" \
                | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"content": d["content"], "tokens_predicted": d.get("tokens_predicted"), "probs": d.get("completion_probabilities")}, ensure_ascii=False))' \
                > $OUT/resp-$scr-$mode-$i.json
        done
    done < "$PROMPTS"
    kill $pid; wait $pid || true
done
grep -h -i "screened output head" $OUT/server-*.log | grep -v "set_screen_head" || true
# the server logs every change of its decision: with the screen requested it is on for greedy and sampled requests
# and goes off for the n_probs requests
for scr in 0 1; do
    echo "server $scr: screen decisions: on $(grep -c 'srv.*screened output head on' $OUT/server-$scr.log), off $(grep -c 'srv.*screened output head off' $OUT/server-$scr.log)"
done
n=0; d=0
for f in $OUT/resp-0-*.json; do
    g=${f/resp-0-/resp-1-}; n=$((n+1))
    if ! cmp -s $f $g; then d=$((d+1)); echo "DIFFERS: $(basename $f)"; fi
done
echo "responses compared: $n, different: $d"
[ $d = 0 ] && echo "SERVER AB PASSED" || echo "SERVER AB FAILED"
