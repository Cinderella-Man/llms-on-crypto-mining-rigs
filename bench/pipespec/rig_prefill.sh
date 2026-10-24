#!/usr/bin/env bash
# Prompt chunks next to decoding conversations in llama-pipespec's streams mode on the rig (Q155): how many chunks are in
# flight, whether a prompt's chunks may overlap, and whether the chunks run on streams of their own below the tokens'
# priority. Build: fork branch pascal-p106-v2-q155 (llama_pipe_ready, LLAMA_PIPE_CHUNK_STREAM, job keys chunks and
# chunks_gen).
#
# One process per configuration in CONFIGS (one model load each), named k<K>[g<G>][s]:
#   k<K>   up to K prompt chunks in flight (job key chunks; k2 is the executor before Q155, which also kept at most one
#          chunk per prompt), several of one prompt, prompts in order of arrival
#   g<G>   at most G chunks in flight while any conversation generates (job key chunks_gen; default K)
#   s      LLAMA_PIPE_CHUNK_STREAM=1: the chunks run on a second stream per card whose priority is below the tokens', so
#          a token overtakes the chunks queued before it at kernel boundaries
# Jobs in each process, N conversations with the prompts of PROMPTS cycled (by default two 2,076-token and two 750-token
# prompts among eight, the Q120 mix), the temperature chain sampled on the last card, no guesses:
#   all      every conversation joins at once
#   arrive   conversation c joins after ARRIVE * c generated tokens, so long prompts arrive while others decode
# plus, in the first configuration only, plain references of every conversation (ref-c<c>, prompt in chunks of CHUNK).
# summary.tsv per configuration and job: generated tokens per second over the job (all conversations), the job's time,
# the mean and longest time from a conversation's arrival to its first token, and whether every conversation equals
# its plain reference.
#
# Run it on the rig, from this repository (~20 min for five configurations):
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5,6,7,8 \
#       [TS=6,6,6,6,6,6,6,5,2] [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg'] [CONFIGS='k2 k8 k8g1 k4s k8s'] [N=8] [CTX=8192] \
#       [NTOK=512] [CHUNK=64] [ARRIVE=64] bash /tmp/pipespec/rig_prefill.sh > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# CTX is per conversation. Each chunk slot holds the compute buffers of a CHUNK-token evaluation on every card, so a
# large K can fail to fit next to the model: that configuration's process stops at its start and the next one runs.
# Output in /tmp/pipespec/out/<TAG>/: jobs-<config>.txt, run-<config>.log, results-<config>.txt, gpu.csv, summary.tsv.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
CONFIGS=${CONFIGS:-k2 k8 k8g1 k4s k8s}
N=${N:-8}
CTX=${CTX:-8192}
NTOK=${NTOK:-512}
CHUNK=${CHUNK:-64}
ARRIVE=${ARRIVE:-64}
PROMPTS=${PROMPTS:-$HERE/prompts/chat.txt,$HERE/prompts/code.txt,$HERE/prompts/edit.txt,$HERE/prompts/rewrite.txt}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

IFS=',' read -r -a files <<< "$PROMPTS"
SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42 --pipe-backend-sample"

gpu_pid=""
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
        > "$OUT/gpu.csv" 2>/dev/null &
    gpu_pid=$!
fi

first=1
for cfg in $CONFIGS; do
    if [[ ! $cfg =~ ^k([0-9]+)(g([0-9]+))?(s)?$ ]]; then
        echo "unknown config $cfg"; continue
    fi
    keys="chunks=${BASH_REMATCH[1]}"
    [ -n "${BASH_REMATCH[3]}" ] && keys="$keys chunks_gen=${BASH_REMATCH[3]}"
    envs="LLAMA_PIPE_UPLOAD=2"
    [ -n "${BASH_REMATCH[4]}" ] && envs="$envs LLAMA_PIPE_CHUNK_STREAM=1"
    common="n=$NTOK chat=1 chunk=$CHUNK"
    jobs=$OUT/jobs-$cfg.txt
    {
        echo "name=warm mode=plain prompt=${files[0]} chat=1 n=64"
        if [ $first = 1 ]; then
            for ((c = 0; c < N; c++)); do
                echo "name=ref-c$c mode=plain prompt=${files[c % ${#files[@]}]} seed=$((42 + c)) seq=$c $common"
            done
        fi
        echo "name=all streams=$N prompts=$PROMPTS seed=42 depth=0 guess=none $keys $common"
        echo "name=arrive streams=$N prompts=$PROMPTS seed=42 depth=0 guess=none arrive=$ARRIVE $keys $common"
    } > "$jobs"

    echo "$(date -Is) start $TAG $cfg: $keys, $envs, GPUs $GPUS"
    # shellcheck disable=SC2086
    env ${ENVS:-} $envs CUDA_VISIBLE_DEVICES="$GPUS" timeout 3600 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on \
        -c $((N * CTX)) -np "$N" ${TS:+-ts $TS} $SAMPLING --pipe-jobs "$jobs" > "$OUT/run-$cfg.log" 2>&1
    echo "$(date -Is) exit $? $TAG $cfg"
    grep -o 'PIPESPEC.*' "$OUT/run-$cfg.log" | sed "s/^/cfg=$cfg /" > "$OUT/results-$cfg.txt"
    grep -q "name=ref-c" "$OUT/results-$cfg.txt" && first=0
done
[ -n "$gpu_pid" ] && kill "$gpu_pid" 2>/dev/null

for cfg in $CONFIGS; do cat "$OUT/results-$cfg.txt" 2>/dev/null; done | awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
{ cfg = f("cfg"); nm = f("name") }
$2 == "PIPESPEC" && nm ~ /^ref-c/ { c = nm; sub(/^ref-c/, "", c); rt[c] = f("tokens") }
$2 == "PIPESPEC" && f("mode") == "streams" {
    job = nm; sub(/-c[0-9]+$/, "", job); c = nm; sub(/^.*-c/, "", c); key = cfg SUBSEP job
    tok[key, c] = f("tokens"); nconv[key]++
    tf = f("t_first") + 0; sum_first[key] += tf; if (tf > max_first[key]) max_first[key] = tf
}
$2 == "PIPESPEC_STREAMS" { key = cfg SUBSEP nm; order[++no] = key; tps[key] = f("total_tok_s"); tall[key] = f("t_all") }
END {
    print "config\tjob\ttok_s\tt_all_s\tfirst_token_mean_s\tfirst_token_max_s\tsame_as_plain"
    for (i = 1; i <= no; i++) {
        key = order[i]; split(key, kk, SUBSEP); bad = 0; miss = 0
        for (c = 0; c < nconv[key]; c++) {
            if (!(c in rt)) miss++
            else if (tok[key, c] != rt[c]) bad++
        }
        same = miss ? "no reference" : (bad ? "NO (" bad ")" : "yes")
        printf "%s\t%s\t%s\t%s\t%.2f\t%.2f\t%s\n", kk[1], kk[2], tps[key], tall[key], \
            nconv[key] ? sum_first[key] / nconv[key] : 0, max_first[key], same
    }
}' > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
