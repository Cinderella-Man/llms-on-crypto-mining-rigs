#!/usr/bin/env bash
# Times several conversations staggered through the cards (llama-pipespec's streams mode, Q104) and checks each
# conversation against its own plain run. One process, one model load:
#   - plain references: one plain job per conversation of the largest N (its prompt, seed + c, the same prompt chunks);
#   - for each N in NS: N conversations at once (prompts cycled from PROMPTS), no guesses;
#   - at N = 2 and N = 4: the same with guesses in the slack (copy >= 4, else continuation counts, depth 2, a budget of
#     BUDGET items in flight), and at the largest N with conversations joining one after another (ARRIVE generated tokens
#     apart), so that prompt chunks run between other conversations' tokens.
# Sampling runs on the last card, one chain per conversation (temperature 1, top-k 20, top-p 0.95; each draw keyed by
# position), unless SAMPLE=host or SAMPLE=greedy.
# summary.tsv: per streams job the total generated tokens per second, the slowest and median conversation, and whether
# every conversation matches its plain run.
#
# Run it on the rig, from this repository:
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5,6,7,8 \
#       [TS=6,6,6,6,6,6,6,5,2] [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg'] [NS='1 2 3 4 6 8'] [CTX=8192] [NTOK=512] \
#       [BUDGET=8] bash /tmp/pipespec/rig_streams.sh > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# CTX is per conversation (the context holds max(NS) streams of CTX tokens). Output in /tmp/pipespec/out/<TAG>/:
# jobs.txt, run.log, results.txt, summary.tsv, gpu.csv, host.csv (the process's CPU use each second).
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
NS=${NS:-1 2 3 4 6 8}
CTX=${CTX:-8192}
NTOK=${NTOK:-512}
CHUNK=${CHUNK:-64}
BUDGET=${BUDGET:-8}
ARRIVE=${ARRIVE:-64}
SAMPLE=${SAMPLE:-device}
PROMPTS=${PROMPTS:-$HERE/prompts/chat.txt,$HERE/prompts/code.txt,$HERE/prompts/edit.txt,$HERE/prompts/rewrite.txt}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

nmax=1
for n in $NS; do [ "$n" -gt "$nmax" ] && nmax=$n; done
case $SAMPLE in
    device) SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42 --pipe-backend-sample" ;;
    greedy) SAMPLING="--temp 0 --pipe-backend-greedy" ;;
    *)      SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42" ;;
esac
IFS=',' read -r -a files <<< "$PROMPTS"

jobs=$OUT/jobs.txt
common="n=$NTOK chat=1 chunk=$CHUNK"
{
    echo "name=warm mode=plain prompt=${files[0]} chat=1 n=64"
    for ((c = 0; c < nmax; c++)); do
        echo "name=ref-c$c mode=plain prompt=${files[c % ${#files[@]}]} seed=$((42 + c)) seq=$c $common"
    done
    for n in $NS; do
        echo "name=n$n streams=$n prompts=$PROMPTS seed=42 depth=0 guess=none $common"
        if [ "$n" = 2 ] || [ "$n" = 4 ]; then
            echo "name=n$n-guess streams=$n prompts=$PROMPTS seed=42 depth=2 guess=cont copy_min=4 budget=$BUDGET $common"
        fi
    done
    echo "name=n$nmax-arrive streams=$nmax prompts=$PROMPTS seed=42 depth=0 guess=none arrive=$ARRIVE $common"
} > "$jobs"

gpu_pid=""
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
        > "$OUT/gpu.csv" 2>/dev/null &
    gpu_pid=$!
fi

echo "$(date -Is) start $TAG: $(wc -l < "$jobs") jobs, GPUs $GPUS"
# shellcheck disable=SC2086
env ${ENVS:-} CUDA_VISIBLE_DEVICES="$GPUS" timeout 10800 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on \
    -c $((nmax * CTX)) -np "$nmax" ${TS:+-ts $TS} $SAMPLING --pipe-jobs "$jobs" > "$OUT/run.log" 2>&1 &
pid=$!
# the process's CPU use each second (utime + stime from /proc, in % of one core)
(
    hz=$(getconf CLK_TCK); prev=""
    while kill -0 "$pid" 2>/dev/null; do
        t=$(awk '{print $14 + $15}' "/proc/$pid/stat" 2>/dev/null) || break
        [ -n "$prev" ] && echo "$(date +%s) $(( (t - prev) * 100 / hz ))"
        prev=$t; sleep 1
    done
) > "$OUT/host.csv" &
wait "$pid"
echo "$(date -Is) exit $? $TAG"
[ -n "$gpu_pid" ] && kill "$gpu_pid" 2>/dev/null
grep -o 'PIPESPEC.*' "$OUT/run.log" > "$OUT/results.txt"

awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
$1 == "PIPESPEC" && f("name") ~ /^ref-c/ { c = f("name"); sub(/^ref-c/, "", c); rt[c] = f("tokens"); rl[c] = f("logits"); rs[c] = f("state"); next }
$1 == "PIPESPEC" && f("mode") == "streams" {
    nm = f("name"); job = nm; sub(/-c[0-9]+$/, "", job); c = nm; sub(/^.*-c/, "", c)
    rate[job, ++k[job]] = f("tok_s")
    if (f("tokens") != rt[c] || f("logits") != rl[c] || f("state") != rs[c]) bad[job]++
    next
}
$1 == "PIPESPEC_STREAMS" { j = f("name"); order[++no] = j; tot[j] = f("total_tok_s"); nconv[j] = f("n_conv") }
END {
    print "job\tconversations\ttotal_tok_s\tslowest_tok_s\tmedian_tok_s\tsame_as_plain"
    for (i = 1; i <= no; i++) {
        j = order[i]; n = k[j]
        for (a = 1; a <= n; a++) v[a] = rate[j, a] + 0
        for (a = 1; a <= n; a++) for (b = a + 1; b <= n; b++) if (v[b] < v[a]) { t = v[a]; v[a] = v[b]; v[b] = t }
        med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
        print j "\t" nconv[j] "\t" tot[j] "\t" v[1] "\t" med "\t" (bad[j] ? "NO (" bad[j] ")" : "yes")
    }
}' "$OUT/results.txt" > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
