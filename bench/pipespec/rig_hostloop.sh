#!/usr/bin/env bash
# Times the host's share of a token in the pipelined executor on the rig: packed asynchronous input uploads (Q120) and
# tokens fed from card to card (Q121). One process per configuration (one model load each), the temperature chain
# (top-k 20, top-p 0.95, seed 42) sampled on the last card:
#   u0   LLAMA_PIPE_UPLOAD=0: one synchronous copy per host input, as llama_decode does (the executor before Q120)
#   u2   LLAMA_PIPE_UPLOAD=2: each card's host inputs in one asynchronous copy from pinned staging memory
#   u2f  u2 with LLAMA_INPUT_ON_GPU=1 (the embedding lookup on card 1), plus the fed jobs (llama_pipe_submit_fed: the next
#        token is submitted before the current one is sampled and its id goes from the last card to card 1)
# Jobs in each process: plain (llama_decode, the reference), d0 (one token in flight, its successor submitted after it
# is sampled), d0-fed (u2f), and for each N in NS: N conversations staggered through the cards without guesses (nN),
# fed (nN-fed, u2f); plain references for every conversation (ref-c<c>).
# summary.tsv: per configuration and job the generated tokens per second (all conversations for streams), the host time
# per generated token inside the submit calls (submit_us), inside the waits for results (wait_us) and the rest
# (rest_us), and whether the output equals its plain run. Host busy time per token = submit_us + rest_us.
#
# Run it on the rig, from this repository:
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5,6,7,8 \
#       [TS=6,6,6,6,6,6,6,5,2] [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1'] [NS='1 4 8'] [CTX=8192] [NTOK=512] \
#       [CONFIGS='u0 u2 u2f'] bash /tmp/pipespec/rig_hostloop.sh > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# CTX is per conversation. Output in /tmp/pipespec/out/<TAG>/: jobs-<config>.txt, run-<config>.log, results-<config>.txt,
# host-<config>.csv (the process's CPU use each second), gpu.csv, summary.tsv, and traces of the single-stream pipe jobs
# (<config>-d0.trace, u2f-d0-fed.trace; traces.py gives the median time from one sampled token to the next).
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
NS=${NS:-1 4 8}
CTX=${CTX:-8192}
NTOK=${NTOK:-512}
CHUNK=${CHUNK:-64}
CONFIGS=${CONFIGS:-u0 u2 u2f}
PROMPTS=${PROMPTS:-$HERE/prompts/chat.txt,$HERE/prompts/code.txt,$HERE/prompts/edit.txt,$HERE/prompts/rewrite.txt}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

nmax=1
for n in $NS; do [ "$n" -gt "$nmax" ] && nmax=$n; done
IFS=',' read -r -a files <<< "$PROMPTS"
SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42 --pipe-backend-sample"

gpu_pid=""
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
        > "$OUT/gpu.csv" 2>/dev/null &
    gpu_pid=$!
fi

for cfg in $CONFIGS; do
    case $cfg in
        u0)  envs="LLAMA_PIPE_UPLOAD=0" ;;
        u2)  envs="LLAMA_PIPE_UPLOAD=2" ;;
        u2f) envs="LLAMA_PIPE_UPLOAD=2 LLAMA_INPUT_ON_GPU=1" ;;
        *)   echo "unknown config $cfg"; continue ;;
    esac
    common="n=$NTOK chat=1 chunk=$CHUNK"
    jobs=$OUT/jobs-$cfg.txt
    {
        echo "name=warm mode=plain prompt=${files[0]} chat=1 n=64"
        echo "name=plain mode=plain prompt=${files[0]} $common"
        echo "name=d0 mode=pipe depth=0 guess=none prompt=${files[0]} $common trace=$OUT/$cfg-d0.trace"
        [ "$cfg" = u2f ] && echo "name=d0-fed mode=pipe depth=0 guess=none feed=1 prompt=${files[0]} $common trace=$OUT/$cfg-d0-fed.trace"
        for ((c = 0; c < nmax; c++)); do
            echo "name=ref-c$c mode=plain prompt=${files[c % ${#files[@]}]} seed=$((42 + c)) seq=$c $common"
        done
        for n in $NS; do
            echo "name=n$n streams=$n prompts=$PROMPTS seed=42 depth=0 guess=none $common"
            [ "$cfg" = u2f ] && echo "name=n$n-fed streams=$n prompts=$PROMPTS seed=42 depth=0 guess=none feed=1 $common"
        done
    } > "$jobs"

    echo "$(date -Is) start $TAG $cfg: $(wc -l < "$jobs") jobs, GPUs $GPUS, $envs"
    # shellcheck disable=SC2086
    env ${ENVS:-} $envs CUDA_VISIBLE_DEVICES="$GPUS" timeout 5400 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on \
        -c $((nmax * CTX)) -np "$nmax" ${TS:+-ts $TS} $SAMPLING --pipe-jobs "$jobs" > "$OUT/run-$cfg.log" 2>&1 &
    pid=$!
    # the process's CPU use each second (utime + stime from /proc, in % of one core)
    (
        hz=$(getconf CLK_TCK); prev=""
        while kill -0 "$pid" 2>/dev/null; do
            t=$(awk '{print $14 + $15}' "/proc/$pid/stat" 2>/dev/null) || break
            [ -n "$prev" ] && echo "$(date +%s) $(( (t - prev) * 100 / hz ))"
            prev=$t; sleep 1
        done
    ) > "$OUT/host-$cfg.csv" &
    wait "$pid"
    echo "$(date -Is) exit $? $TAG $cfg"
    grep -o 'PIPESPEC.*' "$OUT/run-$cfg.log" | sed "s/^/cfg=$cfg /" > "$OUT/results-$cfg.txt"
done
[ -n "$gpu_pid" ] && kill "$gpu_pid" 2>/dev/null

for cfg in $CONFIGS; do cat "$OUT/results-$cfg.txt" 2>/dev/null; done | awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
{ cfg = f("cfg"); nm = f("name") }
$2 == "PIPESPEC" && nm == "plain" { pt[cfg] = f("tokens") }
$2 == "PIPESPEC" && nm ~ /^ref-c/ { c = nm; sub(/^ref-c/, "", c); rt[cfg, c] = f("tokens") }
$2 == "PIPESPEC" && (nm == "d0" || nm == "d0-fed") {
    key = cfg SUBSEP nm; order[++no] = key; tps[key] = f("tok_s"); ngen[key] = f("n_gen")
    sub_us[key] = f("host_submit_us"); wait_us[key] = f("host_wait_us"); rest_us[key] = f("host_rest_us"); tok[key] = f("tokens")
}
$2 == "PIPESPEC" && nm == "plain" { key = cfg SUBSEP nm; order[++no] = key; tps[key] = f("tok_s"); ngen[key] = f("n_gen") }
$2 == "PIPESPEC" && f("mode") == "streams" {
    job = nm; sub(/-c[0-9]+$/, "", job); c = nm; sub(/^.*-c/, "", c); key = cfg SUBSEP job
    if (f("tokens") != rt[cfg, c]) bad[key]++
}
$2 == "PIPESPEC_STREAMS" {
    key = cfg SUBSEP nm; order[++no] = key; tps[key] = f("total_tok_s"); ngen[key] = f("total_tokens"); isstr[key] = 1
    sub_us[key] = f("host_submit_us"); wait_us[key] = f("host_wait_us"); rest_us[key] = f("host_rest_us")
}
END {
    print "config\tjob\tn_gen\ttok_s\tsubmit_us\twait_us\trest_us\tsame_as_plain"
    for (i = 1; i <= no; i++) {
        key = order[i]; split(key, kk, SUBSEP); cfg = kk[1]; job = kk[2]
        if (job == "plain") same = "-"
        else if (isstr[key]) same = bad[key] ? "NO (" bad[key] ")" : "yes"
        else same = (tok[key] == pt[cfg]) ? "yes" : "NO"
        print cfg "\t" job "\t" ngen[key] "\t" tps[key] "\t" (job == "plain" ? "-" : sub_us[key]) "\t" \
            (job == "plain" ? "-" : wait_us[key]) "\t" (job == "plain" ? "-" : rest_us[key]) "\t" same
    }
}' > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
