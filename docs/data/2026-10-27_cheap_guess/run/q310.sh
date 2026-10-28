#!/usr/bin/env bash
# Q310: Flint's MTP head on a faster multi-GPU setup. Flint Q8_0 on three RTX 3090s (compute capability 8.6) as a layer
# split with Miner Llama's pipelined executor, the agent check's recorded traces replayed (bench/agent_check/replay.py:
# the recorded prompts, a fixed seed per request, the recorded reply lengths) to the release line's flags without and
# with the head, paired as on the rig (run/replay.sh): the same tokens in every arm (run/pair_check.py), so the ratio of
# ms per generated token is the head's effect. One llama-server per arm in a container on the GPU machine; the replay
# client runs on that machine too. Run from a machine that reaches it over ssh.
# Env: HOST (ssh alias of the GPU machine), WORK (folder there), BIN (bin folder of the sm_86 build there: build.sh
# CUDA_ARCH=86), MODEL (Flint Q8_0 there; MODEL_DIRS: folders to mount so that it resolves, symlinks included),
# HEAD_GEN, HEAD_NINE (head GGUFs there; an arm is skipped if its head is unset), REPLAY_LOG (a recorded arm's
# proxy.jsonl there), OUT (local folder); optional GPUS (default 2,5,6), TS (17,16,16: Flint's 49 layers), CTX
# (262144), DEPTH (--pipe-depth, 3: the 3-card window of 2 rows plus one), TRACES (default the six of replay.sh),
# ARMS (base gen nine), PORT (18510), COMMON_ENV (GGML_CUDA_MMVQ_MAX_COLS=4, as on the rig), DRAFT_ENV
# (LLAMA_DRAFT_HEAD=q8_0), DOCKER_ARGS (e.g. a label), IMAGE (nvidia/cuda:12.9.2-devel-ubuntu24.04).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
H=${HOST:?ssh alias of the GPU machine}
W=${WORK:?folder on the GPU machine}/q310
: "${BIN:?}" "${MODEL:?}" "${REPLAY_LOG:?}" "${OUT:?}"
GPUS=${GPUS:-2,5,6}
TS=${TS:-17,16,16}
CTX=${CTX:-262144}
DEPTH=${DEPTH:-3}
PORT=${PORT:-18510}
ARMS=${ARMS:-base gen nine}
COMMON_ENV=${COMMON_ENV-GGML_CUDA_MMVQ_MAX_COLS=4}
DRAFT_ENV=${DRAFT_ENV-LLAMA_DRAFT_HEAD=q8_0}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
TRACES=${TRACES:-rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1}
FLAGS="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 \
--top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui -b 512 -ub 128 -ts $TS --pipe-depth $DEPTH"
mkdir -p "$OUT"
# Quiet machine first, as for a benchmark on the rig: no other container, load average under 1 (waits up to 10 min),
# no compute process and no utilization on any GPU, no quantize/build/KLD process, and RIG_STATUS (a command on HOST
# that shows the machine's own presets, read-only) logged. Logged to OUT/preflight.txt; anything else running stops it.
preflight() {
    ssh "$H" "for i in \$(seq 20); do l=\$(cut -d' ' -f1 /proc/loadavg); awk -v l=\$l 'BEGIN{exit !(l < 1)}' && break; sleep 30; done
        echo \"time: \$(date -Is)\"; echo \"loadavg: \$(cat /proc/loadavg)\"
        echo 'containers:'; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
        echo 'gpus:'; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
        echo 'compute processes:'; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader
        echo 'heavy processes:'; ps -eo pid,pcpu,comm,args | grep -E 'llama-|quantize|perplexity|cmake|nvcc|make |python.*(train|score|kld)' | grep -v grep
        ${RIG_STATUS:+echo 'presets:'; $RIG_STATUS}" > "$OUT/preflight.txt" 2>&1
    awk '/^loadavg:/ {if ($2 >= 1) bad = "load " $2}
         /^containers:/ {s = "c"; next} /^gpus:/ {s = "g"; next} /^compute processes:/ {s = "p"; next}
         /^heavy processes:/ {s = "h"; next} /^presets:/ {s = ""; next}
         s == "c" && NF && $1 !~ /^q310-/ {bad = bad " container " $1}
         s == "g" && NF {split($0, f, ", "); if (f[3] + 0 > 0) bad = bad " gpu " f[1] " busy"}
         s == "p" && NF {bad = bad " compute process " $0}
         s == "h" && NF {bad = bad " process " $3}
         END {if (bad != "") {print "preflight: " bad; exit 1}}' "$OUT/preflight.txt"
}
if [ "${PREFLIGHT:-1}" = 1 ]; then
    preflight || { echo "=== not started: the machine is not quiet (OUT/preflight.txt)"; exit 3; }
    echo "=== preflight passed $(date +%T)" | tee "$OUT/run.log"
fi
ssh "$H" "mkdir -p $W && rm -f $W/replay.jsonl"
tar cf - -C "$REPO/bench" --exclude=__pycache__ agent_check | ssh "$H" "tar xf - -C $W"
cat "$REPO/docs/data/2026-10-27_cheap_guess/run/pair_check.py" | ssh "$H" "cat > $W/pair_check.py"
mounts="-v $(dirname "$BIN"):$(dirname "$BIN"):ro -v $(dirname "$MODEL"):$(dirname "$MODEL"):ro"
for d in ${MODEL_DIRS:-}; do mounts="$mounts -v $d:$d:ro"; done
for hd in ${HEAD_GEN:-} ${HEAD_NINE:-}; do mounts="$mounts -v $(dirname "$hd"):$(dirname "$hd"):ro"; done
echo "=== Q310 on GPUs $GPUS: -ts $TS -c $CTX --pipe-depth $DEPTH; arms $ARMS $(date +%T)" | tee -a "$OUT/run.log"
for a in $ARMS; do
    case $a in
        base) extra=""; env="$COMMON_ENV" ;;
        gen)  [ -n "${HEAD_GEN:-}" ] || continue; extra="--pipe-guess staircase --pipe-draft $HEAD_GEN"; env="$COMMON_ENV $DRAFT_ENV" ;;
        nine) [ -n "${HEAD_NINE:-}" ] || continue; extra="--pipe-guess staircase --pipe-draft $HEAD_NINE"; env="$COMMON_ENV $DRAFT_ENV" ;;
        *) echo "unknown arm $a"; continue ;;
    esac
    envs=""
    for e in $env; do envs="$envs -e $e"; done
    echo "=== $a: $env llama-server $FLAGS $extra $(date +%T)" | tee -a "$OUT/run.log"
    ssh "$H" "docker rm -f q310-$a > /dev/null 2>&1; docker run -d --name q310-$a ${DOCKER_ARGS:-} --gpus '\"device=$GPUS\"' \
        --network host $mounts $envs $IMAGE $BIN/llama-server $FLAGS $extra -m $MODEL --host 127.0.0.1 --port $PORT" > /dev/null
    up=0
    for _ in $(seq 240); do
        sleep 5
        ssh "$H" "curl -sf 127.0.0.1:$PORT/health > /dev/null" && { up=1; break; }
        ssh "$H" "docker ps -q -f name=q310-$a | grep -q ." || break
    done
    if [ $up = 1 ]; then
        echo "$a: server up $(date +%T)" | tee -a "$OUT/run.log"
        ssh "$H" "cd $W/agent_check && python3 -I replay.py run --log $REPLAY_LOG --traces $TRACES \
            --server http://127.0.0.1:$PORT --label $a --out $W/replay.jsonl 2>&1 | tail -2" | tee -a "$OUT/run.log"
    else
        echo "$a: server did not start" | tee -a "$OUT/run.log"
    fi
    ssh "$H" "docker logs q310-$a > $W/server-$a.log 2>&1; docker rm -f q310-$a > /dev/null"
    ssh "$H" "grep -E 'staircase drafter|pipelined speculation|out of memory|error' $W/server-$a.log | cut -c1-300 | head -6" \
        | tee -a "$OUT/run.log"
done
ssh "$H" "cd $W/agent_check && python3 -I replay.py summary $W/replay.jsonl --log $REPLAY_LOG --reference base > $W/replay.md; \
    cat $W/replay.md; for a in gen nine; do python3 -I $W/pair_check.py $W/replay.jsonl --a base --b \$a | tail -1; done" \
    | tee "$OUT/replay.md"
for f in replay.jsonl server-base.log server-gen.log server-nine.log; do
    ssh "$H" "cat $W/$f 2> /dev/null" > "$OUT/$f"
done
echo "=== done $(date +%T)" | tee -a "$OUT/run.log"
