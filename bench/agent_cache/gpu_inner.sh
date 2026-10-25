#!/usr/bin/env bash
# gpu_check.sh's phases inside its container (/ac = WORK): servers one after the other on port 18421, the clients next
# to them. MODEL and MODE (check, replay, all) come from the environment; /ac/gpu-check/env, if present, is sourced
# after it (MODEL, MODE, CHECK_CTX, REPLAY_CTX, CHECK_RESTART, REPLAY_ARMS, SERVER_BIN, FORCE_PASCAL), so a job already
# queued on the lock can be redirected.
# The model and both slots must fit one card: a 32 GB V100 holds a 23 GB Flint file with these contexts.
set -uo pipefail
CHECK_CTX=16384      # two slots of 8k: the check scenarios stay below ~6k tokens
REPLAY_CTX=32768     # one slot: the longest replayed session (~26.5k tokens recorded)
CHECK_RESTART=0      # 1: the prefix again from its file after a restart (a third model load)
REPLAY_ARMS="stock agent-tail2 agent"  # agent-tail2 splits prompts as stock does, so its replies must equal stock's
SERVER_BIN=/ac/out-cuda/build/bin/llama-server
FORCE_PASCAL=1       # 0: the GPU's own kernels instead of the Pascal ones (the server logic is the same)
[ -f /ac/gpu-check/env ] && . /ac/gpu-check/env
[ "$FORCE_PASCAL" = 1 ] || unset GGML_CUDA_MMVQ_PASCAL GGML_CUDA_PERSISTENT_MOE LLAMA_INPUT_ON_GPU LLAMA_INPUT_UPLOAD
PORT=18421
U=http://127.0.0.1:$PORT
G=/ac/gpu-check
COMMON="-m $MODEL -ngl 99 -fa on -b 2048 -ub 512 --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --host 127.0.0.1 --port $PORT"
apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y python3 curl > /dev/null 2>&1
cd /ac/bench/agent_cache

start() { # log env args
    local log=$1 env=$2 args=$3
    # shellcheck disable=SC2086
    env $env "$SERVER_BIN" $COMMON $args > "$G/$log.log" 2>&1 &
    SRV=$!
    for _ in $(seq 900); do curl -sf $U/health > /dev/null && return 0; kill -0 $SRV 2>/dev/null || break; sleep 2; done
    echo "$log: server did not start"; tail -20 "$G/$log.log"; return 1
}
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }

status=0
if [ "$MODE" = check ] || [ "$MODE" = all ]; then
    rm -rf $G/pfx; mkdir -p $G/pfx
    AC="-c $CHECK_CTX -np 2 --prefix-cache $G/pfx --checkpoint-gen"
    CHK="python3 check_exact.py --test $U --cold $U --test-slot 0 --cold-slot 1 --sessions /ac/sessions-e2-ud.jsonl"
    start check1 "" "$AC" && { $CHK --scenarios prefix,turns,trim --expect-restore --expect-trim > $G/check1.txt 2>&1; stop; }
    if [ "$CHECK_RESTART" = 1 ]; then
        start check2 "" "$AC" && { $CHK --scenarios prefix_again --expect-restore > $G/check2.txt 2>&1; stop; }
    fi
    grep -hE "^(PASS|FAIL|ALL)" $G/check1.txt $G/check2.txt 2>/dev/null
fi
if [ "$MODE" = replay ] || [ "$MODE" = all ]; then
    rm -f $G/replay.jsonl
    RUN="python3 replay_sessions.py run $G/sessions.jsonl --server $U --out $G/replay.jsonl"
    for arm in $REPLAY_ARMS; do
        case $arm in
            stock)       env="LLAMA_SAMPLER_RESERVE=1"; args="--checkpoint-tail2";;
            agent)       env=""; args="--prefix-cache $G/pfx-$arm --checkpoint-gen";;
            agent-tail2) env=""; args="--prefix-cache $G/pfx-$arm --checkpoint-gen --checkpoint-tail2";;
            *) echo "unknown arm $arm"; continue;;
        esac
        rm -rf $G/pfx-$arm; mkdir -p $G/pfx-$arm
        start replay-$arm "$env" "-c $REPLAY_CTX -np 1 -bs $args" && { $RUN --label $arm > $G/replay-$arm.txt 2>&1 || status=1; stop; }
    done
    python3 replay_sessions.py summary $G/replay.jsonl --reference stock | tee $G/replay-summary.md
    for arm in $REPLAY_ARMS; do
        f=$G/replay-$arm.log
        echo "$arm: $(grep -c 'sched_reserve: reserving' $f) reserves, $(grep -c 'restored context checkpoint' $f) checkpoint restores, $(grep -c 'forcing full prompt' $f) full re-reads, $(grep -c 'prefix cache: restored' $f) prefix restores, $(grep -c 'CUDA error' $f) CUDA errors"
    done
fi
exit $status
