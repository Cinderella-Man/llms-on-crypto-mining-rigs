#!/usr/bin/env bash
# run_tails.sh: the MTP draft's acceptance by context depth (registry Q1172): three runs of bench/deep_lane one after
# another, on a machine with newer cards where no speed is read (QUIET=0, every request valid=0). Tail 1 runs the arms
# draft and nodraft, tails 2 and 3 the draft arm alone. A tail is the recorded turn every prompt ends in
# (deep_client.py, --tail-rank K: the turn with the K-th longest recorded reply), so every depth gets the draft's
# counters at three generation points. The server runs in a container per load, the driver on the host.
#   WORK=<work folder> REPO=<checkout of this repository> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     TEXT=<recorded>/proxy.jsonl.bodies.gz GPUS=<docker device list> MOUNTS="<folders the container reads>" \
#     setsid -f nohup nice -n 19 bash run_tails.sh > console.txt 2>&1 < /dev/null
# Env: TAG (^t1swift/), TAILS ("1 2 3"), NAME (the container's name, deep-lane), IMAGE, DOCKER_ARGS, PORT (8097).
# Writes WORK/tail<k>/ (deep_lane.sh's OUT), the kernels the driver compiles to WORK/jit, and WORK/chain.done or
# WORK/chain.failed. Prints CHAIN_DONE or CHAIN_FAILED as its last line.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${BIN:?BIN=<llama-server>}"
: "${MODEL:?MODEL=<first shard>}" "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${TEXT:?TEXT=<recorded agent text>}"
: "${GPUS:?GPUS=<docker device list, e.g. 0,1,2>}"
TAG=${TAG:-^t1swift/}
TAILS=${TAILS:-1 2 3}
NAME=${NAME:-deep-lane}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PORT=${PORT:-8097}
mounts=''
for m in ${MOUNTS:-$WORK}; do mounts="$mounts -v $m:$m"; done
mkdir -p "$WORK/jit"
rm -f "$WORK/chain.done" "$WORK/chain.failed"
echo $$ > "$WORK/chain.pid"

for k in $TAILS; do
  arms=draft
  [ "$k" = 1 ] && arms="draft nodraft"
  echo "CHAIN tail $k ($arms) $(date -Is)"
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT=$TEXT TAG=$TAG OUT=$WORK/tail$k TAIL_RANK=$k PORT=$PORT \
    QUIET=0 WARM=0 REPS=1 SAMPLER_PASS=1 ARMS="$arms" ARM_DRAFT_nodraft=off \
    ARGS_DROP="-ts -otd" ENV_DROP=GGML_CUDA_KEEPALIVE ENV_ADD=CUDA_CACHE_PATH=$WORK/jit \
    RUNNER="docker run --rm --init --name $NAME ${DOCKER_ARGS:-} --network host --gpus \"device=$GPUS\"$mounts $IMAGE" \
    STOP_CMD="docker rm -f $NAME" \
    bash "$REPO/bench/deep_lane/deep_lane.sh"
  if [ ! -e "$WORK/tail$k/done" ]; then
    touch "$WORK/chain.failed"
    rm -f "$WORK/chain.pid"
    echo "CHAIN_FAILED at tail $k $(date -Is)"
    exit 1
  fi
done
touch "$WORK/chain.done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
