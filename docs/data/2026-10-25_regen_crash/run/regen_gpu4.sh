#!/usr/bin/env bash
# The regenerate crash (2026-10-25) reproduced away from the rig, on one CUDA GPU (a V100 here): --device CUDA0,CUDA0
# makes two llama devices on one card, so the scheduler runs with pipeline parallelism (4 copies of every split input),
# and LLAMA_INPUT_UPLOAD sets the packed uploads that compute 6.1 turned on by default in the release candidates.
# Runs bench/pipespec/rig_regenerate.sh (the server under gdb) with granite-3.1-3b-a800m-instruct Q4_K_M. Correctness only.
# Usage: MODEL=<granite .gguf> WORK=<dir> [GPU=n] [LOCK=file] regen_gpu4.sh <bin dir> <CUDA libs dir or -> <tag> <upload 0|2> <runs>
#   <bin dir>: a CUDA build's bin/ (for example rc4 2d53179aa against its parent 460160e93); <runs>: rig_regenerate.sh's RUNS
set -uo pipefail
BIN=$(realpath "$1"); LIBS=$2; TAG=$3; UP=$4; RUNS=$5
W=${WORK:?WORK=<work dir>}
MODEL=$(realpath "${MODEL:?MODEL=<granite .gguf>}")
K=$W/regen-kit
mkdir -p $K $W/cuda-jit-cache
cp -r "$(cd "$(dirname "$0")/../../../.." && pwd)/bench/pipespec/." $K/
[ "$LIBS" = - ] || LIBS=$(realpath "$LIBS")
${LOCK:+flock -w 3600 $LOCK} docker run --rm --name regen-$TAG --cpu-shares=256 \
    --gpus "\"device=${GPU:-0}\"" -e CUDA_CACHE_PATH=/w/cuda-jit-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
    -e LLAMA_INPUT_UPLOAD=$UP -v "$W:/w" -v "$BIN:/bin-under-test:ro" -v "$MODEL:/m/model.gguf:ro" \
    $([ "$LIBS" = - ] || echo "-v $LIBS:/libs:ro") \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 1500 bash -c "
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /dev/null && apt-get install -y -qq python3 curl gdb procps > /dev/null
    [ $LIBS = - ] || export LD_LIBRARY_PATH=/libs:\${LD_LIBRARY_PATH:-}
    BIN=/bin-under-test MODEL=/m/model.gguf TAG=$TAG GPUS=0 RUNS='$RUNS' \
        ARGS='-ngl 99 -fa on -c 8192 -ub 512 -lv 4 --device CUDA0,CUDA0' bash /w/regen-kit/rig_regenerate.sh
    grep -m2 -E 'pipeline parallelism|input layer' /w/regen-kit/out/$TAG/*.log
"
echo "done $TAG: exit $?"
