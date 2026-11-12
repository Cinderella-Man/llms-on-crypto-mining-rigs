#!/usr/bin/env bash
# C803 (Q1407): the MTP draft's re-plans on a CPU. In a container (no network, low CPU share): test-llama-archs writes
# its tiny qwen4exp model with one MTP block and that block alone as a separate draft (as Flash-Next's draft file), at
# a 65,536-token context; then llama-server (BIN) answers FN84's requests (run/draft_trace.py) in each arm of ARMS with
# LLAMA_UBATCH_TRACE=1, and draft_trace.py parse tables every draft evaluation and every re-plan.
# Usage: WORK=<dir> BIN=<dir with llama-server and test-llama-archs, a CPU build> [ARMS='<name>@<switches,...>@<extra
#   server args, ~ for spaces>' ...] [THREADS=4] [PORT=18803] [IMAGE=pq-c565-cpu:24.04] [DOCKER_ARGS=...] bash trace.sh
# Output: $WORK/models, $WORK/<arm>.server.log, $WORK/<arm>.replies.json, $WORK/trace.tsv, $WORK/trace_summary.txt
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BIN=$(realpath "${BIN:?BIN=<CPU build bin dir>}")
RUN=$(dirname "$(realpath "$0")")
THREADS=${THREADS:-4}
PORT=${PORT:-18803}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
# FN84's line on a CPU: R60's server arguments without the cards' (-ts, -otd, --no-host, --ctx-checkpoints-vram) and
# the CUDA switches; the k-pool and indexer switches as on the rig; the shortlist scaled to the tiny vocabulary
BASE_ARGS="-ngl 0 -c 65536 -b 2048 -ub 512 -np 1 --no-webui -fa on -bs -lv 5 --spec-type draft-mtp --spec-draft-n-max 3 \
--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints 4 --ctx-checkpoints-incall --ubatch-taper on"
ARMS=${ARMS:-r60line@LLAMA_QSA_INDEXER_ROWS=1,LLAMA_QSA_KPOOL_DEV=1,LLAMA_DRAFT_SHORTLIST=100,8,2@--spec-draft-n-ubatch~512 \
r85line@@--spec-draft-n-ubatch~128}
mkdir -p "$WORK"
# shellcheck disable=SC2086
docker run --rm --cpu-shares=256 --network none --label pascal-quants=1 --label c803=1 ${DOCKER_ARGS:-} \
    -v "$WORK:$WORK" -v "$BIN:$BIN:ro" -v "$RUN:$RUN:ro" "$IMAGE" bash -c '
    set -u
    W='"$WORK"' B='"$BIN"' R='"$RUN"' PORT='"$PORT"' TH='"$THREADS"'
    mkdir -p $W/models
    [ -f $W/models/qwen4exp-moe-mtp-draft.gguf ] || LLAMA_ARCHS_N_CTX=65536 nice -n 19 $B/test-llama-archs \
        -a "^qwen4exp$" -s 1 -o $W/models --mtp-draft > $W/models/save.log 2>&1
    ls -l $W/models
    M=$W/models/qwen4exp-moe-mtp.gguf D=$W/models/qwen4exp-moe-mtp-draft.gguf
    specs=""
    for a in '"$ARMS"'; do
        name=${a%%@*} rest=${a#*@} sw=${rest%%@*} extra=${rest#*@}
        envs=$(echo "$sw" | sed "s/,LLAMA_/ LLAMA_/g")
        nice -n 19 python3 -I $R/draft_trace.py arm $name $W $PORT $B/llama-server $M $envs -- \
            '"$BASE_ARGS"' -t $TH -md $D ${extra//\~/ }
        specs="$specs $name=$W/$name.server.log"
    done
    nice -n 19 python3 -I $R/draft_trace.py parse $specs > $W/trace.tsv 2> $W/trace_summary.txt
    cat $W/trace_summary.txt'
