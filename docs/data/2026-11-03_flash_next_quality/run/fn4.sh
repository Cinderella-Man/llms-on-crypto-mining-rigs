#!/usr/bin/env bash
# FN4: an importance matrix from the reference on publishable agent text (CALIB: agentic sessions outside the held-out
# ones the agent-text KL divergence is scored on, made by ../../2026-10-28_ornith35_kld/run/calib_text.py), 200 chunks
# of 512 tokens on all seven GPUs with the bases' split, llama-imatrix of the v3 base. Besides the importance of every
# weight column, the file counts how many tokens each expert of each layer saw (Q232's expert usage).
#   (env: the chain's) bash fn4.sh   ->  $WORK/fn4/imatrix-swift15-q8_0-agent-train.gguf
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
: "${REF:?}" "${CALIB:?}" "${TOOLS:?}" "${KLD:?}" "${GPUS_ALL:?}"
D=$WORK/fn4
IM=$D/imatrix-swift15-q8_0-agent-train.gguf
mkdir -p "$D"
[ -f "$IM" ] && { log "$IM exists"; exit 0; }
wait_gpus "$GPUS_ALL"
split=$(cat "$KLD/gpu_args" 2> /dev/null || true)
ngl="-ngl 99"; case "$split" in *"-fit on"*) ngl="";; esac
log "llama-imatrix on $(basename "$CALIB"), split ${split:-default}"
docker run --rm $DOCKER_BASE --name "pq-$TAG-imatrix" --gpus "\"device=$GPUS_ALL\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    $MOUNT_ARGS "$IMAGE" "$TOOLS/llama-imatrix" -m "$REF" -f "$CALIB" -o "$D/partial-imatrix.gguf" -c 512 -b 512 \
    --chunks 200 $ngl -fa on $split > "$D/imatrix.log" 2>&1
mv "$D/partial-imatrix.gguf" "$IM"
tail -n 3 "$D/imatrix.log"
ls -l "$IM"
