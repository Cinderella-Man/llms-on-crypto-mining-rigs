#!/usr/bin/env bash
# An importance matrix for Ornith-1.5-35B from the makers' Q8_0 on CUDA GPUs (llama-imatrix in Docker): 200 chunks of
# 512 tokens of run/calib_text.py's text (agentic run 1's sessions outside the held-out 40, so the agent-text scores
# stay out of sample; no wikitext and no Exercism code, which the other two scores use).
#   BIN=<llama-imatrix, CUDA build> MODEL=<Q8_0 .gguf> TEXT=<calibration text> OUT=<imatrix .gguf> GPUS=<ids> bash imatrix.sh
set -euo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${OUT:?}" "${GPUS:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
CHUNKS=${CHUNKS:-200}
mounts=()
for d in "$(dirname "$MODEL")" "$(dirname "$(dirname "$BIN")")" "$(dirname "$TEXT")" "$(dirname "$OUT")"; do
    mounts+=(-v "$d:$d")
done
docker run --rm --label "$LABEL" --cpu-shares 256 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "${mounts[@]}" "$IMAGE" "$BIN" -m "$MODEL" -f "$TEXT" -o "$OUT" -c 512 -b 512 --chunks "$CHUNKS" -ngl 99 -fa on \
    > "${OUT%.gguf}.log" 2>&1
tail -3 "${OUT%.gguf}.log"
ls -l "$OUT"
