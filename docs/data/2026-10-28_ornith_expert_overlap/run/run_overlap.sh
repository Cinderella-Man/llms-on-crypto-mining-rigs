#!/usr/bin/env bash
# Q411: the experts Ornith-1.5's router picks for every token and MoE layer of a text (llama-route-dump's ROUTE_TEXT
# mode, teacher-forced through the model on CUDA GPUs in Docker), then the distinct experts per window of 1-8
# consecutive tokens (bench/expert_overlap/analyze.py, standard library only). No timing.
#   BIN=<llama-route-dump> MODEL=<GGUF> TEXT=<text file> CHUNK=<tokens per sequence> GPUS=<ids> OUT=<prefix> \
#     ANALYZE=<bench/expert_overlap/analyze.py> [CHUNKS=<max sequences>] [TS=<-ts split>] bash run_overlap.sh
# Writes OUT.routes.bin (deleted after the analysis unless KEEP=1), OUT.log, OUT.json and OUT.txt (the summary).
set -euo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${CHUNK:?}" "${GPUS:?}" "${OUT:?}" "${ANALYZE:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
mounts=()
for d in "$(dirname "$BIN")" "$(dirname "$MODEL")" "$(dirname "$TEXT")" "$(dirname "$OUT")"; do mounts+=(-v "$d:$d"); done
docker run --rm --label "$LABEL" --cpu-shares 256 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e ROUTE_TEXT="$TEXT" -e ROUTE_CHUNK="$CHUNK" -e ROUTE_CHUNKS="${CHUNKS:-1000000}" -e ROUTE_OUT="$OUT.routes.bin" \
    "${mounts[@]}" "$IMAGE" "$BIN" -m "$MODEL" -c "$CHUNK" -b 512 -ub 512 -ngl 99 -fa on ${TS:+-ts "$TS"} \
    > "$OUT.log" 2>&1
docker run --rm --label "$LABEL" --cpu-shares 256 -v "$(dirname "$OUT"):$(dirname "$OUT")" \
    -v "$(dirname "$ANALYZE"):/tool:ro" python:3.12-slim \
    python -I /tool/analyze.py "$OUT.routes.bin" - "$OUT.json" 256 > "$OUT.txt"
[ "${KEEP:-0}" = 1 ] || rm -f "$OUT.routes.bin"
cat "$OUT.txt"
