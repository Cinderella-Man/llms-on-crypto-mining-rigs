#!/usr/bin/env bash
# run/mtp_accept.cpp, mtp_chain.cpp or mtp_variant.cpp (built by run/build_tool.sh) on one text, on CUDA GPUs in Docker:
# the table to OUT, the log next to it (OUT with .log). mtp_variant's settings (DRAFT, DRAFT_KV, DRAFT_CPU, SHORTLIST,
# SCREEN, FAST) pass through from the environment.
#   BIN=<mtp-accept> MODEL=<GGUF with its MTP layer> TEXT=<text file> CHUNK=<tokens per chunk> GPUS=<ids> OUT=<.tsv> \
#     [CHUNKS=<max chunks>] [TS=<-ts split>] bash run_accept.sh
set -euo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${CHUNK:?}" "${GPUS:?}" "${OUT:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
mounts=() envs=()
for d in "$(dirname "$BIN")" "$(dirname "$MODEL")" "$(dirname "$TEXT")" "$(dirname "$OUT")"; do mounts+=(-v "$d:$d"); done
# run/mtp_variant.cpp's settings, passed through when set (DRAFT and SHORTLIST files need their folders mounted)
for v in DRAFT DRAFT_KV DRAFT_CPU SHORTLIST SCREEN FAST; do
    [ -n "${!v:-}" ] && envs+=(-e "$v=${!v}")
done
for v in DRAFT SHORTLIST; do [ -n "${!v:-}" ] && mounts+=(-v "$(dirname "${!v}"):$(dirname "${!v}")"); done
docker run --rm --label "$LABEL" --cpu-shares 256 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${envs[@]}" \
    -e OMP_NUM_THREADS="${THREADS:-8}" "${mounts[@]}" "$IMAGE" "$BIN" -m "$MODEL" -f "$TEXT" -c "$CHUNK" \
    --chunks "${CHUNKS:--1}" -ngl 99 -fa on ${TS:+-ts "$TS"} 2> "${OUT%.tsv}.log" | sed -n '/^bin_lo/,$p' > "$OUT"
cat "$OUT"   # the sed keeps the table only: the image's entrypoint prints a banner to stdout
