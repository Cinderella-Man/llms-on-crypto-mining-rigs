#!/usr/bin/env bash
# Runs table_bits in Docker, each folder in MOUNTS mounted at the same path (the shard's, the outputs', and any
# folder a symlink points into).
#   BIN=<table_bits> MOUNTS="<dir> ..." [THREADS=16] [DOCKER_ARGS="--label ... --cpu-shares 256"] [SAMPLE=262144] \
#     [HIST=<census histogram TSV>] bash tb.sh census|roundtrip ARGS...
set -euo pipefail
: "${BIN:?}" "${MOUNTS:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
m=(-v "$(dirname "$BIN"):$(dirname "$BIN")" -e OMP_NUM_THREADS="${THREADS:-16}")
for d in $MOUNTS; do m+=(-v "$d:$d"); done
for k in SAMPLE HIST ROUNDS; do [ -n "${!k:-}" ] && m+=(-e "$k=${!k}"); done
exec docker run --rm -i ${DOCKER_ARGS:-} "${m[@]}" --entrypoint "$BIN" "$IMAGE" "$@"
