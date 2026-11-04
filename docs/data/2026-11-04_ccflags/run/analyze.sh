#!/usr/bin/env bash
# Q671: the tables of one or more compile_arms.sh runs (analyze.py in a slim Python container, no GPU).
# Usage: WORKS='<compile_arms.sh output dir> [<another>...]' OUT=<tables dir> [DOCKER_ARGS=...] analyze.sh
#   the first folder holds the arm "base" and the build's own objects; the others add arms
set -euo pipefail
OUT=$(realpath -m "${OUT:?OUT=<tables dir>}")
PYIMAGE=${PYIMAGE:-python:3.12-slim-bookworm}
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$OUT"
mounts=(); args=(); i=0
for w in ${WORKS:?WORKS=<compile_arms.sh output dir>...}; do
    i=$((i + 1)); mounts+=(-v "$(realpath "$w"):/w$i:ro"); args+=("/w$i")
done
docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" \
    "${mounts[@]}" -v "$OUT:/o" -v "$HERE:/r:ro" "$PYIMAGE" nice -n 19 python3 -I /r/analyze.py /o "${args[@]}"
