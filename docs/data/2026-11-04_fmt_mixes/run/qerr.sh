#!/usr/bin/env bash
# Each dense type's error on Flash-Next's own tensors (bench/pascal_mix/qerr.py), in a Python container with numpy,
# CPU only (nice 19, idle I/O priority, CPU shares 256, pinned to CPUS). FILES are "<name>=<first shard>" words;
# every folder they and REF live in must be under MOUNTS. VENV is a Python 3.12 virtual environment with numpy (made
# in the same image, so its interpreter path resolves there); GGUF_PY a llama.cpp tree's gguf-py that knows Q2_0.
#   REPO=<this repository> REF=<first shard of the Swift-1.5 Q8_0> IMATRIX=<imatrix.gguf> FILES="<name>=<shard> ..."
#   OUT=<dir> VENV=<venv dir> GGUF_PY=<gguf-py dir> MOUNTS="<dirs>" [CPUS=20,21]
#   [IMAGE=python:3.12-slim-bookworm] [LAYERS=0,3,4,...] bash qerr.sh
set -euo pipefail
: "${REPO:?}" "${REF:?}" "${IMATRIX:?}" "${FILES:?}" "${OUT:?}" "${VENV:?}" "${GGUF_PY:?}" "${MOUNTS:?}"
CPUS=${CPUS:-20,21} IMAGE=${IMAGE:-python:3.12-slim-bookworm}
mkdir -p "$OUT"
margs=() fargs=()
for d in $MOUNTS; do margs+=(-v "$d:$d"); done
for f in $FILES; do fargs+=(--file "$f"); done
docker run --rm --label pascal-quants=1 --name pq-fmt-qerr --cpu-shares 256 --cpuset-cpus "$CPUS" "${margs[@]}" \
    -v "$VENV:/venv:ro" -e PYTHONPATH="$GGUF_PY" "$IMAGE" nice -n 19 ionice -c3 /venv/bin/python -u "$REPO/bench/pascal_mix/qerr.py" \
    --ref "$REF" --imatrix "$IMATRIX" "${fargs[@]}" ${LAYERS:+--layers "$LAYERS"} --out-dir "$OUT"
