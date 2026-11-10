#!/usr/bin/env bash
# make_head.sh: a checkpoint's own MTP head as a draft GGUF in the types of the draft in use (registry Q1168).
# CPU only, no PyTorch: containers at a low CPU share, each step's output in WORK.
#   1. extract   mtp_head_from_safetensors.py: the 33 tensors from the BF16 safetensors into WORK/<NAME>-BF16.gguf,
#                the key/value pairs and the tensor order of LIKE, and types.txt (LIKE's type per tensor)
#   2. quantize  llama-quantize --tensor-type-file types.txt ... Q4_K_M: WORK/q/<NAME>.gguf
#   3. retype    scripts/gguf_retype.py: llama-quantize leaves the indexer's two projections in the source type and
#                LIKE holds them in Q8_0, so they are converted with ggml's reference Q8_0 rounding, every other
#                tensor copied byte for byte: WORK/<NAME>.gguf, every tensor in LIKE's type; WORK/head.ready
#   4. compare   compare_heads.py: LIKE against the new file and against its BF16 values, tensor by tensor
#   MODEL_DIR=<the BF16 safetensors folder> LIKE=<the draft in use> TARGET=<first shard of the model it serves> \
#   REPO=<checkout of this repository> GGUF_PY=<llama.cpp>/gguf-py QUANTIZE=<llama-quantize> WORK=<scratch> \
#   NAME=<new file's name without .gguf> TITLE="<general.name>" PY_IMAGE=<an image with python3 and numpy> \
#   MOUNTS="<folders the containers read>" \
#     setsid -f nohup nice -n 19 bash make_head.sh > make_head.console.txt 2>&1 < /dev/null
# Env: IMAGE (llama-quantize's image, a CUDA build runs there with the driver stubbed), THREADS (8), DOCKER_ARGS.
# gguf-py imports PyYAML without using it here; an image without it gets an empty yaml.py on its path.
# Writes WORK/head.done or WORK/head.failed; prints HEAD_DONE or HEAD_FAILED as its last line.
set -uo pipefail
: "${MODEL_DIR:?}" "${LIKE:?}" "${TARGET:?}" "${REPO:?}" "${GGUF_PY:?}" "${QUANTIZE:?}" "${WORK:?}" "${NAME:?}"
: "${TITLE:?}" "${PY_IMAGE:?PY_IMAGE=<an image with python3 and numpy>}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
THREADS=${THREADS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mounts="-v $WORK:$WORK -v $HERE:$HERE:ro -v $REPO:$REPO:ro"
for m in ${MOUNTS:-}; do mounts="$mounts -v $m:$m:ro"; done
mkdir -p "$WORK/stubs" "$WORK/q"
: > "$WORK/stubs/yaml.py"
rm -f "$WORK/head.done" "$WORK/head.failed" "$WORK/head.ready"

say() { echo "$(date +%H:%M:%S) $*"; }
fail() { touch "$WORK/head.failed"; echo "HEAD_FAILED: $* $(date -Is)"; exit 1; }
py() {  # py NAME SCRIPT ARGS...: a Python script in the numpy image
  local name=$1; shift
  # shellcheck disable=SC2086
  docker run --rm --name "$name" ${DOCKER_ARGS:-} $mounts -e GGUF_PY="$GGUF_PY" -e STUBS="$WORK/stubs" "$PY_IMAGE" \
    bash -c 'python3 -c "import yaml" 2> /dev/null || GGUF_PY=$GGUF_PY:$STUBS
             PYTHONPATH=$GGUF_PY exec nice -n 19 python3 "$@"' py "$@"
}

say "1: extract"
py "$NAME-extract" "$HERE/mtp_head_from_safetensors.py" --model-dir "$MODEL_DIR" --like "$LIKE" \
  --out "$WORK/$NAME-BF16.gguf" --name "$TITLE" --tsv "$WORK/tensors.tsv" --types "$WORK/types.txt" \
  > "$WORK/extract.log" 2>&1 || fail "extract: $(tail -n 2 "$WORK/extract.log" | tr '\n' ' ')"

say "2: quantize ($(wc -l < "$WORK/types.txt") type rules)"
# shellcheck disable=SC2016,SC2086
docker run --rm --name "$NAME-quantize" ${DOCKER_ARGS:-} $mounts "$IMAGE" \
  bash -c 'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
           LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec nice -n 19 "$@"' quantize \
  "$QUANTIZE" --tensor-type-file "$WORK/types.txt" "$WORK/$NAME-BF16.gguf" "$WORK/q/$NAME.gguf" Q4_K_M "$THREADS" \
  > "$WORK/quantize.log" 2>&1 || fail "quantize: $(tail -n 2 "$WORK/quantize.log" | tr '\n' ' ')"

say "3: retype"
{ py "$NAME-retype" "$REPO/scripts/gguf_retype.py" retype --rule 'indexer\.[qk]_proj\.weight$=Q8_0@BF16' \
    --out-dir "$WORK" --jobs 4 "$WORK/q/$NAME.gguf" &&
  py "$NAME-retype" "$REPO/scripts/gguf_retype.py" verify "$WORK/q/$NAME.gguf" --new "$WORK/$NAME.gguf"
} > "$WORK/retype.log" 2>&1 || fail "retype: $(tail -n 2 "$WORK/retype.log" | tr '\n' ' ')"
md5sum "$WORK/$NAME.gguf" "$WORK/q/$NAME.gguf" "$WORK/$NAME-BF16.gguf" > "$WORK/md5.txt"
sha256sum "$WORK/$NAME.gguf" > "$WORK/sha256.txt"
ls -l "$WORK/$NAME.gguf" "$WORK/q/$NAME.gguf" "$WORK/$NAME-BF16.gguf"
touch "$WORK/head.ready"

say "4: compare"
py "$NAME-compare" "$HERE/compare_heads.py" "$LIKE" "$WORK/$NAME.gguf" --exact "$WORK/$NAME-BF16.gguf" \
  --target "$TARGET" --same-types --out "$WORK/compare_heads.tsv" > "$WORK/compare.log" 2>&1 \
  || fail "compare: $(tail -n 2 "$WORK/compare.log" | tr '\n' ' ')"

touch "$WORK/head.done"
echo "HEAD_DONE $(date -Is)"
