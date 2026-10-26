#!/usr/bin/env bash
# Serve Flint (the makers' NVFP4 GGUF, W4A16 on Marlin) with vLLM's OpenAI-compatible server, pipeline-parallel over
# two GPUs below compute 9.0 (two RTX 3090s), as the writer of drafter training text (generate.py --api vllm).
#
# usage: GPUS=2,5 MODEL_DIR=/models/flint-nvfp4 HF_DIR=/models/flint WORK=/work/vllm bash serve_vllm.sh
#   MODEL_DIR  directory holding flint-nvfp4.gguf (RecursiveSI/flint-nvfp4-gguf) and nothing else ending in .gguf
#   HF_DIR     the BF16 repository's small files (RecursiveSI/flint: config.json, tokenizer*, chat_template.jinja);
#              the GGUF carries neither a config vLLM reads nor the tokenizer
#   WORK       where the patched model file and the server log go
#   MOUNTS     extra "-v a:a" options, so every path the container reads is mounted at the same path
#   PREFIX_CACHE  1: vLLM's prefix caching on (agent sessions resend their growing history every turn); default
#              off. Check it against the patched Flint attention first: the same turns, greedy, with and without
#   PP         pipeline stages (default 2), one per GPU; stages follow the GPUs' PCI bus order
#   PARTITION  layers per stage, e.g. 11,12,26 for a 10 GB + 12 GB + 24 GB set (default: even). Every stage also holds
#              the BF16 embedding and output head (1.15 GiB), and all stages get the same number of KV blocks, so
#              smaller cards take fewer layers.
# Two servers on one machine need their own NAME, PORT and WORK.
# Stop: docker stop $NAME. The patched model file is rebuilt from the image's own copy on every start, and
# patch_flint_attention.py refuses to patch if the image's file changed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${GPUS:?GPUS=a,b}" "${MODEL_DIR:?}" "${HF_DIR:?}" "${WORK:?}"
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:1440cb63fac5fa19040c295fcd5c6e6d551a506aa72a68fe3425cfd6ef8cf4b8}  # nightly c50aca14 (CUDA 13.0)
NAME=${NAME:-pq-vllm-gen}
PORT=${PORT:-18560}
MAX_SEQS=${MAX_SEQS:-256}
MAX_LEN=${MAX_LEN:-8192}
MEM=${MEM:-0.92}
SITE=/usr/local/lib/python3.12/dist-packages
mkdir -p "$WORK"

docker run --rm --entrypoint cat "$IMAGE" $SITE/vllm/models/flint/model.py > "$WORK/model_image.py"
python3 "$here/patch_flint_attention.py" "$WORK/model_image.py" "$WORK/model_patched.py"

docker rm -f "$NAME" 2>/dev/null || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --label pascal-quants=1 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ${PARTITION:+-e VLLM_PP_LAYER_PARTITION=$PARTITION} \
    --restart on-failure:3 --ipc=host --network host ${MOUNTS:-} -v "$MODEL_DIR:$MODEL_DIR:ro" -v "$HF_DIR:$HF_DIR:ro" -v "$WORK:$WORK" \
    -v "$WORK/model_patched.py:$SITE/vllm/models/flint/model.py:ro" \
    "$IMAGE" "$MODEL_DIR" --hf-config-path "$HF_DIR" --tokenizer "$HF_DIR" --served-model-name flint \
    --pipeline-parallel-size "${PP:-2}" --attention-backend TRITON_ATTN --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$MEM" $([ "${PREFIX_CACHE:-0}" = 1 ] && echo --enable-prefix-caching || echo --no-enable-prefix-caching) \
    --host 127.0.0.1 --port "$PORT"
echo "started $NAME on GPUs $GPUS, port $PORT; log: docker logs -f $NAME; ready when curl -s 127.0.0.1:$PORT/health succeeds"
