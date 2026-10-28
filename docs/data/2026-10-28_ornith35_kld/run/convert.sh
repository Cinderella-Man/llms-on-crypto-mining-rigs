#!/usr/bin/env bash
# Ornith-1.5-35B-A3B's BF16 safetensors to one BF16 GGUF with llama.cpp's convert_hf_to_gguf.py (the text model and
# its MTP layer; the vision tower is left out), in a PyTorch container at a low CPU share. The converter's Python
# packages besides torch go to PYLIB once.
#   SRC=<folder of the safetensors> LLAMA=<llama.cpp checkout> OUT=<output .gguf> PYLIB=<folder for pip packages> \
#     bash convert.sh
# LLAMA should be the checkout of the llama.cpp that quantizes and scores the files (here d77256999).
set -euo pipefail
: "${SRC:?}" "${LLAMA:?}" "${OUT:?}" "${PYLIB:?}"
IMAGE=${IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
LABEL=${LABEL:-pascal-quants=1}
mkdir -p "$(dirname "$OUT")" "$PYLIB"
docker run --rm --label "$LABEL" --cpu-shares 256 -e PYTHONPATH="$PYLIB:$LLAMA/gguf-py" \
    -v "$SRC:$SRC:ro" -v "$LLAMA:$LLAMA:ro" -v "$(dirname "$OUT"):$(dirname "$OUT")" -v "$PYLIB:$PYLIB" "$IMAGE" \
    bash -euo pipefail -c "
    [ -d '$PYLIB/transformers' ] || pip install -q --target '$PYLIB' transformers sentencepiece protobuf > /dev/null
    python -c 'import transformers; print(\"transformers\", transformers.__version__)'
    nice -n 19 python '$LLAMA/convert_hf_to_gguf.py' --outtype bf16 --outfile '$OUT' '$SRC'"
ls -l "$OUT"
