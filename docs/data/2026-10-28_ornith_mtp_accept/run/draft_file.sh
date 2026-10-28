#!/usr/bin/env bash
# A draft file for run/mtp_variant.cpp (DRAFT=): the MTP block, token embedding and output head in chosen types, the rest
# of the model, which the MTP graph never reads, in TQ1_0 to keep the file small. From a BF16 (or any) GGUF with the
# CPU build of llama-quantize in Docker (--cpu-shares 256, nice -n 19, 8 threads).
#   QUANT=<llama-quantize> SRC=<.gguf> OUT=<.gguf> BLOCK=<MTP block index, 40 for the 35B, 48 for the 122B>
#     [MTP=q8_0 | e:<type> for the routed experts only, the rest of the block q8_0] [HEAD=q8_0] [EMBD=q8_0]
#     [ALLOW_REQUANT=1 for a quantized SRC] bash draft_file.sh
set -euo pipefail
: "${QUANT:?}" "${SRC:?}" "${OUT:?}" "${BLOCK:?}"
MTP=${MTP:-q8_0}; HEAD=${HEAD:-q8_0}; EMBD=${EMBD:-q8_0}; TRUNK=${TRUNK:-TQ1_0}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
case $MTP in
    e:*) T="--tensor-type blk\\.$BLOCK\\.ffn_(gate|up|down)_exps=${MTP#e:} --tensor-type blk\\.$BLOCK\\.=q8_0" ;;
    *)   T="--tensor-type blk\\.$BLOCK\\.=$MTP" ;;
esac
bin=$(dirname "$QUANT")
docker run --rm --label "$LABEL" --cpu-shares 256 -v "$bin:$bin:ro" -v "$(dirname "$SRC"):$(dirname "$SRC"):ro" \
    -v "$(dirname "$OUT"):$(dirname "$OUT")" "$IMAGE" nice -n 19 "$QUANT" ${ALLOW_REQUANT:+--allow-requantize} \
    --token-embedding-type "$EMBD" --output-tensor-type "$HEAD" $T "$SRC" "$OUT" "$TRUNK" 8 > "${OUT%.gguf}.log" 2>&1
grep -E "quant size" "${OUT%.gguf}.log"
ls -l "$OUT"
