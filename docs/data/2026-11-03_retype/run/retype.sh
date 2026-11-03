#!/usr/bin/env bash
# FN17: one three-shard Swift file with its hyper-connection projections BF16 -> Q8_0 and its router F32 -> F16,
# every other tensor byte for byte (scripts/gguf_retype.py, the CPU at the lowest priority); then `verify` against
# the unmodified shards, the md5 of each new shard, and the converted tensors against llama.cpp's own quantizer
# (scripts/gguf_retype_test.py with llama-quantize and libggml-base.so from build.sh).
#   SCRIPTS=<repo>/scripts MODEL=<first shard of the unmodified file> OUT=<folder> PREFIX=<new file name> \
#     QUANTIZE=<llama-quantize> GGML=<libggml-base.so> TEST=<scratch folder> [RULES="hc_=Q8_0@BF16 ..."] [JOBS=8] \
#     bash retype.sh
# RULES defaults to both conversions; RULES=hc_=Q8_0@BF16 leaves the router in F32.
set -euo pipefail
: "${SCRIPTS:?}" "${MODEL:?}" "${OUT:?}" "${PREFIX:?}" "${QUANTIZE:?}" "${GGML:?}" "${TEST:?}"
old=("${MODEL%-00001-of-*.gguf}"-0000?-of-*.gguf)
rules=()
for r in ${RULES:-hc_=Q8_0@BF16 ffn_gate_inp=F16@F32}; do rules+=(--rule "$r"); done
stamp() { echo "== $(date -Is) $*"; }
stamp retype
nice -n 19 python3 -I "$SCRIPTS/gguf_retype.py" retype "${rules[@]}" --out-dir "$OUT" --prefix "$PREFIX" \
    --jobs "${JOBS:-8}" "${old[@]}"
new=("$OUT/$PREFIX"-0000?-of-*.gguf)
stamp verify
nice -n 19 python3 -I "$SCRIPTS/gguf_retype.py" verify "${old[@]}" --new "${new[@]}"
stamp md5
(cd "$OUT" && nice -n 19 md5sum "$PREFIX"-0000?-of-*.gguf | tee "$PREFIX.md5")
stamp "against llama.cpp"
nice -n 19 python3 -I "$SCRIPTS/gguf_retype_test.py" "$TEST" --quantize "$QUANTIZE" --ggml "$GGML" --old "${old[@]}" \
    --new "${new[@]}"
stamp done
