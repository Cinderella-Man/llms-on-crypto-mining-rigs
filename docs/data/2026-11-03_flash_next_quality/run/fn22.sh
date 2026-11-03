#!/usr/bin/env bash
# FN22 (Q579): what the published IQ4_NL n-gram table costs against a near-lossless one. The reference with its Q8_0
# table replaced by the published IQ4_NL table (swap_tensor.py; the other shards are links to the reference's),
# scored against the reference's bases on all seven GPUs.
#   (env: the chain's) bash fn22.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
: "${REF:?}" "${HOT:?}" "${KLD:?}" "${PUBLISHED_TABLE:?}" "${GPUS_ALL:?}"
D=$HOT/fn22/q8_0-iq4nl-table
mkdir -p "$D"
if [ ! -f "$D/table.done" ]; then
    for f in "$(dirname "$REF")"/*-of-*.gguf; do ln -sfn "$f" "$D/"; done
    python3 -I "$HERE/swap_tensor.py" --tensor per_layer_token_embd.weight --source "$PUBLISHED_TABLE" \
        --out "$D/table.new" "$D"/*-of-*.gguf | tee "$D/table.log"
    mv "$D/table.new" "$(sed -n 's/^\(.*\.gguf\): per_layer_token_embd.*/\1/p' "$D/table.log")"
    date -Is > "$D/table.done"
fi
wait_gpus "$GPUS_ALL"
GPUS=$GPUS_ALL OUT=$KLD bash "$HERE/kld.sh" kld "$D/$(basename "$REF")" q8_0-iq4nl-table
