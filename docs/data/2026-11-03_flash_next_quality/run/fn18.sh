#!/usr/bin/env bash
# FN18 (Q552): the two-rig Pascal-format file. Routed experts (ffn_gate/up/down_exps) in Q4_0 with FN4's importance
# matrix, every other tensor as in the Q8_0 reference: llama-quantize --allow-requantize from the reference (the
# source is the Q8_0, not the BF16; tensors already Q8_0 are copied, and llama-quantize never touches hc_*), with
# --keep-split so the n-gram table stays alone in its shard. That shard is then replaced by the published IQ4_NL table
# (swap_tensor.py: the table bytes every published build ships). Then its KL divergence on all seven GPUs.
#   (env: the chain's) bash fn18.sh   ->  $HOT/fn18/Swift-1.5-Qwen3.8-Flash-Next-PQ4exp-0000k-of-0000N.gguf
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
: "${REF:?}" "${TOOLS:?}" "${HOT:?}" "${KLD:?}" "${PUBLISHED_TABLE:?}" "${GPUS_ALL:?}"
D=$HOT/fn18
NAME=Swift-1.5-Qwen3.8-Flash-Next-PQ4exp
IM=$WORK/fn4/imatrix-swift15-q8_0-agent-train.gguf
mkdir -p "$D"
if [ ! -f "$D/quantize.done" ]; then
    log "llama-quantize: experts Q4_0 with $(basename "$IM"), the rest as in the reference"
    docker run --rm $DOCKER_BASE --name "pq-$TAG-quantize" $MOUNT_ARGS "$IMAGE" nice -n 19 "$TOOLS/llama-quantize" \
        --allow-requantize --keep-split --imatrix "$IM" --tensor-type ffn_gate_exps=q4_0 \
        --tensor-type ffn_up_exps=q4_0 --tensor-type ffn_down_exps=q4_0 "$REF" "$D/$NAME" Q8_0 "${THREADS:-8}" \
        > "$D/quantize.log" 2>&1
    date -Is > "$D/quantize.done"
fi
if [ ! -f "$D/table.done" ]; then
    python3 -I "$HERE/swap_tensor.py" --tensor per_layer_token_embd.weight --source "$PUBLISHED_TABLE" \
        --out "$D/table.new" "$D/$NAME"-0000?-of-0000?.gguf | tee "$D/table.log"
    mv "$D/table.new" "$(sed -n 's/^\(.*\.gguf\): per_layer_token_embd.*/\1/p' "$D/table.log")"
    date -Is > "$D/table.done"
fi
python3 -I "$HERE/layout.py" "$D/$NAME"-0000?-of-0000?.gguf | tee "$D/layout.txt"
md5sum "$D/$NAME"-0000?-of-0000?.gguf > "$D/md5.txt"
wait_gpus "$GPUS_ALL"
GPUS=$GPUS_ALL OUT=$KLD bash "$HERE/kld.sh" kld "$(ls "$D/$NAME"-00001-of-0000?.gguf)" fn18-pq4exp
