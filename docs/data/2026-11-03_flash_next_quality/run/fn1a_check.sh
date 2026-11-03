#!/usr/bin/env bash
# FN1a's identity check: the downloaded Q8_0 against the BF16 safetensors of the same model (same_model.py): two
# hyper-connection projections, one expert from each of two fused expert tensors, two routers and a shared expert, from
# layers 3, 20 and 40. Any difference fails the step, and the chain stops: the reference would be another model. Then
# the file's layout (layout.py).
#   (env: BF16 Q8_HOT) bash fn1a_check.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
: "${BF16:?}" "${Q8_HOT:?}"
L=model.language_model.layers
log "identity check: $BF16 against $Q8_HOT"
nice -n 19 python3 -I "$HERE/same_model.py" "$BF16" "$Q8_HOT"/*-of-*.gguf \
    --pair "$L.3.attn_hyper_connection.input_mix_weight_up.weight=blk.3.hc_attn_up.weight" \
    --pair "$L.40.mlp_hyper_connection.input_mix_weight_down.weight=blk.40.hc_ffn_down.weight" \
    --pair "$L.3.mlp.experts.down_proj=blk.3.ffn_down_exps.weight@0" \
    --pair "$L.40.mlp.experts.down_proj=blk.40.ffn_down_exps.weight@511" \
    --pair "$L.3.mlp.gate.weight=blk.3.ffn_gate_inp.weight" \
    --pair "$L.40.mlp.gate.weight=blk.40.ffn_gate_inp.weight" \
    --pair "$L.20.mlp.shared_expert.down_proj.weight=blk.20.ffn_down_shexp.weight"
log "identity check passed"
python3 -I "$HERE/layout.py" "$Q8_HOT"/*-of-*.gguf
