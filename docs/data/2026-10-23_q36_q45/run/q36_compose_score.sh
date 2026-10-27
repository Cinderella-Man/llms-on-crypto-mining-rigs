#!/usr/bin/env bash
# Q36 (2026-10-23), one of the eighteen group probes: a1 = S4 with attention output back to Q4_1 in layers 0-8, its
# tensors taken byte for byte from S1 (which holds attention output in Q4_1), scored with the rescale off on the GPU
# set of class a, whose own S4 score is the reference. The other probes change the class (a: attention output from
# S1; d: expert down-projections 2-45 and u: expert gate/up 2-48, both from S2) and the 8-layer group. Run from the
# repository root. Env: MODELS (holds flint-mixS1/S2/S4.gguf, made from quants/recipes/flint-speed-s*.txt), PYTHONPATH
# (<llama.cpp>/gguf-py), WORK, BIN, GPUS as for kld_gpu.sh.
set -euo pipefail
O=$MODELS/q36/q36-S4-a1.gguf; mkdir -p "$MODELS/q36"
python3 scripts/compose_gguf.py "$MODELS/flint-mixS4.gguf" "$O" \
  'blk\.(0|1|2|3|4|5|6|7|8)\.attn_output\.weight='"$MODELS/flint-mixS1.gguf"
ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 docs/data/2026-10-23_q36_q45/run/kld_gpu.sh "$O" _nr
rm -f "$O"
