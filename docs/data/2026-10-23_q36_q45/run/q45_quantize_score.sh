#!/usr/bin/env bash
# Q45 (2026-10-23), one of its eleven recipe variants: Q1 = S4-L0's recipe plus attention q Q8_0 in layers 0-16,
# quantized from Flint BF16 with Unsloth's importance matrix, its bytes per token estimated, then scored with the
# rescale off. The other variants (and Q73's, from base type Q5_0) differ only in EXTRA and the base type: every
# recipe string is in quants/recipes/flint-mixes.tsv. Run from the repository root.
# Env: LLAMA_SRC (llama.cpp source tree for quantize_docker.sh), MODELS (holds flint-BF16.gguf and
#      imatrix_unsloth.gguf), WORK, BIN, GPUS as for kld_gpu.sh.
set -euo pipefail
EXTRA="--tensor-type blk\.([0-9]|1[0-6])\.attn_q\.weight=q8_0 --tensor-type blk\.(3[3-9]|4[0-8])\.attn_output\.weight=q4_1 --output-tensor-type q8_0 --tensor-type blk\.(46|47|48)\.ffn_down_exps\.weight=q8_0 --tensor-type attn_(k|v)\.weight=q8_0 --tensor-type _shexp=q8_0 --tensor-type blk\.[01]\.ffn_(gate|up|down)\.weight=q8_0 --tensor-type attn_output\.weight=q8_0 --tensor-type ffn_down_exps=q8_0 --tensor-type ffn_(gate|up)_exps=q5_0" \
  scripts/quantize_docker.sh "$LLAMA_SRC" "$MODELS" flint-BF16.gguf q45/flint-q45-Q1.gguf Q4_1 imatrix_unsloth.gguf
python3 scripts/token_cost.py --fixed-ms 8.4 "$MODELS/q45/flint-q45-Q1.gguf" | head -2
ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 docs/data/2026-10-23_q36_q45/run/kld_gpu.sh "$MODELS/q45/flint-q45-Q1.gguf" _nr
