#!/usr/bin/env bash
# Q112 E1 (2026-10-24), one group: GPTQ Q5_0 tensors (gptq_body.py, with its proxy table) and the second file of a
# split GGUF whose first file is G3 without all three groups' tensors (made once with split_compose.py base, see
# ../README.md); checked tensor for tensor. $1 = group (o08, sh916, kv916). Run from the repository root with numpy and
# gguf-py (PYTHONPATH=<llama.cpp>/gguf-py). Env: MODELS (flint-BF16.gguf, flint-q73-G3.gguf, imatrix_unsloth.gguf),
# WORK (hess/ from act_hessian; gets the donors and proxy tables), OUT (the split files).
set -euo pipefail
G=$1; D=$WORK; H=$OUT; T=quants/gptq_head
ALL='blk\.[0-8]\.attn_output\.weight|blk\.(9|1[0-6])\.ffn_(gate|up|down)_shexp\.weight|blk\.(9|1[0-6])\.attn_(k|v)\.weight'
OPENBLAS_NUM_THREADS=12 python $T/gptq_body.py $G $MODELS/flint-BF16.gguf $MODELS/flint-q73-G3.gguf \
  $MODELS/imatrix_unsloth.gguf $D/hess $D/donor-$G.gguf | tee $D/proxy-$G.md
python $T/split_compose.py shard $H/flint-q112-G3-base.gguf $MODELS/flint-q73-G3.gguf $H/G3-$G flint-q112-G3-$G "$ALL" $D/donor-$G.gguf
python $T/split_compose.py verify $MODELS/flint-q73-G3.gguf $H/G3-$G/flint-q112-G3-$G-00001-of-00002.gguf $D/donor-$G.gguf
