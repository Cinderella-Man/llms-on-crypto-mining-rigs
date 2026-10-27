#!/usr/bin/env bash
# Q92 (2026-10-24): G3 (Q8_0 head) + the GPTQ Q4_0 screen (searched scales, H from F3's head inputs on wikitext-2 train
# and all 38 Q51 transcripts, H-mix; Q91's table used H-mixa, without its 8 held-out transcripts), top-M 256, checked
# byte for byte. Run from the repository root with numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
# Env: MODELS, S (the screen's .q4_0 file from gptq_head.py quant --type q4_0).
set -euo pipefail
G="python quants/gptq_head/gptq_head.py"
O=$MODELS/flint-q73-G3-screen.gguf
$G screen $MODELS/flint-q73-G3.gguf $S $O --top-m 256
$G verifyscreen $MODELS/flint-q73-G3.gguf $S $O --top-m 256
md5sum $S
