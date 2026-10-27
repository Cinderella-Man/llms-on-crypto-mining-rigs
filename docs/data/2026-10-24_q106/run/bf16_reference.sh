#!/usr/bin/env bash
# Q106 bias check (2026-10-24): Flint BF16 reference logits on 12 of the 24 agent chunks (3 per category, 6 per
# producing model) on two 24 GB GPUs: the routed experts of layers 21-48 in system RAM, -ts 11,39 (the GPU-resident
# experts are all in layers 2-20), -ub 2048. Then every file is scored against it with score1, e.g. G3 below.
# Env as for common.sh; $W/agent12/eval.txt from subset_eval.py (see ../README.md).
set -e; . "$(dirname "$0")/common.sh"
run $MOUNTS $IMG $B/llama-perplexity -m $MODELS/flint-BF16.gguf -f $W/agent12/eval.txt --parse-special --no-escape \
  -c 8192 --chunks 12 -b 2048 -ub 2048 -ngl 99 -ts 11,39 -fa on -ot "blk\.(2[1-9]|3[0-9]|4[0-8])\.ffn_(gate|up|down)_exps\.weight=CPU" \
  --kl-divergence-base $W/bf16.agent12.kld > $L/base-bf16-agent12.log 2>&1
grep -E "Final estimate|chunks|model buffer|out of memory" $L/base-bf16-agent12.log | head
score1 g3 $MODELS/flint-q73-G3.gguf "-e LLAMA_NO_MOE_DOWN_RESCALE=1" bf16.agent12.kld bf16agent12 agent12
