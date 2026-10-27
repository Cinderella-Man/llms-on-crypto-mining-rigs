#!/usr/bin/env bash
# Q112 E3 (2026-10-24, CPU): one candidate = K2-r16 with GPTQ on its 120 Q5_0 always-read tensors and GPTQ-Q5_0
# demotions of the Q8_0 always-read tensors DEMOTE matches, as the shared base + a small second file (1.2 GB); checked
# tensor for tensor. Needs c03-solve.sh first. Env as for common.sh.
#   make_candidate.sh NAME 'DEMOTE_REGEX'      e.g. make_candidate.sh K2-r16-G5-o08 '^blk\.[0-8]\.attn_output\.weight$'
set -e; . "$(dirname "$0")/common.sh"
N=$1; R=$2
py "python $T/split_compose.py shard $HM/flint-q112e2-K2-r16-base.gguf $K $HM/$N flint-q112e2-$N '$UA' '$HD/donor-q5.gguf,$HD/donor-q8.gguf@$R' && \
    python $T/split_compose.py verify $K $HM/$N/flint-q112e2-$N-00001-of-00002.gguf '$HD/donor-q5.gguf,$HD/donor-q8.gguf@$R'"
