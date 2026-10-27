#!/usr/bin/env bash
# Q112 E3 (2026-10-24, CPU): GPTQ for K2-r16's always-read tensors with K2-r16's Hessians (12 BLAS threads, so reruns
# give the same bytes): the 120 Q5_0 ones at the same type (donor-q5) and every Q8_0 one into Q5_0 (donor-q8, E3's
# demotion menu); proxy tables; variant K2-r16-G5 = the shared base + the 120 GPTQ tensors (no byte change), checked
# tensor for tensor. Needs c01-base.sh and g01-hess.sh first. Env as for common.sh.
set -e; . "$(dirname "$0")/common.sh"
py "python $T/gptq_body.py k2h_q5 $BF16 $K $IMAT $HD/hess $HD/donor-q5.gguf" | tee $D/proxy-q5.md
py "python $T/gptq_body.py k2h_q8 $BF16 $K $IMAT $HD/hess $HD/donor-q8.gguf" | tee $D/proxy-q8.md
py "python $T/split_compose.py shard $HM/flint-q112e2-K2-r16-base.gguf $K $HM/K2-r16-G5 flint-q112e2-K2-r16-G5 '$UA' $HD/donor-q5.gguf && \
    python $T/split_compose.py verify $K $HM/K2-r16-G5/flint-q112e2-K2-r16-G5-00001-of-00002.gguf $HD/donor-q5.gguf"
