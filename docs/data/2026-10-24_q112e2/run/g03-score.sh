#!/usr/bin/env bash
# Q112 E3 (2026-10-24, GPU): 80 x 512 wikitext KLD against BF16 on the same cards: K2-r16 (the paired baseline) and
# K2-r16-G5 (GPTQ on its 120 Q5_0 always-read tensors, same bytes), rescale off. Scoring as in
# docs/data/2026-10-23_q36_q45/run/kld_gpu.sh. Needs c03-solve.sh first. Env as for common.sh and kld_gpu.sh.
set -e; . "$(dirname "$0")/common.sh"
S=docs/data/2026-10-23_q36_q45/run/kld_gpu.sh
ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 $S $K _nr_v100
ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 $S $HM/K2-r16-G5/flint-q112e2-K2-r16-G5-00001-of-00002.gguf _nr_v100
