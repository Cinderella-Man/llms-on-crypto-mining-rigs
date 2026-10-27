#!/usr/bin/env bash
# Q112 E2/E3 (2026-10-24, CPU): the shared first file for every always-read variant of K2-r16 (K2-r16 without all its
# always-read matrices: 436 of 779 tensors, 28.3 GB) and an identity variant (second file = K2-r16's own 343
# always-read tensors, 1.18 GB), checked tensor for tensor. Env as for common.sh.
set -e; . "$(dirname "$0")/common.sh"
py "python $T/split_compose.py base $K $HM/flint-q112e2-K2-r16-base.gguf '$UA' && \
    python $T/split_compose.py shard $HM/flint-q112e2-K2-r16-base.gguf $K $HM/K2-r16-id flint-q112e2-K2-r16-id '$UA' && \
    python $T/split_compose.py verify $K $HM/K2-r16-id/flint-q112e2-K2-r16-id-00001-of-00002.gguf"
