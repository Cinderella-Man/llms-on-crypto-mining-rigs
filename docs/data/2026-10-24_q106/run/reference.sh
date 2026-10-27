#!/usr/bin/env bash
# Q106 (2026-10-24): Flint Q8_0 (Unsloth) reference logits on the agent text (24 BFCL v4 multi-turn conversations of
# 8,192 tokens each, read with --parse-special --no-escape), wikitext-2 test at 8,192 (12 chunks) and at 512 (80
# chunks). The agent reference file is 29.5 GB. Env as for common.sh.
set -e; . "$(dirname "$0")/common.sh"
run $MOUNTS $IMG $B/llama-perplexity -m $Q8 -f $W/agent/eval.txt --parse-special --no-escape -c 8192 --chunks 24 $FLAGS \
  --kl-divergence-base $W/q8.agent.kld > $L/base-agent.log 2>&1
run $MOUNTS $IMG $B/llama-perplexity -m $Q8 -f $WIKI -c 8192 --chunks 12 $FLAGS --kl-divergence-base $W/q8.wiki8k.kld > $L/base-wiki8k.log 2>&1
run $MOUNTS $IMG $B/llama-perplexity -m $Q8 -f $WIKI -c 512 --chunks 80 $FLAGS --kl-divergence-base $W/q8.wiki512.kld > $L/base-wiki512.log 2>&1
grep -E "Final estimate|tokenizing|chunks" $L/base-*.log | head -20
ls -l $W/*.kld
