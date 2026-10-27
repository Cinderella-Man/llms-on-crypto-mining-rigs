#!/usr/bin/env bash
# Q112 E2/E3 check (2026-10-24, CPU only): the identity variant (shared base + K2-r16's own always-read tensors) loads
# and computes exactly K2-r16. One 512-token chunk of wikitext-2 test on the CPU: K2-r16's logits saved, the split pair
# scored against them (KLD must be 0, same top token 100 %). Env as for common.sh, plus BIN_CPU (bin/ of a CPU build).
set -e; . "$(dirname "$0")/common.sh"
P=$BIN_CPU/llama-perplexity
run() { docker run --rm $MOUNTS -v "$(dirname "$BIN_CPU"):$(dirname "$BIN_CPU"):ro" -e LLAMA_NO_MOE_DOWN_RESCALE=1 $IMG "$@"; }
run $P -m $K -f $WORK/wikitext-2-raw/wiki.test.raw -c 512 --chunks 1 -t 12 --kl-divergence-base $HD/idcheck.kld > $D/idcheck-base.log 2>&1
run $P -m $HM/K2-r16-id/flint-q112e2-K2-r16-id-00001-of-00002.gguf -f $WORK/wikitext-2-raw/wiki.test.raw -c 512 \
  --chunks 1 -t 12 --kl-divergence-base $HD/idcheck.kld --kl-divergence > $D/idcheck.log 2>&1
grep -E "^Mean +KLD|^Maximum KLD|^Same top p|split|loaded" $D/idcheck.log | head
