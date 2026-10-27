#!/bin/bash
# Q38 (2026-10-23), run on the rig after run5.sh: noise floor for the output check. Base: stock kernels with 4 warps per
# block (the standard build); against it the same stock kernels with 2 warps (wide-rows shape; only the summation order
# changes) and the Pascal kernels, S4-L0 on 6 cards, 4 x 512 tokens one token per ubatch.
# Env: STD (standard build, graphs61 at d77256999), BIN (fork build at 7d1f070fd), MODEL, OUT, TEXT.
OUT=${OUT:-/tmp/q38}
while ! grep -q DONE $OUT/run5.log 2>/dev/null; do sleep 10; done
STD=${STD:-/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type/bin}
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-7d1f070fd-pascal-p106/bin}
M=${MODEL:-/mnt/models2/flint/flint-S4L0.gguf}
TEXT=${TEXT:-/opt/llama-builds/data/wiki.test.raw}
A="-m $M -ngl 99 -fa on -f $TEXT -c 512 --chunks 4 -b 512 -ub 1"
cd "$OUT"
export LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
echo "== base: stock 4 warps $(date +%T)"
$STD/llama-perplexity $A --kl-divergence-base $OUT/kld-base-std.bin > kld-std-base.txt 2>&1; echo "   exit $?"; grep "Final estimate" kld-std-base.txt
for mode in 0 1; do
  echo "== kld vs stock 4 warps, mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode $B/llama-perplexity $A --kl-divergence-base $OUT/kld-base-std.bin --kl-divergence > kld-std-mode$mode.txt 2>&1; echo "   exit $?"
  grep -E "Mean PPL\(Q\)  |Mean    KLD|Median  KLD|Same top" kld-std-mode$mode.txt
done
rm -f $OUT/kld-base-std.bin
echo DONE $(date +%T)
