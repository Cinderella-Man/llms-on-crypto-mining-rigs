#!/bin/bash
# Q46 (2026-10-23), run on the rig: UD-Q4_K_XL noise floor on 6 cards. Base: stock kernels with 4 warps (the standard
# build); against it the same stock kernels with 2 warps (fork build, kernels off) and the Pascal kernels.
# Env: STD (standard build, graphs61 at d77256999), BIN (fork build at eeaa59660), MODEL, OUT, TEXT.
STD=${STD:-/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type/bin}
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
M=${MODEL:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
OUT=${OUT:-/tmp/q46}
TEXT=${TEXT:-/opt/llama-builds/data/wiki.test.raw}
A="-m $M -ngl 99 -fa on -f $TEXT -c 512 --chunks 4 -b 512 -ub 1"
mkdir -p "$OUT"; cd "$OUT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
echo "== base: stock 4 warps $(date +%T)"
$STD/llama-perplexity $A --kl-divergence-base $OUT/kld-base-std.bin > kld-ud-std-base.txt 2>&1; echo "   exit $?"; grep "Final estimate" kld-ud-std-base.txt
for mode in 0 1; do
  echo "== kld vs stock 4 warps, mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode $B/llama-perplexity $A --kl-divergence-base $OUT/kld-base-std.bin --kl-divergence > kld-ud-std-mode$mode.txt 2>&1; echo "   exit $?"
  grep -E "Mean PPL\(Q\)  |Mean    KLD|Same top" kld-ud-std-mode$mode.txt
done
rm -f $OUT/kld-base-std.bin
echo DONE $(date +%T)
