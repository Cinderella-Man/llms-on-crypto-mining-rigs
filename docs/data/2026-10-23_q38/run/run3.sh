#!/bin/bash
# Q38 (2026-10-23), run on the rig after run2.sh: the build with the Q5_0 kernel (112cdc597): S4-L0 tg128 and a profile
# with the full port, then KLD of the port against the stock kernels on the same binary, one token per ubatch (-ub 1)
# so that every token goes through the one-token kernels. Env: BIN, MODEL, OUT, BENCH (as run2.sh), TEXT (wikitext-2
# test file on the rig).
OUT=${OUT:-/tmp/q38}
while ! grep -q DONE $OUT/run2.log; do sleep 10; done
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-112cdc597-pascal-p106/bin}
M=${MODEL:-/mnt/models2/flint/flint-S4L0.gguf}
BENCH=${BENCH:-$OUT/bench.sh}
TEXT=${TEXT:-/opt/llama-builds/data/wiki.test.raw}
cd "$OUT"
echo "== tg128 q50 mode 1 $(date +%T)"
LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M TAG=q38-s4l0-q50 GPU_SETS=0,1,2,3,4,5 BENCH_ARGS="-ngl 99 -fa on -p 512 -n 128 -r 3" bash $BENCH
echo "== prof q50 mode 1 $(date +%T)"
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUT/prof-s4l0-q50.csv \
  $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 32 -r 1 -o md > $OUT/prof-s4l0-q50.txt 2>&1
echo "   exit $? $(date +%T)"; grep tg32 $OUT/prof-s4l0-q50.txt
for mode in 0 1; do
  echo "== kld mode $mode $(date +%T)"
  if [ $mode = 0 ]; then KL="--kl-divergence-base $OUT/kld-base-stock.bin"; else KL="--kl-divergence-base $OUT/kld-base-stock.bin --kl-divergence"; fi
  GGML_CUDA_MMVQ_PASCAL=$mode LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $B/llama-perplexity -m $M -ngl 99 -fa on \
    -f $TEXT -c 512 --chunks 4 -b 512 -ub 1 $KL > $OUT/kld-mode$mode.txt 2>&1
  echo "   exit $? $(date +%T)"; grep -E "Final estimate|Mean    KLD|Same top|Mean PPL" $OUT/kld-mode$mode.txt
done
rm -f $OUT/kld-base-stock.bin
echo DONE $(date +%T)
