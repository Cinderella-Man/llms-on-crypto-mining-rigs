#!/bin/bash
# Q46 (2026-10-23), run on the rig: one-token test-backend-ops cases for Q5_K, IQ4_XS and Q5_0 on one P106-090,
# Unsloth UD-Q4_K_XL on 6 cards with the Pascal kernels on/off (tg128 via scripts/bench.sh, GGML_CUDA_PROFILE tg32),
# then KLD of the kernels against the stock kernels on the same binary (4 x 512 tokens, one token per ubatch).
# Env: BIN (fork build at eeaa59660), MODEL (UD-Q4_K_XL GGUF), OUT (work dir), BENCH (copy of scripts/bench.sh), TEXT.
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
M=${MODEL:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
OUT=${OUT:-/tmp/q46}
BENCH=${BENCH:-$OUT/bench.sh}
TEXT=${TEXT:-/opt/llama-builds/data/wiki.test.raw}
mkdir -p "$OUT"; cd "$OUT"
echo "== tests $(date +%T)"
for t in q5_K iq4_xs q5_0; do
  CUDA_VISIBLE_DEVICES=0 $B/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ADD,MUL_MAT_ID,MUL_MAT_ID_SHARED -p "type_a=$t,.*,n=1," > tbo-$t.txt 2>&1
  CUDA_VISIBLE_DEVICES=0 $B/test-backend-ops -b CUDA0 -o MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION -p "type=$t,.*m=1," > tbo-fusion-$t.txt 2>&1
  echo "$t: $(grep -h "tests passed" tbo-$t.txt tbo-fusion-$t.txt | tr "\n" " ")"
done
for mode in 1 0; do
  echo "== tg128 mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode BIN=$B MODEL=$M TAG=q46-ud-mode$mode GPU_SETS=0,1,2,3,4,5 BENCH_ARGS="-ngl 99 -fa on -p 512 -n 128 -r 3" bash $BENCH
done
for mode in 1 0; do
  echo "== prof mode $mode $(date +%T)"
  GGML_CUDA_MMVQ_PASCAL=$mode CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUT/prof-ud-mode$mode.csv $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 32 -r 1 -o md > prof-ud-mode$mode.txt 2>&1
  echo "   exit $?"; grep tg32 prof-ud-mode$mode.txt
done
A="-m $M -ngl 99 -fa on -f $TEXT -c 512 --chunks 4 -b 512 -ub 1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
echo "== kld base mode 0 $(date +%T)"
GGML_CUDA_MMVQ_PASCAL=0 $B/llama-perplexity $A --kl-divergence-base $OUT/kld-base-ud.bin > kld-ud-mode0.txt 2>&1; echo "   exit $?"; grep "Final estimate" kld-ud-mode0.txt
echo "== kld mode 1 $(date +%T)"
GGML_CUDA_MMVQ_PASCAL=1 $B/llama-perplexity $A --kl-divergence-base $OUT/kld-base-ud.bin --kl-divergence > kld-ud-mode1.txt 2>&1; echo "   exit $?"
grep -E "Mean PPL\(Q\)  |Mean    KLD|Median  KLD|Same top" kld-ud-mode1.txt
rm -f $OUT/kld-base-ud.bin
echo DONE $(date +%T)
