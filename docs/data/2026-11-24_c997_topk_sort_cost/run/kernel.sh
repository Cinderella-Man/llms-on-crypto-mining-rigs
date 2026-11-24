#!/usr/bin/env bash
# C997 (Q1730): the card's top-k with GGML_CUDA_TOP_K_SORT unset and set on one P106 (CUDA0 only), test-backend-ops'
# perf cases of TOP_K whose shape holds a sampler row (151,936 and 248,320 columns) or a sparse-attention indexer row
# (131,072, 33,024, 132,096 columns), alternated u1 s1 u2 s2. Env: BIN (the build's bin folder), O (output folder).
set -u
: "${BIN:?}" "${O:?}"
cd "$O" || exit 1
md5sum "$BIN/test-backend-ops" > md5.txt
F='151936|248320|131072|33024|132096'
for a in u1 s1 u2 s2; do
  case $a in s*) E="GGML_CUDA_TOP_K_SORT=1" ;; *) E="GGML_CUDA_TOP_K_SORT_UNSET=1" ;; esac
  date +%T > "t-$a.txt"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,memory.used --format=csv,noheader -i 0 >> "t-$a.txt"
  env $E CUDA_VISIBLE_DEVICES=0 timeout 300 "$BIN/test-backend-ops" perf -o TOP_K -b CUDA0 -p "$F" > "perf-$a.txt" 2>&1
  echo "$a rc=$?" >> rc.txt
  date +%T >> "t-$a.txt"
done
touch done
