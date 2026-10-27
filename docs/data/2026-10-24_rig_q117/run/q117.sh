#!/bin/bash
# Q117 headline (2026-10-24): stock llama.cpp, Miner Llama v1, the v2 base and the combined v2 build on UD-Q4_K_XL, 6 and 9 cards;
# then the combined build's ablation switches on 6 cards. One llama-bench at a time.
# Usage: [RIG=rig] [OUT=dir] q117.sh   (writes headline.md and ablations.md; ../llama_bench.csv holds both)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
R=/opt/llama-builds
M=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
ARGS="-m $M -ngl 99 -fa on -p 512,2048,8192 -n 128 -r 3 -o md"
declare -A ENVS=(
  [stock]="$R/llama.cpp-20261022-d77256999/bin|"
  [v1]="$R/llama.cpp-20261023-eeaa59660-pascal-p106/bin|"
  [v2]="$R/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin|LLAMA_NO_MOE_DOWN_RESCALE=tg"
  [combined]="$R/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin|LLAMA_NO_MOE_DOWN_RESCALE=tg"
)
bench() { # $1 label, $2 bin, $3 env, $4 visible devices
  echo "== $1 cards=$4 $(date +%T)"
  ssh "$RIG" "nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader | tr '\n' ' '"; echo
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$4 $3 $2/llama-bench $ARGS" 2>&1 | grep -E '^\||error|fail' 
}
echo "== warm-up"; ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LLAMA_NO_MOE_DOWN_RESCALE=tg $R/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin/llama-bench -m $M -ngl 99 -fa on -p 512 -n 32 -r 1 -o md" > /dev/null 2>&1
for cards in 0,1,2,3,4,5 0,1,2,3,4,5,6,7,8; do
  for b in stock v1 v2 combined; do
    IFS='|' read -r bin env <<< "${ENVS[$b]}"
    bench "$b" "$bin" "$env" "$cards"
  done
done | tee "$OUT/headline.md"
C="$R/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin"
{
  bench "combined-all-on" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg" 0,1,2,3,4,5
  bench "combined-no-normq8(Q100)" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_NORM_Q8=0" 0,1,2,3,4,5
  bench "combined-mmq-legacy(Q108)" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_MMQ_MOE_LEGACY=1" 0,1,2,3,4,5
  bench "combined-reuse-wait(Q109)" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg LLAMA_GRAPH_REUSE_WAIT=1" 0,1,2,3,4,5
  bench "combined-host-mask(Q110)" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg LLAMA_GPU_KQ_MASK=0" 0,1,2,3,4,5
  bench "combined-all-on-again" "$C" "LLAMA_NO_MOE_DOWN_RESCALE=tg" 0,1,2,3,4,5
} | tee "$OUT/ablations.md"
echo Q117_DONE
