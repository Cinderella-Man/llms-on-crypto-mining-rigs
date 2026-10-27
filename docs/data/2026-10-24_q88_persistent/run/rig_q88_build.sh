#!/bin/bash
# Q88 build-in: the persistent MoE layer kernel in the fork (branch pascal-p106-v2-persistent, on top of the combined
# build). One binary, switch off (GGML_CUDA_PERSISTENT_MOE=0, the combined build's path) against on (=1, compute 6.1).
# K2h-r16 is the target (all 47 MoE layers covered); UD-Q4_K_XL has 42 of its 47 layers covered from tarball 8de13900f
# on (its Q5_K/IQ4_XS kernels spill 16-76 bytes on sm_61, so the rig decides whether they gain). llama-bench runs pp512 before
# tg128 in one process (warm clocks, Q141); off and on alternate twice to expose order effects.
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] [SKIP_CHECK=1] rig_q88_build.sh
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
T=${T:?T=<tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
{
if [ "${SKIP_CHECK:-0}" != 1 ]; then
echo "== 0. correctness on the P106s (test-backend-ops MOE_LAYER needs ~1 GB of host memory per weight tensor: not on the rig) $(date +%T)"
echo "-- K2h-r16, 6 cards, one token per step, self-check per graph evaluation (count: device, layers, compared, differed)"
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E GGML_CUDA_PERSISTENT_MOE=1 GGML_CUDA_PERSISTENT_MOE_CHECK=1 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 1 -b 128 -ub 1 -ngl 99 -fa on -lv 4 2>&1" \
    | grep -oE "CUDA[0-9]+: persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed" \
    | sed -E 's/persistent MoE kernel ran //; s/ compared with the per-op path,//' | sort | uniq -c
fi
for cards in 0,1,2,3,4,5 0,1,2,3,4,5,6,7,8; do
  n=$(echo $cards | tr ',' '\n' | wc -l)
  for round in 1 2; do
    for arm in "K2h-r16 off|$K2H|0" "K2h-r16 on|$K2H|1" "UD off|$UD|0" "UD on|$UD|1"; do
      IFS='|' read -r label m s <<< "$arm"
      echo "== $label, ${n} cards, round $round $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards GGML_CUDA_PERSISTENT_MOE=$s $E $B/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 -o md 2>&1" \
          | grep -E 'pp512|tg128|error|fail'
    done
  done
done
echo Q88_BUILD_DONE
} | tee "$OUT/rig_q88_build.md"
