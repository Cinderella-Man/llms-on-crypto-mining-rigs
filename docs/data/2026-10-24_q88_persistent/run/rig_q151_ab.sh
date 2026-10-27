#!/bin/bash
# Q151 A/B on the rig: the Pascal decode attention kernel (GGML_CUDA_FATTN_PASCAL=1, the default on compute 6.1) against
# the stock TILE kernel (=0) in one binary: tg128 at depth 0 and at 32k, Flint K2h-r16 on nine cards, off and on
# alternating twice (warm clocks: each llama-bench process runs both depths). Works with the Q151 tarball (8d30766c2) and
# with the rc tarball, which carries the kernel behind the same switch.
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] rig_q151_ab.sh
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
T=${T:?T=<tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
{
echo "== Q151 A/B, $T, K2h-r16, 9 cards, tg128 at depth 0 and 32768 $(date +%T)"
for round in 1 2; do
  for s in 0 1; do
    echo "-- GGML_CUDA_FATTN_PASCAL=$s, round $round $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E GGML_CUDA_FATTN_PASCAL=$s $B/llama-bench -m $K2H -ngl 99 -fa on -ub 512 -p 0 -n 128 -d 0,32768 -r 2 -o md 2>&1" \
      | grep -E 'tg128|error|fail|out of memory'
  done
done
echo Q151_AB_DONE
} | tee -a "$OUT/rig_q151_ab.md"
