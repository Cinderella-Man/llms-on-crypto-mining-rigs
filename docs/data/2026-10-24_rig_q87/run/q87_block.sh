#!/bin/bash
# Q87 block on the rig, one run at a time: identity check, then plain against pipelined (rig_run.sh) on 6 and 9 cards,
# then the host loop (rig_hostloop.sh, Q120/Q121). K2h-r16 (the fastest file, Q125), the pipespec2 build (pipespec +
# Q120/Q121), rescale off as for the custom mixes. Results copied back to $OUT after each step. Ran 2026-10-24.
# Usage: [RIG=rig] [OUT=dir] q87_block.sh   (the build and the model at the paths below on the rig)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
B=/opt/llama-builds/llama.cpp-20261024-f249104ab-pascal-p106-v2-pipespec2/bin
M=/mnt/models/flint/flint-q113-K2h-r16.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=1"
ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
back() { ssh "$RIG" "tar czf - -C /tmp pipespec" | tar xzf - -C "$OUT"; }
echo "== 1 identity, host sampling $(date +%T)"
ssh "$RIG" "$E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=$M ARGS='-ngl 99 -fa on -c 4096' N=128 DEPTHS='1 3 5' MISSES='0 0.3' OUT=/tmp/pipespec/check-k2h bash /tmp/pipespec/check_identity.sh 2>&1 | tail -15"
echo "== 1b identity, sampling on the last card $(date +%T)"
ssh "$RIG" "$E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=$M ARGS='-ngl 99 -fa on -c 4096 --pipe-backend-greedy' N=128 DEPTHS='1 3 5' MISSES='0 0.3' OUT=/tmp/pipespec/check-k2h-bg bash /tmp/pipespec/check_identity.sh 2>&1 | tail -15"
back
echo "== 2 plain vs pipelined, 6 cards $(date +%T)"
ssh "$RIG" "env BIN=$B MODEL=$M TAG=k2h-6c GPUS=0,1,2,3,4,5 ENVS=$E bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/k2h-6c.out 2>&1; tail -40 /tmp/pipespec/k2h-6c.out"
back
echo "== 3 plain vs pipelined, 9 cards $(date +%T)"
ssh "$RIG" "env BIN=$B MODEL=$M TAG=k2h-9c GPUS=0,1,2,3,4,5,6,7,8 ENVS=$E DEPTHS='1 2 3 4 5 6 8' bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/k2h-9c.out 2>&1; tail -40 /tmp/pipespec/k2h-9c.out"
back
echo "== 4 host loop (Q120/Q121), 9 cards $(date +%T)"
ssh "$RIG" "env BIN=$B MODEL=$M TAG=k2h-hostloop GPUS=0,1,2,3,4,5,6,7,8 ENVS=$E NS='1 4 8' bash /tmp/pipespec/rig_hostloop.sh > /tmp/pipespec/k2h-hostloop.out 2>&1; tail -40 /tmp/pipespec/k2h-hostloop.out"
back
echo "Q87_BLOCK_DONE $(date +%T)"
