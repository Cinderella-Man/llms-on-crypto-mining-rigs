#!/bin/bash
# Q150 on the rig: pascal-p106-v2-next with Q150 (embedding lookup on card 1, packed uploads and one synchronize sweep in
# plain decoding, K-last attention order so the fused K cache write fires), all on by default on compute 6.1. One binary,
# defaults against the off switches (LLAMA_INPUT_ON_GPU=0 LLAMA_INPUT_UPLOAD=0: stock input layer, uploads and
# synchronizes; the attention order has no switch), plus the previous v2-next tarball (62fa5fc3b) as the reference.
#   0. the defaults took effect (log lines), and logits byte for byte: defaults against off switches, K2h-r16, 6 cards
#   1. llama-bench pp512 then tg128 in one process (warm clocks), 9 cards, K2h-r16 and UD: v2-next, Q150 off, Q150 on,
#      alternating twice
#   2. llama-pipespec plain against pipelined at depth 3 (bench/pipespec/rig_run.sh), K2h-r16, 9 cards
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] [STEPS='0 1 2'] rig_q150.sh
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<Q150 tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
BN=/opt/llama-builds/llama.cpp-20261024-62fa5fc3b-pascal-p106-v2-next/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
OFF="LLAMA_INPUT_ON_GPU=0 LLAMA_INPUT_UPLOAD=0"
STEPS=${STEPS:-0 1 2}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
if has 0; then
  echo "== 0. defaults on compute 6.1, and logits: defaults against the off switches, K2h-r16, 6 cards, -ub 1 $(date +%T)"
  for arm in "on|" "off|$OFF"; do
    IFS='|' read -r tag envs <<< "$arm"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E $envs $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q150-$tag.kld -lv 3 2>&1" \
      | grep -E "input layer assigned|Final estimate" | sed "s/^/$tag: /"
  done
  ssh "$RIG" "cmp /tmp/q150-on.kld /tmp/q150-off.kld && echo 'logits: defaults identical to off switches'; rm -f /tmp/q150-on.kld /tmp/q150-off.kld"
fi
if has 1; then
  for round in 1 2; do
    for arm in "K2h-r16 v2-next|$K2H|$BN|" "K2h-r16 Q150 off|$K2H|$B|$OFF" "K2h-r16 Q150 on|$K2H|$B|" \
               "UD v2-next|$UD|$BN|" "UD Q150 off|$UD|$B|$OFF" "UD Q150 on|$UD|$B|"; do
      IFS='|' read -r label m b envs <<< "$arm"
      echo "== 1. $label, 9 cards, round $round $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E $envs $b/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 -o md 2>&1" | grep -E 'pp512|tg128|error|fail'
    done
  done
fi
if has 2; then
  ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  echo "== 2. llama-pipespec plain against pipelined, K2h-r16, 9 cards, depths 1-3 $(date +%T)"
  ssh "$RIG" "env BIN=$B MODEL=$K2H TAG=q150-k2h-9c GPUS=0,1,2,3,4,5,6,7,8 ENVS=$E DEPTHS='1 2 3' N=256 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/q150-k2h-9c.out 2>&1; tail -30 /tmp/pipespec/q150-k2h-9c.out"
  ssh "$RIG" "tar czf - -C /tmp pipespec" | tar xzf - -C "$OUT"
fi
echo Q150_DONE
} | tee -a "$OUT/rig_q150.md"
