#!/bin/bash
# Q152 on the rig: the attention block of Flint's 35 RoPE layers in two launches instead of six (GGML_CUDA_ATTN_BLOCK, on
# by default on compute 6.1, 0 off): the Q/K/V products in one Pascal launch, then Q's and K's norms, RoPE and both cache
# writes in one epilogue. One binary, on against off.
#   0. logits byte for byte, on against off, K2h-r16, 6 cards, -ub 1; launch groups per token (GGML_CUDA_PROFILE, counts
#      only: 35 x 4 = 140 fewer expected)
#   1. llama-bench pp512 then tg128 in one process, 9 cards, K2h-r16 and UD: off, on, alternating twice
#   r. (bdac267bb tarball only) the compute-6.1 rescale default: logits with LLAMA_NO_MOE_DOWN_RESCALE unset against =tg,
#      byte for byte, and the persistent MoE kernel's first-use log line with the variable unset
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] [STEPS='0 1' or 'r'] rig_q152.sh
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<Q152 tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
STEPS=${STEPS:-0 1}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
if has 0; then
  echo "== 0. logits, attention block on against off, K2h-r16, 6 cards, -ub 1 $(date +%T)"
  for s in 1 0; do
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E GGML_CUDA_ATTN_BLOCK=$s $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q152-$s.kld 2>&1" \
      | grep -E "Final estimate" | sed "s/^/GGML_CUDA_ATTN_BLOCK=$s: /"
  done
  ssh "$RIG" "cmp /tmp/q152-1.kld /tmp/q152-0.kld && echo 'logits: on identical to off'; rm -f /tmp/q152-1.kld /tmp/q152-0.kld"
  echo "-- launch groups per token, 9 cards (GGML_CUDA_PROFILE; counts, not times)"
  scp -q $R/scripts/cuda_profile_summary.py "$RIG":/tmp/
  for s in 0 1; do
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E GGML_CUDA_ATTN_BLOCK=$s GGML_CUDA_PROFILE=/tmp/q152-prof-$s.csv $B/llama-bench -m $K2H -ngl 99 -fa on -p 0 -n 32 -r 1 > /dev/null 2>&1; python3 /tmp/cuda_profile_summary.py /tmp/q152-prof-$s.csv --tokens 24 --by op 2>&1 | head -30" \
      | sed "s/^/ATTN_BLOCK=$s: /"
  done
  ssh "$RIG" "rm -f /tmp/q152-prof-*.csv /tmp/cuda_profile_summary.py"
fi
if has r; then
  echo "== r. the rescale default on compute 6.1: LLAMA_NO_MOE_DOWN_RESCALE unset against =tg, K2h-r16, 6 cards, -ub 1 $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q152-unset.kld -lv 3 2>&1" \
    | grep -E "Final estimate|persistent MoE|rescales" | sort | uniq -c | sed "s/^/unset: /"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q152-tg.kld 2>&1" \
    | grep -E "Final estimate" | sed "s/^/tg: /"
  ssh "$RIG" "cmp /tmp/q152-unset.kld /tmp/q152-tg.kld && echo 'logits: unset identical to tg'; rm -f /tmp/q152-unset.kld /tmp/q152-tg.kld"
fi
if has 1; then
  for round in 1 2; do
    for arm in "K2h-r16 off|$K2H|0" "K2h-r16 on|$K2H|1" "UD off|$UD|0" "UD on|$UD|1"; do
      IFS='|' read -r label m s <<< "$arm"
      echo "== 1. $label, 9 cards, round $round $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E GGML_CUDA_ATTN_BLOCK=$s $B/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 -o md 2>&1" | grep -E 'pp512|tg128|error|fail'
    done
  done
fi
echo Q152_DONE
} | tee -a "$OUT/rig_q152.md"
