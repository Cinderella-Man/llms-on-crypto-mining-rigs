#!/bin/bash
# Q151 on the rig: the Pascal decode attention kernel (GGML_CUDA_FATTN_PASCAL, on by default on compute 6.1; 0 = the
# stock TILE kernel) in one binary, on against off.
#   0. kernel only, one card: test-backend-ops perf on Flint's one-token attention (KV 256 to 131072), off and on
#   1. model output, K2h-r16, 6 cards: 2 x 128 tokens at -ub 1, KLD of on against off (not bit-identical: a different
#      summation order; Flint's own noise floor is KLD ~0.06 per pascal_mmvq_integration.md section 9)
#   2. llama-bench tg128 at depth 0 and 8192, 9 cards, K2h-r16 and UD: off, on, alternating twice
#   3. card 0's attention per layer (GGML_CUDA_PROFILE, depths 0 and 4096), off and on: FLASH_ATTN_EXT per call
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] [STEPS='0 1 2 3'] rig_q151.sh
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
STEPS=${STEPS:-0 1 2 3}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
if has 0; then
  for s in 0 1; do
    echo "== 0. test-backend-ops perf, Flint one-token attention, card 0, GGML_CUDA_FATTN_PASCAL=$s $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 GGML_CUDA_FATTN_PASCAL=$s $B/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'hsk=128,hsv=128,nh=4,nr23=\[8,1\],kv=[0-9]+,nb=1,mask=1,sinks=1' 2>&1" \
      | grep -E "FLASH_ATTN_EXT" | sed -E 's/hsk=128,hsv=128,nh=4,nr23=\[8,1\],//; s/,max_bias.*permute=\[0,2,1,3\]//'
  done
fi
if has 1; then
  echo "== 1. K2h-r16 logits, 6 cards, -ub 1: the Pascal kernel against TILE $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E GGML_CUDA_FATTN_PASCAL=0 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q151-off.kld 2>&1 | grep -E 'Final estimate'; \
           CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E GGML_CUDA_FATTN_PASCAL=1 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on --kl-divergence-base /tmp/q151-off.kld --kl-divergence 2>&1 | grep -E 'Mean PPL|Mean    KLD|Maximum KLD|Same top p'; rm -f /tmp/q151-off.kld"
fi
if has 2; then
  for round in 1 2; do
    for arm in "K2h-r16 off|$K2H|0" "K2h-r16 on|$K2H|1" "UD off|$UD|0" "UD on|$UD|1"; do
      IFS='|' read -r label m s <<< "$arm"
      echo "== 2. $label, 9 cards, round $round $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E GGML_CUDA_FATTN_PASCAL=$s $B/llama-bench -m $m -ngl 99 -fa on -p 0 -n 128 -d 0,8192 -r 3 -o md 2>&1" | grep -E 'tg128|error|fail'
    done
  done
fi
if has 3; then
  scp -q $R/scripts/cuda_profile_summary.py "$RIG":/tmp/
  for d in 0 4096; do
    for s in 0 1; do
      echo "== 3. profile, K2h-r16, 9 cards, depth $d, GGML_CUDA_FATTN_PASCAL=$s, card 0's launch groups $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E GGML_CUDA_FATTN_PASCAL=$s GGML_CUDA_PROFILE=/tmp/q151-d$d-$s.csv $B/llama-bench -m $K2H -ngl 99 -fa on -ub 512 -p 0 -n 32 -r 1 -d $d -o md > /dev/null 2>&1; awk -F, 'NR == 1 || \$1 == 0' /tmp/q151-d$d-$s.csv > /tmp/q151-card0.csv; python3 /tmp/cuda_profile_summary.py /tmp/q151-card0.csv --tokens 24 --by op --top 12; rm -f /tmp/q151-d$d-$s.csv /tmp/q151-card0.csv"
    done
  done
  ssh "$RIG" "rm -f /tmp/cuda_profile_summary.py"
fi
echo Q151_DONE
} | tee -a "$OUT/rig_q151.md"
