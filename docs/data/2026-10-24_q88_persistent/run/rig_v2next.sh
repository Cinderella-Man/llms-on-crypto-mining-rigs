#!/bin/bash
# The unified v2 build on the rig: branch pascal-p106-v2-next = the combined build + the persistent MoE kernel (on by
# default on compute 6.1) + the pipelined work (packed uploads, device-fed token, kv-cache head fix, llama-server
# --pipe-depth). Against the combined build (b3052a4f8) on the same cards, K2h-r16 and UD-Q4_K_XL, 6 and 9 cards.
#   0. correctness: the persistent kernel's self-check on 6 cards (-ub 1: 0 differed at every stage), and the pipelined
#      decoder's identity check (rig_run.sh checks every pipelined job against plain decoding)
#   1. llama-bench pp512 then tg128 in one process (warm clocks), combined and v2-next alternating twice
#   2. pipelined decoding with llama-pipespec (bench/pipespec/rig_run.sh), depths 1-4, 9 cards
#   3. llama-server with --pipe-depth (bench/pipespec/check_server.sh): same outputs as without guesses, with speeds
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] [STEPS='0 1 2 3'] rig_v2next.sh. Takes several hours: run steps one
# at a time with STEPS='0 1' etc.
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<v2-next tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
BC=/opt/llama-builds/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
STEPS=${STEPS:-0 1 2 3}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
if has 0; then
  echo "== 0. persistent kernel self-check, K2h-r16, 6 cards, one token per step $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $E GGML_CUDA_PERSISTENT_MOE_CHECK=1 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 1 -b 128 -ub 1 -ngl 99 -fa on -lv 4 2>&1" \
      | grep -oE "CUDA[0-9]+: persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed" \
      | sed -E 's/persistent MoE kernel ran //; s/ compared with the per-op path,//' | sort | uniq -c
fi
if has 1; then
  for cards in 0,1,2,3,4,5 0,1,2,3,4,5,6,7,8; do
    n=$(echo $cards | tr ',' '\n' | wc -l)
    for round in 1 2; do
      for arm in "K2h-r16 combined|$K2H|$BC" "K2h-r16 v2-next|$K2H|$B" "UD combined|$UD|$BC" "UD v2-next|$UD|$B"; do
        IFS='|' read -r label m b <<< "$arm"
        echo "== 1. $label, ${n} cards, round $round $(date +%T)"
        ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards $E $b/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 -o md 2>&1" | grep -E 'pp512|tg128|error|fail'
      done
    done
  done
fi
if has 2; then
  ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    echo "== 2. llama-pipespec plain against pipelined, $tag, 9 cards, depths 1-4 $(date +%T)"
    ssh "$RIG" "env BIN=$B MODEL=$m TAG=v2next-$tag-9c GPUS=0,1,2,3,4,5,6,7,8 ENVS=$E DEPTHS='1 2 3 4' N=256 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/v2next-$tag-9c.out 2>&1; tail -30 /tmp/pipespec/v2next-$tag-9c.out"
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec" | tar xzf - -C "$OUT"
fi
if has 3; then
  ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_server"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    echo "== 3. llama-server --pipe-depth against no guesses, $tag, 9 cards $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $E BIN=$B MODEL=$m ARGS='-ngl 99 -fa on -c 8192 -ub 512' DEPTHS='3' GUESSES='cont' KEEP_SPEED=1 OUT=/tmp/pipespec_server/$tag bash /tmp/pipespec/check_server.sh 2>&1 | tail -20"
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec_server" | tar xzf - -C "$OUT"
fi
echo V2NEXT_DONE
} | tee -a "$OUT/rig_v2next.md"
