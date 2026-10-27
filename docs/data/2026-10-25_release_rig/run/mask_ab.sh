#!/usr/bin/env bash
# Decode with the card-built attention mask off (the default of the release build 37e717ff6) against on
# (LLAMA_GPU_KQ_MASK=1), the same settings, alternated (2026-10-25): Flint-Pascal-Q5 at short context and after 32k
# tokens, Flint Q8_0 at short context, nine cards. Run with nothing else on the rig.
# Env: RIG (ssh alias), BIN (the build's bin folder on the rig), K2H, Q8 (GGUF paths on the rig), OUT (folder for mask_ab.md).
set -uo pipefail
RIG=${RIG:-rig}
T=${T:-llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4}
B=${BIN:-/opt/llama-builds/$T/bin}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
O=${OUT:-.}
mkdir -p "$O"
run() { # <label> <model> <env or -> <args>
  local label=$1 m=$2 e=$3; shift 3
  [ "$e" = - ] && e=""
  echo "== $label $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $e $B/llama-bench -m $m -ngl 99 -fa on $* -o md 2>&1" \
    | grep -E '^\| .*(pp|tg)[0-9]+|error|failed|out of memory' | sed "s/^/$label | /"
}
{
echo "# Card-built mask off against on, decode, $T, $(date -Is)"
run "K2h mask off (1)" $K2H - -p 512 -n 128 -r 3
run "K2h mask on (1)" $K2H LLAMA_GPU_KQ_MASK=1 -p 512 -n 128 -r 3
run "K2h mask off (2)" $K2H - -p 512 -n 128 -r 3
run "K2h mask on (2)" $K2H LLAMA_GPU_KQ_MASK=1 -p 512 -n 128 -r 3
run "K2h 32k mask off" $K2H - -p 0 -n 64 -d 32768 -r 2
run "K2h 32k mask on" $K2H LLAMA_GPU_KQ_MASK=1 -p 0 -n 64 -d 32768 -r 2
run "Q8_0 mask off" $Q8 - -p 512 -n 128 -r 3 -ts 6/6/5/5/6/5/5/6/6
run "Q8_0 mask on" $Q8 LLAMA_GPU_KQ_MASK=1 -p 512 -n 128 -r 3 -ts 6/6/5/5/6/5/5/6/6
echo "== done $(date +%T)"
} | tee -a "$O/mask_ab.md"
