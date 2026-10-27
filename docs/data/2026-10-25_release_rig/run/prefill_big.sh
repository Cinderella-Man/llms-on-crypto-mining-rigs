#!/usr/bin/env bash
# Prompt reading with whole-prompt batches (-b 8192 -ub 512) on the release build (2026-10-25): 4 scheduler copies
# against the 8-copy build (Q157), and the Pascal prefill-attention kernel on against off (GGML_CUDA_FATTN_TILE_PASCAL=0,
# Q163). Flint-Pascal-Q5 at 8k and 32k, Flint Q8_0 at 8k, nine cards, llama-bench -n 0 -r 2. Run with nothing else on
# the rig. Env: RIG (ssh alias), T (build folder name under /opt/llama-builds; T-sched8 is the 8-copy build), K2H, Q8
# (GGUF paths on the rig), OUT (folder for prefill_big.md).
set -uo pipefail
RIG=${RIG:-rig}
T=${T:-llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
O=${OUT:-.}
mkdir -p "$O"
run() { # <label> <build> <model> <env or -> <args>
  local label=$1 b=$2 m=$3 e=$4; shift 4
  [ "$e" = - ] && e=""
  echo "== $label $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $e /opt/llama-builds/$b/bin/llama-bench -m $m -ngl 99 -fa on -n 0 -b 8192 -ub 512 -r 2 $* -o md 2>&1" \
    | grep -E '^\| .*pp[0-9]+|error|failed|out of memory' | sed "s/^/$label | /"
}
{
echo "# Prompt reading, whole-prompt batches, $T, $(date -Is)"
run "K2h-r16 4 copies" $T $K2H - -p 8192,32768
run "K2h-r16 4 copies, prompt attention off" $T $K2H GGML_CUDA_FATTN_TILE_PASCAL=0 -p 8192,32768
run "K2h-r16 8 copies" $T-sched8 $K2H - -p 8192,32768
run "Q8_0 4 copies" $T $Q8 - -p 8192 -ts 6/6/5/5/6/5/5/6/6
run "Q8_0 8 copies" $T-sched8 $Q8 - -p 8192 -ts 6/6/5/5/6/5/5/6/6
echo "== done $(date +%T)"
} | tee -a "$O/prefill_big.md"
