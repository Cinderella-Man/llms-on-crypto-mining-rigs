#!/usr/bin/env bash
# What fits at the release context and how fast it reads a prompt, per batch / micro-batch size, on the release build
# 37e717ff6 (2026-10-26): llama-batched-bench allocates the whole context's KV cache, reads a 16k-token prompt, then
# generates 64 tokens (-npl 1), nine cards. Run with nothing else on the rig. Env: RIG (ssh alias), BIN (the build's
# bin folder on the rig), MODEL (q8 or k2h), Q8, K2H (GGUF paths), TS_K2H, CTX, NPP, GRID ("b/ub ..."), ENVS (variables
# set for the run, e.g. LLAMA_GPU_KQ_MASK=1 for the card-built mask), OUT (folder for fit_prefill-<model>.md).
# fit_prefill.tsv's context and mask rows are this command with CTX=32768 / 20480 / 131072, GRID=<b>/128 or <b>/512,
# and ENVS for the mask.
set -uo pipefail
RIG=${RIG:-rig}
T=${T:-llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4}
B=${BIN:-/opt/llama-builds/$T/bin}
MODEL=${MODEL:-q8}
case $MODEL in
  q8)  M=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}; TS="-ts 6,6,5,5,6,5,5,6,6" ;;
  k2h) M=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}; TS="${TS_K2H:-}" ;;
  *) echo "MODEL=q8 or k2h"; exit 1 ;;
esac
CTX=${CTX:-262144}
NPP=${NPP:-16384}
GRID=${GRID:-512/128 2048/256 2048/512 8192/512}
ENVS=${ENVS:-}
O=${OUT:-.}
mkdir -p "$O"
{
echo "# Fit and prompt speed at -c $CTX, $MODEL ($TS), $T, -npp $NPP -ntg 64, $(date -Is)"
for bu in $GRID; do
  b=${bu%/*}; u=${bu#*/}
  echo "== $MODEL -b $b -ub $u $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $ENVS $B/llama-batched-bench -m $M -ngl 99 -fa on -c $CTX -b $b -ub $u $TS -npp $NPP -ntg 64 -npl 1 2>&1" \
    | grep -E '^\|  *[0-9]|out of memory|failed|error|unable' | sed "s/^/$MODEL b$b ub$u | /"
done
echo "== done $(date +%T)"
} | tee -a "$O/fit_prefill-$MODEL.md"
