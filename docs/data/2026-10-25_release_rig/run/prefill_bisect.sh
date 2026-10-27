#!/usr/bin/env bash
# Where prompt reading collapsed (2026-10-25): the same llama-bench command (Flint-Pascal-Q5, nine cards, one
# 8,192-token prompt in one batch: -p 8192 -n 0 -b 8192 -ub 512 -r 1) on every build in history order, stock first and
# last for drift. Run with nothing else on the rig; every build must already be unpacked there.
# Env: RIG (ssh alias), K2H (GGUF path on the rig), BUILDS (build folders on the rig, in order), OUT (folder for
# prefill_bisect.md). The 2026-10-25 run stopped after rc1 594d20e08, where the drop appeared.
set -uo pipefail
RIG=${RIG:-rig}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
R=/opt/llama-builds
BUILDS=${BUILDS:-$R/llama.cpp-20261022-d77256999 $R/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined $R/llama.cpp-20261024-62fa5fc3b-pascal-p106-v2-next $R/llama.cpp-20261024-d379da309-pascal-p106-v2-next $R/llama.cpp-20261024-594d20e08-pascal-p106-v2-rc $R/llama.cpp-20261025-7e6ec762f-pascal-p106-v2-rc3 $R/llama.cpp-20261025-460160e93-pascal-p106-v2-rc4 $R/llama.cpp-20261025-2d53179aa-pascal-p106-v2-rc4 $R/llama.cpp-20261025-bff28aba6-pascal-p106-v2-rc4 $R/llama.cpp-20261022-d77256999}
O=${OUT:-.}
mkdir -p "$O"
{
echo "# Prompt reading per build, K2h-r16, 9 cards, llama-bench -p 8192 -n 0 -b 8192 -ub 512 -r 1, $(date -Is)"
for b in $BUILDS; do
  echo "== ${b##*/} $(date +%T)"
  ssh "$RIG" "test -x $b/bin/llama-bench || { echo 'not on the rig'; exit 0; }
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 LLAMA_NO_MOE_DOWN_RESCALE=tg $b/bin/llama-bench -m $K2H -ngl 99 -fa on -p 8192 -n 0 -b 8192 -ub 512 -r 1 -o md 2>&1" \
    | grep -E '^\| .*pp[0-9]+|error|failed|not on the rig|out of memory' | sed "s|^|${b##*/} \| |"
done
echo "== done $(date +%T)"
} | tee -a "$O/prefill_bisect.md"
