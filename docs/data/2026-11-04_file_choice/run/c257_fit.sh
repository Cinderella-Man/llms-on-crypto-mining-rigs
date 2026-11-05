#!/usr/bin/env bash
# C257 (Q813): llama-server's own fit projection for Swift IQ3_XXS's three 64k lines of ../fit.tsv (split_plan.py's
# plans), on header-only copies of the three shards (../../2026-11-04_fewer_hops/run/header_copies.sh, unpacked on the
# rig with tar -xSzf), on v3's base (v2.0, 37e717ff6) and on -ladder3 (7bc1552f4). Seconds per line, no weights load.
#   RIG=<ssh host of the rig> MODEL=<rig path of the IQ3_XXS copy's shard 1> DRAFT=<rig path of DFlash2's Q8_0 GGUF> \
#   BUILDS=<rig folder holding <build>/bin/llama-server> OUT=fit_probe.md bash c257_fit.sh
# Then the smallest free MiB per line: grep 'projected memory use' and the per-device lines of $OUT (../fit_probe_rig.tsv).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RIG:?}" "${MODEL:?}" "${DRAFT:?}" "${BUILDS:?}" "${OUT:?}"
BASES=${BASES:-llama.cpp-20261102-37e717ff6-pascal-p106-v3 llama.cpp-20261104-7bc1552f4-pascal-p106-v3-b12076-ladder3}
OT8='blk\.5\.ffn_down_exps=CUDA1,blk\.18\.ffn_up_exps=CUDA3,blk\.18\.ffn_down_exps=CUDA3,blk\.23\.ffn_down_exps=CUDA4,blk\.29\.ffn_up_exps=CUDA5,blk\.29\.ffn_down_exps=CUDA5,blk\.38\.ffn_down_exps=CUDA7,blk\.43\.ffn_down_exps=CUDA8'
OT9='blk\.12\.ffn_up_exps=CUDA2,blk\.12\.ffn_down_exps=CUDA2,blk\.18\.ffn_down_exps=CUDA3,blk\.29\.ffn_down_exps=CUDA5,blk\.34\.ffn_up_exps=CUDA6,blk\.34\.ffn_down_exps=CUDA6,blk\.44\.ffn_gate_exps=CUDA8,blk\.44\.ffn_up_exps=CUDA8,blk\.44\.ffn_down_exps=CUDA8'
DFA="-md $DRAFT --spec-type draft-dflash --spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99"
C=""
for b in $BASES; do
  s=${b%%-pascal-p106-v3*}; s=${s##*-}
  C="$C;whole-ub64-$s|$b|65536|-b 2048 -ub 64 -ts 6,6,7,5,5,5,5,5,5"
  C="$C;ot8-ub128-$s|$b|65536|-b 2048 -ub 128 -ts 6,6,7,5,6,4,5,5,5 -ot $OT8"
  C="$C;ot8-ub256-$s|$b|65536|-b 2048 -ub 256 -ts 6,6,7,5,6,4,5,5,5 -ot $OT8"
  C="$C;dflash2-ot9-ub256-$s|$b|65536|-b 512 -ub 256 -ts 6,7,6,5,6,5,4,6,4 -ot $OT9 $DFA"
done
RIG="$RIG" MODEL="$MODEL" BUILDS="$BUILDS" OUT="$OUT" KEEP=300 CONFIGS="${C#;}" \
  bash "$HERE/../../../../scripts/fit_probe.sh" > /dev/null
echo C257_FIT_DONE
