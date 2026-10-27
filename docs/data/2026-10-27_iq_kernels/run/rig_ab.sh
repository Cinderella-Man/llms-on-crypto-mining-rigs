#!/usr/bin/env bash
# The IQ2_S / IQ3_XXS Pascal kernels on the rig, against llama.cpp's own kernels in the same binary
# (GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs), with a model whose experts use those types. Run from a machine that
# reaches both the rig (RIG) and the build machine (CLIENT) over ssh, with nothing else on the rig:
#   1. install: each build's tarball from $CLIENT:$WORK/out to $BUILDS on the rig, md5 checked, unpacked
#   2. probe: scripts/serve_probe.sh with each build, and with the first build's new kernels off
#   3. profile: scripts/serve_probe.sh with QUICK=1 under GGML_CUDA_PROFILE, new kernels on and off (decode time per
#      launch group, summarized by scripts/prof_summary.py)
# Environment: RIG, CLIENT, WORK, BUILDS, MODEL, OUT (folder), TS (the split), SPECS ("name:md5 ..."), STEPS ("1 2 3").
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
RIG=${RIG:-rig}
CLIENT=${CLIENT:?ssh alias of the build machine}
WORK=${WORK:?build folder on the build machine}
BUILDS=${BUILDS:?builds folder on the rig}
MODEL=${MODEL:?the GGUF on the rig}
OUT=${OUT:?output folder}
TS=${TS:?-ts split}
SPECS=${SPECS:?"name:md5 ..." of the builds}
STEPS=${STEPS:-1 2 3}
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$OUT"
read -ra SP <<< "$SPECS"
FIRST=${SP[0]%%:*}
SKIP=GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs
L="-b 2048 -ub 512 -ts $TS -bs -fit off"

if has 1; then
  for s in "${SP[@]}"; do
    n=${s%%:*}; m=${s#*:}
    echo "== install $n $(date +%T)"
    ssh "$CLIENT" "cat $WORK/out/$n.tar.gz" | ssh "$RIG" "cat > $BUILDS/$n.tar.gz"
    ssh "$RIG" "cd $BUILDS && echo '$m  $n.tar.gz' | md5sum -c && rm -rf $n && tar xzf $n.tar.gz && rm $n.tar.gz" || exit 1
  done
fi
if has 2; then
  C=""
  for s in "${SP[@]}"; do C="$C${C:+;}iq-${s%%:*}|${s%%:*}|49152|$L|"; done
  C="$C;iq-off|$FIRST|49152|$L|$SKIP"
  RIG=$RIG MODEL=$MODEL BUILDS=$BUILDS OUT=$OUT/serve_probe.md CONFIGS="$C" bash "$ROOT/scripts/serve_probe.sh"
fi
if has 3; then
  P=GGML_CUDA_PROFILE=/tmp/ornith-prof.csv
  QUICK=1 RIG=$RIG MODEL=$MODEL BUILDS=$BUILDS OUT=$OUT/serve_probe.md \
    CONFIGS="iq-prof-on|$FIRST|49152|$L|$P;iq-prof-off|$FIRST|49152|$L|$P $SKIP" bash "$ROOT/scripts/serve_probe.sh"
fi
echo "== done $(date +%T)"
