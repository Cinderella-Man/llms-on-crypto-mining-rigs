#!/usr/bin/env bash
# SASS identity check for a kernel change: the sm_61 SASS of every Pascal matrix-vector kernel (mul_mat_vec_q_pascal)
# in two llama.cpp builds' test-backend-ops, one file per kernel with the instruction offsets stripped. Every kernel the
# two builds share must be identical; kernels only in the new build should all be of the new type. CPU only (cuobjdump
# in the CUDA container). Used for Q149 (2026-10-24): the Q6_K build against the Q4_K build, 173/173 shared kernels
# identical and 15 new Q4_K ones.
#
# Usage: WORK=<dir> sass_cmp.sh <old build dir> <new build dir> [type tag of the new kernels, default ggml_type12E = Q4_K]
# Writes $WORK/sass/old-<kernel>.sass and $WORK/sass/new-<kernel>.sass (forkloops.py reads them) and prints the tally.
set -euo pipefail
WORK=${WORK:?WORK=<work dir>}
OLD=$(realpath "$1"); NEW=$(realpath "$2"); NEWTYPE=${3:-ggml_type12E}
mkdir -p "$WORK/sass"
docker run --rm -v "$WORK:$WORK" -v "$OLD:/old:ro" -v "$NEW:/new:ro" -e S="$WORK/sass" -e NEWTYPE="$NEWTYPE" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
for b in old:/old new:/new; do
    tag=${b%%:*}; dir=${b#*:}
    rm -f $S/$tag-*.sass
    cuobjdump -sass -arch sm_61 $dir/bin/test-backend-ops | awk -v out=$S/$tag "
        /Function : / { f = \$3; keep = (f ~ /mul_mat_vec_q_pascal/) ; if (keep) { n++; file = out \"-\" f \".sass\" } }
        keep { sub(/\\/\\*[0-9a-f]+\\*\\//, \"\"); print > file }"
done
cd $S
same=0; diff=0; new=0
for f in new-*.sass; do
    g=old-${f#new-}
    if [ ! -f $g ]; then new=$((new+1)); case $f in *${NEWTYPE}*) ;; *) echo "new kernel of another type: $f" ;; esac
    elif cmp -s $f $g; then same=$((same+1)); else diff=$((diff+1)); echo "differs: $f"; fi
done
echo "shared kernels identical: $same, differing: $diff, only in the new build: $new, only in the old build: $(for g in old-*.sass; do [ -f new-${g#old-} ] || echo x; done | wc -l)"
'
