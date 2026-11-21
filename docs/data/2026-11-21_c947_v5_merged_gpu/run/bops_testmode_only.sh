#!/usr/bin/env bash
# C947 (Q1687): the fourth share of run.sh's bops step. run.sh splits test-backend-ops by the operator names of its
# support listing, but test mode also runs fused cases (MUL_MAT_VEC_FUSION, TOPK_MOE, MUL_MAT_GLU, RMS_NORM_MUL_ROPE and
# 39 more) whose names the support listing does not print. ops-testmode-only.txt lists those 43 names: every operator
# name in a full `test -b CUDA0 -p "^(?!.*dup=1)"` run of the same test file (pull request 4's, 26,645 cases) that the
# support listing lacks. Started by hand on the third card next to run.sh's last share, writing into run.sh's bops folder.
# Env: WORK, NEW (as run.sh), MOUNTS, DOCKER_ARGS, CARD (2), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04)
set -uo pipefail
: "${WORK:?}" "${NEW:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CARD=${CARD:-2}
OPS=$(paste -sd, "$(dirname "$0")/ops-testmode-only.txt")
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
o=$WORK/bops
# shellcheck disable=SC2086
docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" -v "$WORK:$WORK" -e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 \
    --gpus "\"device=$CARD\"" "$IMG" \
    bash -c "nice -n 19 timeout 10800 $NEW/test-backend-ops test -b CUDA0 -o '$OPS' -p '^(?!.*dup=1)' > $o/fusion.log 2>&1; echo fusion rc=\$? >> $o/rc.txt"
date -Is > "$o/DONE-fusion"
