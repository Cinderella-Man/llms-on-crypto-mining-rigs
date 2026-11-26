#!/usr/bin/env bash
# C1179's whole sequence on one box with a CUDA GPU, natively (no Docker). WORK/src is a clone of the fork holding the
# branch and its base commit; the script makes WORK/build-a, WORK/build-b and WORK/logs.
#
#   WORK=<folder> CUDA=/usr/local/cuda BASE=2d839da1d BRANCH=<branch commit> bash all.sh
#
# 0. the branch's changed CUDA objects compiled alone first (build B), so a compile error shows in minutes;
# 1. build A (sm_70 + compute-6.1 PTX) of the base commit, and the digest of every ggml-cuda object's code;
# 2. the branch built into the same folder (only what the change touches is rebuilt), and its digest: every object
#    but the ones the change is in must keep its machine code and PTX;
# 3. build B (compute-6.1 PTX only, MMQ forced) of the branch, while build A's checks run on the GPU;
# 4. build B's checks.
set -euo pipefail
: "${WORK:?}" "${BASE:?}" "${BRANCH:?}"
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
export CUDA=${CUDA:-/usr/local/cuda}
export CCACHE_DIR=${CCACHE_DIR:-$WORK/ccache}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$WORK/nvcache}
export CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-4294967296}
export TMPDIR=${TMPDIR:-$WORK/tmp}
export NSYS=${NSYS:-nsys}
src=$WORK/src
logs=$WORK/logs
mkdir -p "$logs" "$CCACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
step() { echo "$(date '+%F %T') $*" | tee -a "$logs/steps.log"; }
objs=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir

step "0 the branch's changed objects alone (build B)"
git -C "$src" checkout -q "$BRANCH"
SRC=$src BUILD=$WORK/build-b ARCHS=61-virtual FORCE_MMQ=1 \
    TARGETS="$objs/template-instances/mmq-instance-iq1_m.cu.o $objs/mmq.cu.o tests/CMakeFiles/test-backend-ops.dir/test-backend-ops.cpp.o" \
    bash "$here/build.sh" > "$logs/build0.log" 2>&1

step "1 build A of the base commit"
git -C "$src" checkout -q "$BASE"
SRC=$src BUILD=$WORK/build-a ARCHS="61-virtual;70" bash "$here/build.sh" > "$logs/build1.log" 2>&1
BUILD=$WORK/build-a bash "$here/code_digest.sh" > "$logs/digest_base.tsv"

step "2 build A of the branch (incremental)"
git -C "$src" checkout -q "$BRANCH"
SRC=$src BUILD=$WORK/build-a ARCHS="61-virtual;70" bash "$here/build.sh" > "$logs/build2.log" 2>&1
BUILD=$WORK/build-a bash "$here/code_digest.sh" > "$logs/digest_branch.tsv"
{
    echo "# objects whose machine code or PTX differ (object, base sass, base ptx, branch sass, branch ptx)"
    join -t $'\t' "$logs/digest_base.tsv" "$logs/digest_branch.tsv" | awk -F'\t' '$2 != $4 || $3 != $5'
    echo "# objects only in the base build"
    join -t $'\t' -v 1 "$logs/digest_base.tsv" "$logs/digest_branch.tsv"
    echo "# objects only in the branch build"
    join -t $'\t' -v 2 "$logs/digest_base.tsv" "$logs/digest_branch.tsv"
} > "$logs/digest_diff.tsv"

step "3 build B of the branch, build A's checks meanwhile"
SRC=$src BUILD=$WORK/build-b ARCHS=61-virtual FORCE_MMQ=1 bash "$here/build.sh" > "$logs/build3.log" 2>&1 &
build_b=$!
BIN_A=$WORK/build-a/bin BIN_B=$WORK/build-b/bin OUT=$logs/checks STEPS="a_all a_on" bash "$here/checks.sh"
wait "$build_b"

step "4 build B's checks, then the kernel traces"
BIN_A=$WORK/build-a/bin BIN_B=$WORK/build-b/bin OUT=$logs/checks STEPS="b_all b_on trace" bash "$here/checks.sh"
step "done"
