#!/usr/bin/env bash
# C1179: the branch's CUDA code against the base commit's, function by function, for every object of ggml-cuda, in
# build A (a V100's sm_70 machine code and the P106's compute-6.1 PTX). Run after all.sh's step 3 (nothing else may
# build from WORK/src meanwhile); WORK/src and WORK/build-a as all.sh leaves them.
#
#   WORK=<folder> CUDA=/usr/local/cuda BASE=2d839da1d BRANCH=<branch commit> bash code_check.sh
#
# 1. dumps the branch's objects (cuobjdump -sass and -ptx);
# 2. checks out BASE and rebuilds only the ggml-cuda objects into the same build folder (the compiler cache holds them
#    from all.sh's step 1), and dumps them;
# 3. checks out BRANCH, rebuilds the objects again (the build folder is the branch's again) and compares them
#    (code_compare.py). test-backend-ops is not relinked.
set -euo pipefail
: "${WORK:?}" "${BASE:?}" "${BRANCH:?}"
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA:-/usr/local/cuda}
export PATH="$CUDA/bin:$PATH"
export CCACHE_DIR=${CCACHE_DIR:-$WORK/ccache}
src=$WORK/src
build=$WORK/build-a
out=${OUT:-$WORK/logs/code}
rel=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
mkdir -p "$out"

# folder, then a pattern of objects to leave out (the branch's new object stays in the build folder at the base)
dump() {
    mkdir -p "$1"
    (cd "$build" && find "$rel" -name '*.o' | sort | grep -v "${2:-^$}") | while read -r obj; do
        name=${obj#"$rel"/}
        name=${name//\//_}
        cuobjdump -sass "$build/$obj" > "$1/$name.sass" 2> /dev/null || true
        cuobjdump -ptx "$build/$obj" > "$1/$name.ptx" 2> /dev/null || true
    done
}

dump "$out/branch"
# every object but the branch's new one, which the base has no rule for
mapfile -t targets < <(cd "$build" && find "$rel" -name '*.o' | sort | grep -v 'mmq-instance-iq1_m\.cu\.o$')

git -C "$src" checkout -q "$BASE"
nice -n 19 ninja -C "$build" -j "${JOBS:-8}" "${targets[@]}" > "$out/rebuild_base.log" 2>&1
dump "$out/base" 'mmq-instance-iq1_m\.cu\.o$'

git -C "$src" checkout -q "$BRANCH"
nice -n 19 ninja -C "$build" -j "${JOBS:-8}" "${targets[@]}" "$rel/template-instances/mmq-instance-iq1_m.cu.o" \
    > "$out/rebuild_branch.log" 2>&1

python3 -I "$here/code_compare.py" "$out/base" "$out/branch" "$out/compare" | tee "$out/compare_totals.txt"
