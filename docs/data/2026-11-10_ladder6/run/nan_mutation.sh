#!/usr/bin/env bash
# C662 (Q1150, Q1158): does the test case nan_clear catch the bug it was written for? The branch's tip with the fill in
# build_rs turned back into the multiply by zero (src/llama-graph.cpp only; the test case stays), a native CPU build of
# test-llama-context, run on the random recurrent model (mamba) the CPU checks wrote: nan_clear must FAIL there and
# pass on the branch's own build.
# Usage (on the build machine): WORK=<dir> SRC=<clone> REF=<tip> MODEL=<mamba gguf> GOOD_BIN=<branch CPU bin/> [DOCKER_ARGS=...] bash nan_mutation.sh
set -uo pipefail
: "${WORK:?}" "${SRC:?}" "${REF:?}" "${MODEL:?}" "${GOOD_BIN:?}"
mkdir -p "$WORK/ccache"; rm -rf "$WORK/src" "$WORK/done"; mkdir -p "$WORK/src"
git -C "$SRC" archive "$REF" | tar -x -C "$WORK/src"
sed -i 's/ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));/ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));/' "$WORK/src/src/llama-graph.cpp"
grep -c 'ggml_scale_inplace(ctx0, state_zero, 0)' "$WORK/src/src/llama-graph.cpp" > "$WORK/mutated.txt"
docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} -v "$WORK:/work" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    cmake -S /work/src -B /work/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/cmake.log 2>&1
    nice -n 19 cmake --build /work/build -j 8 --target test-llama-context > /work/build.log 2>&1; rc=$?
    chown -R "$HOST_UID:$HOST_GID" /work; exit $rc' || { echo "build failed" > "$WORK/verdict.txt"; date -Is > "$WORK/done"; exit 1; }
for b in mutant="$WORK/build/bin" branch="$GOOD_BIN"; do
    docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} -v "$WORK:$WORK" -v "$GOOD_BIN:$GOOD_BIN:ro" -v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 nice -n 19 "${b#*=}/test-llama-context" -m "$MODEL" -t 8 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$WORK/${b%%=*}.log"
    echo "${b%%=*}: $(grep -aoE '(^| )(nan_clear|one_slot|rollback|shared) +(OK|FAIL|SKIP)' "$WORK/${b%%=*}.log" | tr -s ' \n' ' ')" >> "$WORK/verdict.txt"
done
cat "$WORK/mutated.txt" "$WORK/verdict.txt"
date -Is > "$WORK/done"
