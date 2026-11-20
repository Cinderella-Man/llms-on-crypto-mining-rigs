#!/usr/bin/env bash
# C881 (Q1590): CPU builds of the trees (every target, the tests included) and the CPU test suite, one Docker container
# per tree, one at a time under a build lock.
# Usage (on the build machine):
#   WORK=<work dir> LOCK=<lock file> [JOBS=8] [MEM=16g] [IMAGE=pq-c565-cpu:24.04] bash build.sh name=ref[:ctest] ...
#   $WORK/src     a llama.cpp clone that holds the refs
#   name=ref      git archive of <ref> into $WORK/t/<name>/src, every target built in $WORK/t/<name>/build
#   :ctest        then ctest in the same container
# Output per tree in $WORK/t/<name>/: COMMIT, cmake.log, build.log, RC ("rc=<n> s=<seconds>"), ccache.txt, and with
# :ctest ctest.log and CTRC. Done marker: $WORK/logs/BUILD_DONE.
# Containers: label pascal-quants=1 and c881=1, --cpu-shares=256, --memory=$MEM, --network none, only $WORK mounted;
# ccache in $WORK/ccache. The image is Ubuntu 24.04 with gcc 13.3 and cmake 3.28 (any such image does).
set -uo pipefail
WORK=${WORK:?WORK=<work dir>}
LOCK=${LOCK:?LOCK=<lock file>}
JOBS=${JOBS:-8}
MEM=${MEM:-16g}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
mkdir -p "$WORK/logs"
rm -f "$WORK/logs/BUILD_DONE"
for spec in "$@"; do
    name=${spec%%=*}
    rest=${spec#*=}
    ref=${rest%%:*}
    ct=0
    [[ $rest == *:ctest ]] && ct=1
    T=$WORK/t/$name
    rm -rf "$T"
    mkdir -p "$T/src"
    git -C "$WORK/src" archive "$ref" | tar -x -m -C "$T/src"  # -m: the files get the time of extraction
    git -C "$WORK/src" rev-parse "$ref" > "$T/COMMIT"
    echo "=== $name $ref $(cat "$T/COMMIT") ctest=$ct $(date -Is)"
    flock -w 7200 "$LOCK" docker run --rm --name "c881-build-$name" --label pascal-quants=1 --label c881=1 \
        --cpu-shares=256 --memory="$MEM" --memory-swap="$MEM" --network none -v "$WORK:$WORK" \
        -e CCACHE_DIR="$WORK/ccache" -e CCACHE_BASEDIR="$T" -e HF_UI_VERSION=b12483 \
        -e T="$T" -e JOBS="$JOBS" -e CT="$ct" "$IMAGE" bash -c '
        cmake -S "$T/src" -B "$T/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > "$T/cmake.log" 2>&1 \
            || { echo "rc=cmake" > "$T/RC"; exit 0; }
        s=$(date +%s)
        nice -n 19 cmake --build "$T/build" -j "$JOBS" -- -k > "$T/build.log" 2>&1
        echo "rc=$? s=$(( $(date +%s) - s ))" > "$T/RC"
        ccache -s > "$T/ccache.txt" 2>&1 || true
        if [ "$CT" = 1 ]; then
            s=$(date +%s)
            cd "$T/build" && nice -n 19 ctest -j "$JOBS" --timeout 3600 --output-on-failure > "$T/ctest.log" 2>&1
            echo "rc=$? s=$(( $(date +%s) - s ))" > "$T/CTRC"
        fi'
    echo "    $name: $(cat "$T/RC" 2>/dev/null) $(cat "$T/CTRC" 2>/dev/null)"
done
echo "=== all done $(date -Is)"
date -Is > "$WORK/logs/BUILD_DONE"
