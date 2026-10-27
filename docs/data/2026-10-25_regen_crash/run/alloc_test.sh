#!/usr/bin/env bash
# The regenerate crash's allocator test (2026-10-25): tests/test-alloc on the CPU for two llama.cpp trees, the fixed one
# (fork commit 2d53179aa, leafcheck.diff) and its parent with only the new test case (test_unread_leaf_needs_memory),
# which must crash there (exit 139 = segfault). One container, low CPU share, -j 8, ccache in $WORK/ccache.
# Usage: WORK=<dir> alloc_test.sh <fixed source tree> <unfixed source tree with the new test>
set -uo pipefail
W=${WORK:?WORK=<work dir>}
FIXED=$(realpath "$1"); UNFIXED=$(realpath "$2")
mkdir -p $W/ccache
docker rm -f alloc-test > /dev/null 2>&1 || true
docker run --rm --name alloc-test --cpu-shares=256 -v "$W:/w" -v "$FIXED:/src-fixed:ro" -v "$UNFIXED:/src-unfixed:ro" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -uo pipefail -c '
  export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/w/ccache
  apt-get update -qq && apt-get install -y -qq cmake ccache > /dev/null
  cd /w
  for p in "build-fixed /src-fixed" "build-unfixed /src-unfixed"; do
    set -- $p
    [ -f $1/CMakeCache.txt ] || cmake -S $2 -B $1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > cmake-$1.log
    cmake --build $1 -j 8 --target test-alloc > build-$1.log 2>&1 || { grep error build-$1.log | head; continue; }
    echo "== $1 ($2):"
    $1/bin/test-alloc > test-$1.txt 2>&1; rc=$?
    tail -6 test-$1.txt; echo "exit $rc (139 = segfault)"
  done
'
