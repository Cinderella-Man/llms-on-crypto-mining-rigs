#!/usr/bin/env bash
# Q163 op checks (2026-10-24/25) on one GPU that runs the build's compute_61 PTX (a V100 here; JIT-compiled once into
# $WORK/cuda-cache), each step in its own container:
#   1. FLASH_ATTN_EXT, all cases, the compute-6.1 prompt kernel forced (GGML_CUDA_FATTN_TILE_PASCAL=2: not a 6.1 card)
#   2. the kernel's own cases (head 128/256, 8 or 16 query heads per KV head) with the kernel off (=0): the stock kernels
#   3. faults 1-3 (output x 1.05, sinks dropped, last key tile dropped) on the kernel's cases: they must fail
# Usage: BUILD=<out dir of scripts/build_llamacpp_tests_docker.sh, fork branch pascal-p106-v2-prefill-attn> WORK=<dir>
#        [GPU=n] [LOCK=file] [FAULTS='1 2 3'] t1.sh       (logs in $WORK/logs/t1-*.log)
W=${WORK:?WORK=<work dir>}
BUILD=$(realpath "${BUILD:?BUILD=<test build out dir>}")
mkdir -p $W/logs $W/cuda-cache
dock() { # <space-separated VAR=value list> <command...>
    local envs=() e
    for e in $1; do envs+=(-e "$e"); done
    shift
    ${LOCK:+flock $LOCK} docker run --rm --name q163-t1 --gpus "\"device=${GPU:-0}\"" \
        -v "$BUILD:/b:ro" -v $W/cuda-cache:/cc -e LD_LIBRARY_PATH=/b/cuda12-libs -e CUDA_CACHE_PATH=/cc \
        -e CUDA_CACHE_MAXSIZE=4294967296 "${envs[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@"
}
T="/b/build/bin/test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT"
OWN='hsk=(128|256),hsv=(128|256),nh=[0-9]+,nr23=\[(8|16),'
summ() { echo "$1: $(grep -oE '[0-9]+/[0-9]+ tests passed' $2 | tail -1), FAIL lines $(grep -c FAIL $2)"; }
echo "=== 1. all FLASH_ATTN_EXT cases, kernel forced  $(date +%T)"
dock "GGML_CUDA_FATTN_TILE_PASCAL=2" $T > $W/logs/t1-all-on.log 2>&1
summ "on, all" $W/logs/t1-all-on.log
echo "=== 2. own cases, kernel off  $(date +%T)"
dock "GGML_CUDA_FATTN_TILE_PASCAL=0" $T -p "$OWN" > $W/logs/t1-own-off.log 2>&1
summ "off, own" $W/logs/t1-own-off.log
echo "=== 2b. own cases, kernel forced, plain score layout at head 128 (QK=1)  $(date +%T)"
dock "GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_QK=1" $T -p "$OWN" > $W/logs/t1-own-qk1.log 2>&1
summ "on QK=1, own" $W/logs/t1-own-qk1.log
echo "=== 3. faults  $(date +%T)"
for f in ${FAULTS:-1 2 3}; do
    dock "GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_FAULT=$f" $T -p "$OWN" > $W/logs/t1-fault$f.log 2>&1
    summ "fault $f" $W/logs/t1-fault$f.log
done
echo "t1 ended $(date +%T)"
