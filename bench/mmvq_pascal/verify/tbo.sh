#!/usr/bin/env bash
# test-backend-ops of a llama.cpp CUDA build on one GPU, in the CUDA container, with the fork's Pascal matrix-vector
# kernels set by GGML_CUDA_MMVQ_PASCAL (0 off, 1 compute 6.1 only, 2 forced on any GPU). A build with sm_61 code only
# runs on a newer GPU through its compute_61 PTX, JIT-compiled once into $WORK/nvcache. Correctness only, no timings.
# Used for the Q4_K kernel's checks (Q149, 2026-10-24); run_suite.sh calls it once per op group.
#
# Usage: [GPU=n] [LOCK=file] [STAGE_ACT=0|1] [SKIP=q4_K,q6_K] WORK=<dir> tbo.sh <tag> <GGML_CUDA_MMVQ_PASCAL> <build dir> [test-backend-ops args...]
#   <build dir>  a llama.cpp build with bin/test-backend-ops (absolute, or relative to $WORK), e.g. an unpacked tarball
#                of scripts/build_llamacpp_docker.sh with TESTS=1
#   STAGE_ACT    GGML_CUDA_MMVQ_PASCAL_STAGE_ACT (staged activations); SKIP: GGML_CUDA_MMVQ_PASCAL_SKIP (types left to
#                llama.cpp's kernels)
#   LOCK         hold this flock while the container runs (a GPU shared with other jobs); the container gets a low CPU share
# The log goes to $WORK/logs/<tag>.log, its last line "exit <status>".
set -euo pipefail
tag=$1; mode=$2; bin=$3; shift 3
WORK=${WORK:?WORK=<work dir>}
case $bin in /*) ;; *) bin=$WORK/$bin ;; esac
bin=$(realpath "$bin")
mkdir -p "$WORK/logs" "$WORK/nvcache"
${LOCK:+flock $LOCK} docker run --rm --name tbo-$tag --cpu-shares=256 --gpus "\"device=${GPU:-0}\"" \
    -v "$WORK:$WORK" -v "$bin:$bin:ro" \
    -e GGML_CUDA_MMVQ_PASCAL=$mode -e GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=${STAGE_ACT:-0} -e GGML_CUDA_MMVQ_PASCAL_SKIP=${SKIP:-} \
    -e CUDA_CACHE_PATH=$WORK/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 900 $bin/bin/test-backend-ops -b CUDA0 "$@" > $WORK/logs/$tag.log 2>&1
echo "exit $?" >> $WORK/logs/$tag.log
