#!/usr/bin/env bash
# Q88 test 1 (correctness only, no timings), 2026-10-24: test-backend-ops MOE_LAYER on one GPU (GPUS=<n>), the
# persistent MoE kernel against the CPU and bit for bit against the per-op path, natively and through the compute_61
# PTX; grid sizes; deliberate errors; the existing products and fusions the header move touches. Ran on an RTX 3090.
# Usage: BUILD=<out dir of scripts/build_llamacpp_tests_docker.sh, fork branch pascal-p106-v2-persistent> GPUS=<n> t1.sh
set -euo pipefail
GPUS=${GPUS:?GPUS=<device>}
BUILD=$(realpath "${BUILD:?BUILD=<test build out dir>}")
docker run --rm --name q88-t1 --gpus "\"device=$GPUS\"" -v "$BUILD:/b:ro" \
  -e LD_LIBRARY_PATH=/b/cuda12-libs nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
B=/b/build/bin
T="$B/test-backend-ops -b CUDA0"
export GGML_CUDA_MMVQ_PASCAL=2
s() { grep -E "FAIL|tests passed|Backend .*: (OK|FAIL)" | sed -E "s/^ +//" | sort | uniq -c || true; }
chk() { grep -E "persistent MoE kernel ran" | sed -E "s/.*kernel ran/ran/" | sort | uniq -c || true; }
echo "== 1. MOE_LAYER, per-op path (switch off), against the CPU"
$T -o MOE_LAYER 2>&1 | s
for jit in 0 1; do
  echo "== 2.$jit persistent kernel (GGML_CUDA_PERSISTENT_MOE=2), CUDA_FORCE_PTX_JIT=$jit"
  echo "-- against the CPU"
  CUDA_FORCE_PTX_JIT=$jit GGML_CUDA_PERSISTENT_MOE=2 $T -o MOE_LAYER 2>&1 | s
  echo "-- self-check: bit for bit against the per-op path"
  CUDA_FORCE_PTX_JIT=$jit GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1 $T -o MOE_LAYER 2>&1 | tee /tmp/c.log | chk; s < /tmp/c.log
  for g in 1 2 3 5248; do
    printf -- "-- grid %-5s self-check: " $g
    CUDA_FORCE_PTX_JIT=$jit GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1 GGML_CUDA_PERSISTENT_MOE_GRID=$g $T -o MOE_LAYER 2>&1 | chk | tr "\n" " "; echo
  done
done
echo "== 3. deliberate errors (must be caught)"
echo "-- 2: routing weights x 1.0001: self-check"
GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1 GGML_CUDA_PERSISTENT_MOE_FAULT=2 $T -o MOE_LAYER 2>&1 | chk
echo "-- 3: shared expert left out: against the CPU"
GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_FAULT=3 $T -o MOE_LAYER 2>&1 | s
echo "-- 1: top-k never published (every wait times out): the next graph compute must abort"
GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_FAULT=1 timeout 600 $T -o MOE_LAYER > /tmp/f1.log 2>&1; echo "exit $?"
grep -E "timed out|FAIL|passed" /tmp/f1.log | head -3
echo "== 4. existing tests the change touches (Pascal kernels forced), with the switch on"
for o in ADD_RMS_NORM_MUL_Q8 MUL_MAT_ID_FUSION MUL_MAT_ID_SHARED MOE_REDUCE TOPK_MOE; do
  printf "%-20s " $o; GGML_CUDA_PERSISTENT_MOE=2 $T -o $o 2>&1 | s | tr "\n" " "; echo
done
echo "-- one-token products of the six Pascal types, native and PTX"
for jit in 0 1; do
  printf "jit=%s " $jit; CUDA_FORCE_PTX_JIT=$jit $T -o MUL_MAT,MUL_MAT_ID -p "type_a=(q4_0|q4_1|q5_0|q8_0|q5_K|iq4_xs),.*n=1," 2>&1 | s | tr "\n" " "; echo
done
'
