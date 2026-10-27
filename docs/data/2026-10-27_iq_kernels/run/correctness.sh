#!/usr/bin/env bash
# Correctness of the IQ2_S / IQ3_XXS Pascal matrix-vector kernels on one GPU of the build machine (GPUS=<n>; a V100 was
# used): test-backend-ops against the CPU with the Pascal kernels forced on any GPU (GGML_CUDA_MMVQ_PASCAL=2),
#   1. natively (sm_70): the new types' MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION and MUL_MAT_ID_SHARED cases, and Q6_K's
#      (whose staged read the new types share);
#   2. with CUDA_FORCE_PTX_JIT=1 (the compute_61 code the P106 runs), the same cases at 1 to 4 tokens: larger batches run
#      llama.cpp's own batched kernels, which this branch does not touch and which abort or differ under the JIT;
#   3. compute-sanitizer memcheck under the JIT on one case per type and kind.
# Prints pass counts and failures only; no timings.
#   ssh <build machine> "WORK=<folder> NAME=<build name> GPUS=<n> bash -s" < run/correctness.sh
set -euo pipefail
WORK=${WORK:?folder holding out/<NAME>}
NAME=${NAME:?build name}
GPUS=${GPUS:?GPUS=<device>}
LABEL=${LABEL:-pascal-quants=1}
docker run --rm --label "$LABEL" --name v3iq-check --gpus "\"device=$GPUS\"" -v "$WORK/out:/out:ro" -e NAME="$NAME" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
T="/out/$NAME/bin/test-backend-ops -b CUDA0"
export GGML_CUDA_MMVQ_PASCAL=2
s() { grep -E "FAIL|tests passed|Abort|CUDA error|terminate called|Segmentation" | sed -E "s/^ +//" | sort | uniq -c || true; }
f() {  # filter for 1 to 4 tokens: the fusion test names tokens m, the others n
  case $1 in MUL_MAT_VEC_FUSION) echo "m=[1-4],";; MUL_MAT_ID_SHARED) echo "n=[1-4]";; *) echo "n=[1-4],";; esac
}
for jit in 0 1; do
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_ID_SHARED; do
    for t in iq2_s iq3_xxs; do
      p="=$t,"; [ $jit = 1 ] && p="=$t,.*$(f $op)"
      echo "== JIT=$jit $op $t"
      CUDA_FORCE_PTX_JIT=$jit $T -o $op -p "$p" 2>&1 | s
    done
  done
  echo "== JIT=$jit Q6_K (MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION)"
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
    p="=q6_K,"; [ $jit = 1 ] && p="=q6_K,.*$(f $op)"
    CUDA_FORCE_PTX_JIT=$jit $T -o $op -p "$p" 2>&1 | s
  done
done
CS=/usr/local/cuda/bin/compute-sanitizer
for t in iq2_s iq3_xxs; do
  for c in "MUL_MAT_ID|=$t,.*m=1024,n=1,k=3072" "MUL_MAT_ID|=$t,.*m=3072,n=4,k=1024" \
           "MUL_MAT_VEC_FUSION|=$t,.*m=2,n=1024,k=3072" "MUL_MAT|=$t,.*m=2048,n=1,k=512"; do
    echo "== memcheck JIT=1 ${c%%|*} ${c#*|}"
    CUDA_FORCE_PTX_JIT=1 $CS --tool memcheck --print-limit 5 $T -o ${c%%|*} -p "${c#*|}" 2>&1 \
      | grep -E "ERROR SUMMARY|Invalid|out of bounds|tests passed|FAIL" | sort | uniq -c
  done
done
echo "== done"
'
