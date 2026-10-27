#!/usr/bin/env bash
# Correctness of N2P/N3P (fork branch pascal-p106-v3-lowbit) on one GPU of the quality machine (a V100), test-backend-ops
# against the CPU, from the CUDA build of run/build.sh:
#   - the Pascal kernels forced on (GGML_CUDA_MMVQ_PASCAL=2), natively (sm_70) and with CUDA_FORCE_PTX_JIT=1 (the
#     compute_61 code the P106 runs): MUL_MAT (1-8 tokens; 9-128 tokens take MMQ, natively only), MUL_MAT_ID, the fused
#     gate/up, the shared expert in the routed launch, GET_ROWS; the P106 checks its own MMQ (rig_kernels.sh step 2);
#   - Q6_K's cases again (its 2-byte-aligned staged read is now shared with N2P);
#   - the N2P/N3P cases on llama.cpp's matrix-vector kernel (GGML_CUDA_MMVQ_PASCAL=0);
#   - compute-sanitizer memcheck on one case per type and kind (MMQ natively). No timings, ~30 minutes.
#   QM=<ssh alias> WORK=<work dir on QM, as for build.sh> NAME=<build name> GPUS=<device> bash correctness.sh
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?work dir on QM}
NAME=${NAME:?build name}
GPUS=${GPUS:?GPU index on QM}
ssh "$QM" "docker run -i --rm --cpu-shares=256 --label pascal-quants=1 --gpus device=$GPUS -e NAME=$NAME -v $WORK/out:/b:ro \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -s" <<'EOF'
T="/b/$NAME/bin/test-backend-ops -b CUDA0"
s() { grep -E "FAIL|tests passed|Backend .*: (OK|FAIL)" | sed -E "s/^ +//" | sort | uniq -c || true; }
for jit in 0 1; do
  # through the compute 6.1 code only 1-8 tokens: the host picks the MMQ tile configuration for the card (Volta) while
  # the forced compute 6.1 code holds Pascal's, so MMQ (9 tokens and more) cannot run correctly that way, for any type
  b=""; [ "$jit" = 1 ] && b=".*,n=[1-8],"
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_ID_SHARED GET_ROWS; do
    for t in n2p n3p; do
      f="=$t,"; case $op in MUL_MAT|MUL_MAT_ID) f="=$t,$b" ;; esac
      echo "== Pascal kernels, JIT=$jit $op $t"
      GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $T -o $op -p "$f" 2>&1 | s
    done
  done
  echo "== Pascal kernels, JIT=$jit Q6_K regression"
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
    f="=q6_K,"; case $op in MUL_MAT|MUL_MAT_ID) f="=q6_K,$b" ;; esac
    GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $T -o $op -p "$f" 2>&1 | s
  done
done
for t in n2p n3p; do
  echo "== llama.cpp kernel (GGML_CUDA_MMVQ_PASCAL=0) MUL_MAT, MUL_MAT_ID $t"
  for op in MUL_MAT MUL_MAT_ID; do
    GGML_CUDA_MMVQ_PASCAL=0 $T -o $op -p "=$t," 2>&1 | s
  done
done
CS=/usr/local/cuda/bin/compute-sanitizer
for t in n2p n3p; do
  for p in "1|MUL_MAT_ID|=$t,.*m=1024,n=1,k=3072" "1|MUL_MAT_ID|=$t,.*m=3072,n=4,k=1024" \
           "1|MUL_MAT_VEC_FUSION|=$t,.*n=1024,k=3072,use_id=1" "1|MUL_MAT|=$t,.*m=2048,n=1,k=512" \
           "0|MUL_MAT|=$t,.*m=1024,n=33,k=3072"; do
    jit=${p%%|*}; p=${p#*|}; op=${p%%|*}; pat=${p#*|}
    echo "== memcheck JIT=$jit $op $pat"
    GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $CS --tool memcheck --print-limit 5 $T -o $op -p "$pat" 2>&1 \
      | grep -E "ERROR SUMMARY|Invalid|out of bounds|tests passed|FAIL" | sort | uniq -c
  done
done
echo "== done"
EOF
