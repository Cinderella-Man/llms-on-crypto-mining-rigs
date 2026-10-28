#!/usr/bin/env bash
# Correctness of the 2-4-token Pascal kernels that decode each weight once (fork branch pascal-p106-v3-mtpk,
# mul_mat_vec_q_pascal_mt) on one GPU of the quality machine (a V100), from a CUDA build with test-backend-ops for
# compute 6.1 (+ compute_61 PTX) and sm_70:
#   - test-backend-ops against the CPU with the Pascal kernels forced on (GGML_CUDA_MMVQ_PASCAL=2), natively (sm_70)
#     and with CUDA_FORCE_PTX_JIT=1 (the compute_61 code a P106 runs), for Ornith's types (IQ2_S, IQ3_XXS, N2P, N3P,
#     Q8_0, Q5_0): MUL_MAT and MUL_MAT_ID at 1-4 tokens (MMQ, 9 tokens and more, natively only: under the forced
#     compute 6.1 code the host picks Volta's MMQ tiles), the fused gate/up, the shared expert in the routed launch;
#     the same with tiny activation tiles and several row blocks per block (GGML_CUDA_MMVQ_PASCAL_MT_ACT_KB=1,
#     _MT_RB=3), so the tile and row-block loops run on every shape, and natively with GGML_CUDA_MMVQ_PASCAL_MT=0 (the
#     earlier multi-token kernel);
#   - the Flint types' multi-token cases, which the new kernel must leave alone except Q8_0 and Q5_0;
#   - multitoken_bitid (bench/mmvq_pascal/verify) in both modes, native and PTX: every 2-4-token product bit-identical
#     to the single-token products of its columns;
#   - compute-sanitizer memcheck on one case per kind. No timings.
#   BIN=<build bin dir with test-backend-ops> BITID=<multitoken_bitid binary> GPUS=<device> OUT=<log file> \
#     [JITS="0 1"] [MTS="1|1 1 3|0"] [FLINT=1] [MEMCHECK=1] bash correctness.sh
# JITS: the code paths (0 = native, 1 = compute_61 PTX; a GPU without code of its own runs the PTX either way); MTS: the
# kernel settings, "MT [ACT_KB [RB]]" separated by |, the first one for every operation, the others for MUL_MAT and
# MUL_MAT_ID and the bit-identity check; FLINT=0 skips the Flint types, MEMCHECK=0 the sanitizer.
set -euo pipefail
: "${BIN:?}" "${BITID:?}" "${GPUS:?}" "${OUT:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
docker run -i --rm --cpu-shares=256 --label "$LABEL" --gpus "device=$GPUS" -v "$BIN:/b:ro" \
    -v "$(dirname "$BITID"):/t:ro" -e BITID="/t/$(basename "$BITID")" -e JITS="${JITS:-0 1}" \
    -e MTS="${MTS:-1|1 1 3|0}" -e FLINT="${FLINT:-1}" -e MEMCHECK="${MEMCHECK:-1}" "$IMAGE" bash -s > "$OUT" 2>&1 <<'EOF'
T="/b/test-backend-ops -b CUDA0"
s() { grep -E "FAIL|tests passed|Backend .*: (OK|FAIL)" | sed -E "s/^ +//" | sort | uniq -c || true; }
types="iq2_s iq3_xxs n2p n3p q8_0 q5_0"
IFS='|' read -r -a mtl <<< "$MTS"
for jit in $JITS; do
  b=""; [ "$jit" = 1 ] && b=".*,n=[1-8],"
  ops="MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_ID_SHARED"
  for mt in "${mtl[@]}"; do
    set -- $mt
    mtenv="GGML_CUDA_MMVQ_PASCAL_MT=$1 ${2:+GGML_CUDA_MMVQ_PASCAL_MT_ACT_KB=$2} ${3:+GGML_CUDA_MMVQ_PASCAL_MT_RB=$3}"
    for op in $ops; do
      for t in $types; do
        f="=$t,"; case $op in MUL_MAT|MUL_MAT_ID) f="=$t,$b" ;; esac
        echo "== JIT=$jit MT=$mt $op $t"
        env GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $mtenv $T -o $op -p "$f" 2>&1 | s
      done
    done
    echo "== JIT=$jit MT=$mt bit identity (ornith)"
    env GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $mtenv $BITID ornith 2>&1 | grep -E "DIFFERENT|comparisons" || true
    ops="MUL_MAT MUL_MAT_ID"
  done
  if [ "$FLINT" = 1 ]; then
    echo "== JIT=$jit Flint types, multi-token"
    for t in q4_0 q4_1 q4_K q5_K q6_K iq4_xs; do
      for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
        GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $T -o $op -p "=$t,.*n=[2-4]," 2>&1 | s
      done
    done
    echo "== JIT=$jit bit identity (flint)"
    GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $BITID flint 2>&1 | grep -E "DIFFERENT|comparisons" || true
  fi
done
CS=/usr/local/cuda/bin/compute-sanitizer
[ "$MEMCHECK" = 1 ] && for p in "MUL_MAT_ID|=n2p,.*n_mats=10,n_used=8,b=0,m=3072,n=4,k=1024" "MUL_MAT_ID|=iq2_s,.*n_mats=8,n_used=8,b=0,m=3072,n=3,k=1024" \
         "MUL_MAT_VEC_FUSION|=n3p,.*n=1024,k=3072,use_id=1,n_mats=10" "MUL_MAT|=q5_0,.*m=3072,n=4,k=8192" \
         "MUL_MAT_ID_SHARED|=iq3_xxs,"; do
  op=${p%%|*}; pat=${p#*|}
  for jit in $JITS; do
    echo "== memcheck JIT=$jit $op $pat"
    GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=$jit $CS --tool memcheck --print-limit 5 $T -o $op -p "$pat" 2>&1 \
      | grep -E "ERROR SUMMARY|Invalid|out of bounds|tests passed|FAIL" | sort | uniq -c
  done
done
echo "== done"
EOF
grep -E "FAIL|DIFFERENT|comparisons|ERROR SUMMARY|tests passed" "$OUT" | sort | uniq -c | sort -rn | head -60
