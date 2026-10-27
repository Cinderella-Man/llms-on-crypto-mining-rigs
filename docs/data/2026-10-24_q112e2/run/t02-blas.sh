#!/usr/bin/env bash
# Q112 E2 check (2026-10-24, CPU only): act_hessian with cblas_ssyrk (a CPU build with -DHESS_BLAS_LIB) against its
# plain loops (a CPU build without it) on K2-r16, 4 chunks of wikitext-2 train: dense inputs of several widths (2,048 to
# 6,144) and the experts of layers 9 and 47 (flushes of 16 rows). check_hessians.py recomputes the BLAS run's Hessians
# from its dumped rows and compares every output file of the two builds; the expert files (17 GB) are deleted afterwards.
# Env as for common.sh, plus BIN_CPU_BLAS and BIN_CPU (bin/ of the two CPU builds). Result: ../checks/t02-check.md.
set -e
. "$(dirname "$0")/common.sh"
O=$HD/t02; mkdir -p $O/blas $O/loops
for v in blas loops; do
  b=$BIN_CPU_BLAS; [ $v = loops ] && b=$BIN_CPU
  dump=""; [ $v = blas ] && dump="-e HESS_DUMP=ffn_norm-9,ffn_moe_topk-9,ffn_moe_swiglu-9,ffn_norm-47,ffn_moe_topk-47,ffn_moe_swiglu-47"
  docker run --rm $MOUNTS -v "$(dirname "$b"):$(dirname "$b"):ro" -e OMP_NUM_THREADS=12 -e LLAMA_NO_MOE_DOWN_RESCALE=1 \
    -e HESS_TENSORS=ffn_norm-9,ffn_norm-47,kqv_out-33,attn_norm-0,ffn_swiglu-0,ffn_swiglu-20 -e HESS_EXPERTS=9,47 \
    -e HESS_EXP_FLUSH=16 $dump -e HESS_OUT=$O/$v -e HEAD_TEXT=$WORK/wiki.train.raw -e HEAD_CHUNK0=0 -e HEAD_CHUNKS=4 \
    $IMG $b/llama-act-hessian -m $K -c 512 -b 512 -t 12 > $O/$v.log 2>&1
  grep -E "rows|never seen|done" $O/$v.log | sed "s/^/$v: /"
done
py "python $T/act_hessian/check_hessians.py $O/blas 9 && python $T/act_hessian/check_hessians.py $O/blas 47 && \
    python $T/act_hessian/check_hessians.py --compare $O/blas $O/loops" | tee $O/check.md
rm $O/blas/experts-*.e64 $O/loops/experts-*.e64
