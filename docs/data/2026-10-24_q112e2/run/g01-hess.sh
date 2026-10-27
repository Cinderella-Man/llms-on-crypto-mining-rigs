#!/usr/bin/env bash
# Q112 E2/E3 (2026-10-24, GPU): input Hessians of all 196 always-read inputs from K2-r16's own forward pass (rescale
# off), wikitext-2 train chunks 0-499 of 512 tokens as in E1: attn_norm (q/k/v), kqv_out (attention output), ffn_norm
# (shared expert and dense gate/up), ffn_swiglu (their down). ~11 GB of float64; Gram matrices by cblas_ssyrk.
# Env as for common.sh (BIN: a CUDA build with -DHESS_BLAS_LIB).
set -e; . "$(dirname "$0")/common.sh"; mkdir -p $HD/hess
HT=$(for l in $(seq 0 48); do printf 'attn_norm-%d,kqv_out-%d,ffn_norm-%d,ffn_swiglu-%d,' $l $l $l $l; done); HT=${HT%,}
gpu -e OMP_NUM_THREADS=12 -e LLAMA_NO_MOE_DOWN_RESCALE=1 -e HESS_TENSORS=$HT \
  -e HESS_OUT=$HD/hess -e HEAD_TEXT=$WORK/wiki.train.raw -e HEAD_CHUNK0=0 -e HEAD_CHUNKS=500 \
  $IMG $B/llama-act-hessian -m $K -c 512 -b 512 -ngl 99 -fa on > $D/hess.log 2>&1
grep -E "tokens, chunks|never seen|done" $D/hess.log; grep -c "rows$" $D/hess.log
