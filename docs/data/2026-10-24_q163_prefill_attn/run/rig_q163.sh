#!/bin/bash
# Q163 on the rig: the compute-6.1 prompt attention kernel (GGML_CUDA_FATTN_TILE_PASCAL, on by default on compute 6.1;
# 0 = the stock TILE kernel) in one binary (pascal-p106-v2-next d379da309 + the kernel), on against off.
#   0. kernel only, card 0: test-backend-ops perf on the Q163 prompt cases (512-token micro-batch with a causal mask at
#      depth 8k/32k/64k, and Flint's 1280-cell window ring; head 128 = Flint, head 256 = Qwen3.6; F16 and Q8_0): off,
#      on, and on with the plain score layout at head 128 (GGML_CUDA_FATTN_TILE_PASCAL_QK=1). If QK=1 is faster at head
#      128, run steps 1-4 with QK=1 (QK env below).
#   1. model output, K2h-r16, 9 cards: one 8192-token chunk at -ub 512, KLD of on against off
#   2. prompt speed, K2h-r16, 9 cards: llama-batched-bench at $DEPTHS tokens, -ub 512, off then on per depth
#   3. prompt speed, Qwen3.6 Q4_0-noK, $QCARDS: the same
#   4. Q8_0 cache, K2h-r16, 9 cards: compute buffers at -c 65536 (the F16 copy of a Q8_0 cache is gone with the kernel
#      on) and an 8k prompt, off and on
# Times (est.): step 0 ~5 min, 1 ~3 min, 2 ~35 min with 64k (off ~13 min + on ~8 min at 64k), 3 ~30 min, 4 ~5 min.
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=<tarball name without .tar.gz> [RIG=rig] [OUT=dir] rig_q163.sh
#   STEPS='0 1' etc.; DEPTHS='8192 32768' skips 64k; QCARDS=0,1,2,3,4 for Qwen3.6 on five cards; UB=256 if 64k runs
#   out of memory at -ub 512 (Q116 used 256 at 64k with the host mask).
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where rig_q163.md goes
D=$(cd "$(dirname "$0")" && pwd)
T=${T:?T=<tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
QWEN=${QWEN:-/mnt/models2/qwen36/Qwen3.6-35B-A3B-Q4_0-noK.gguf}
QK=${QK:-2}
E="LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_FATTN_TILE_PASCAL_QK=$QK"
ALL=0,1,2,3,4,5,6,7,8
QCARDS=${QCARDS:-$ALL}
STEPS=${STEPS:-0 1 2 3 4}
DEPTHS=${DEPTHS:-8192 32768 65536}
UB=${UB:-512}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
bb() { # <cards> <model> <kernel 0/1> <prompt tokens> [extra args]
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$1 $E GGML_CUDA_FATTN_TILE_PASCAL=$3 $B/llama-batched-bench -m $2 -ngl 99 -fa on -c $(($4 + 512)) -b 2048 -ub $UB -npp $4 -ntg 32 -npl 1 -lv 4 ${5:-} 2>&1" \
    | grep -E '^\|  *[0-9]|error|fail|out of memory|prompt attention kernel' | sed -E 's/^[0-9.]+ I //'
}
{
echo "#### Q163 $T $(date '+%F %T')"
if has 0; then
  # off, on (split scores at head 128, the default), on with the plain 4x4 score layout (QK=1; head 256 has only that)
  for arm in "0 2" "1 2" "1 1"; do
    read -r s qk <<< "$arm"
    echo "== 0. test-backend-ops perf, prompt attention, card 0, GGML_CUDA_FATTN_TILE_PASCAL=$s QK=$qk $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 GGML_CUDA_FATTN_TILE_PASCAL=$s GGML_CUDA_FATTN_TILE_PASCAL_QK=$qk $B/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'window=' 2>&1" \
      | grep -E "FLASH_ATTN_EXT" | sed -E 's/,mask=1//; s/,max_bias=[^,]*,logit_softcap=[^,]*,prec=f32//; s/,permute=\[0,2,1,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0//'
  done
fi
if has 1; then
  echo "== 1. K2h-r16 logits, 9 cards, one 8192-token chunk, -ub 512: the prompt kernel against TILE $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$ALL $E GGML_CUDA_FATTN_TILE_PASCAL=0 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 8192 --chunks 1 -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence-base /tmp/q163-off.kld 2>&1 | grep -E 'Final estimate'; \
           CUDA_VISIBLE_DEVICES=$ALL $E GGML_CUDA_FATTN_TILE_PASCAL=1 $B/llama-perplexity -m $K2H -f /opt/llama-builds/data/wiki.test.raw -c 8192 --chunks 1 -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence-base /tmp/q163-off.kld --kl-divergence 2>&1 | grep -E 'Mean PPL|Mean    KLD|Maximum KLD|Same top p'; rm -f /tmp/q163-off.kld"
fi
if has 2; then
  for npp in $DEPTHS; do
    for s in 0 1; do
      echo "== 2. K2h-r16, 9 cards, $npp-token prompt, GGML_CUDA_FATTN_TILE_PASCAL=$s $(date +%T)"
      bb $ALL $K2H $s $npp
    done
  done
fi
if has 3; then
  for npp in $DEPTHS; do
    for s in 0 1; do
      echo "== 3. Qwen3.6 Q4_0-noK, cards $QCARDS, $npp-token prompt, GGML_CUDA_FATTN_TILE_PASCAL=$s $(date +%T)"
      bb $QCARDS $QWEN $s $npp
    done
  done
fi
if has 4; then
  for s in 0 1; do
    echo "== 4. K2h-r16, Q8_0 cache, 9 cards, -c 65536: compute buffers, GGML_CUDA_FATTN_TILE_PASCAL=$s $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$ALL $E GGML_CUDA_FATTN_TILE_PASCAL=$s $B/llama-batched-bench -m $K2H -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -b 2048 -ub 512 -npp 512 -ntg 1 -npl 1 -lv 4 2>&1" \
      | grep -E 'compute buffer size' | grep CUDA | sed -E 's/.*(CUDA[0-9]+ compute buffer size = *[0-9.]+ MiB).*/\1/' | tr '\n' ';'
    echo
    echo "== 4. K2h-r16, Q8_0 cache, 9 cards, 8192-token prompt, GGML_CUDA_FATTN_TILE_PASCAL=$s $(date +%T)"
    bb $ALL $K2H $s 8192 "-ctk q8_0 -ctv q8_0"
  done
fi
echo "Q163_DONE $(date +%T)"
} 2>&1 | tee -a "$OUT/rig_q163.md"
