#!/usr/bin/env bash
# Each v2 switch on its own (2026-10-25): one binary (release candidate rc4 2d53179aa), everything on against one switch
# off, llama-bench on nine cards; "all on" runs at the start, the middle and the end, so clock or temperature drift shows.
# Env: RIG (ssh alias), BIN (the build's bin folder on the rig), K2H, Q8, UD (GGUF paths on the rig), OUT (folder for
# ablation.md), STEPS (default 'decode q8 kquant long'). Run with nothing else on the rig.
#   decode   Flint-Pascal-Q5 (recipe K2h-r16): pp512 + tg128, all on, then one switch off at a time:
#            GGML_CUDA_MMVQ_PASCAL=0 (Pascal matrix-vector kernels), GGML_CUDA_PERSISTENT_MOE=0 (persistent MoE kernel),
#            GGML_CUDA_FATTN_PASCAL=0 (Pascal decode attention), GGML_CUDA_ATTN_BLOCK=0 (attention block in two
#            launches), GGML_CUDA_NORM_Q8=0 (norm + Q8 fusion), LLAMA_INPUT_UPLOAD=0 (packed input uploads),
#            LLAMA_INPUT_ON_GPU=0 (input embeddings on the card), LLAMA_GPU_KQ_MASK=0 (attention mask built on the cards)
#   q8       Flint Q8_0, -ts 6/6/5/5/6/5/5/6/6: the same switches (stopped after its first run on 2026-10-25)
#   kquant   UD-Q4_K_XL: all on against GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K (Pascal Q4_K/Q6_K kernels)
#   long     Flint-Pascal-Q5, decode after a 32k-token context (-d 32768 -n 64): all on, GGML_CUDA_FATTN_PASCAL=0,
#            LLAMA_GPU_KQ_MASK=0 (the two that matter most at depth)
set -uo pipefail
RIG=${RIG:-rig}
STEPS=${STEPS:-decode q8 kquant long}
T=${T:-llama.cpp-20261025-2d53179aa-pascal-p106-v2-rc4}
B=${BIN:-/opt/llama-builds/$T/bin}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
TSQ8=6/6/5/5/6/5/5/6/6
C9=0,1,2,3,4,5,6,7,8
O=${OUT:-.}
mkdir -p "$O"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
bench() { # <label> <model> <env or -> <extra llama-bench args>
  local label=$1 m=$2 e=$3; shift 3
  [ "$e" = - ] && e=""
  echo "== $label $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 $e $B/llama-bench -m $m -ngl 99 -fa on $* -o md 2>&1" \
    | grep -E '^\| .*(pp|tg)[0-9]+|error|failed|out of memory' | sed "s/^/$label | /"
}
{
echo "# One improvement at a time, $T, $(date -Is)"
if has decode; then
  bench "all on (1)" $K2H - -p 512 -n 128 -r 3
  for e in GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0; do
    bench "$e" $K2H "$e" -p 512 -n 128 -r 3
  done
  bench "all on (2)" $K2H - -p 512 -n 128 -r 3
  for e in GGML_CUDA_NORM_Q8=0 LLAMA_INPUT_UPLOAD=0 LLAMA_INPUT_ON_GPU=0 LLAMA_GPU_KQ_MASK=0; do
    bench "$e" $K2H "$e" -p 512 -n 128 -r 3
  done
  bench "all on (3)" $K2H - -p 512 -n 128 -r 3
fi
if has q8; then
  bench "Q8_0 all on (1)" $Q8 - -p 512 -n 128 -r 3 -ts $TSQ8
  for e in GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0 GGML_CUDA_NORM_Q8=0; do
    bench "Q8_0 $e" $Q8 "$e" -p 512 -n 128 -r 3 -ts $TSQ8
  done
  bench "Q8_0 all on (2)" $Q8 - -p 512 -n 128 -r 3 -ts $TSQ8
  for e in LLAMA_INPUT_UPLOAD=0 LLAMA_INPUT_ON_GPU=0 LLAMA_GPU_KQ_MASK=0 GGML_CUDA_PERSISTENT_MOE=0; do
    bench "Q8_0 $e" $Q8 "$e" -p 512 -n 128 -r 3 -ts $TSQ8
  done
  bench "Q8_0 all on (3)" $Q8 - -p 512 -n 128 -r 3 -ts $TSQ8
fi
if has kquant; then
  bench "UD all on" $UD - -p 512 -n 128 -r 3
  bench "UD GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K" $UD GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K -p 512 -n 128 -r 3
fi
if has long; then
  for e in - GGML_CUDA_FATTN_PASCAL=0 LLAMA_GPU_KQ_MASK=0; do
    bench "32k depth ${e/-/all on}" $K2H "$e" -p 0 -n 64 -d 32768 -r 2
  done
fi
echo "== done $(date +%T)"
} | tee -a "$O/ablation.md"
