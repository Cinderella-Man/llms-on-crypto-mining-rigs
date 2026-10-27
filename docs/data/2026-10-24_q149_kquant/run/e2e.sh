#!/usr/bin/env bash
# Q149 end to end (2026-10-24): Qwen3.6-35B-A3B Q4_K_M (ggml-org) with its last 2 layers on one GPU (compute_61 PTX of
# an sm_61 build; < 2.5 GB of VRAM), the other 38 layers on the CPU, loaded into RAM (-lm none). Pascal kernels off
# (GGML_CUDA_MMVQ_PASCAL=0) against forced (2): KLD over 256 wikitext tokens (the last 128 scored) decoded one token per
# step (-ub 1) and four per step (-ub 4), and greedy text of 2 prompts x 48 tokens (prompts/ next to this script).
# The -ot override meant to put the Q6_K output head on the GPU did not take (placement.sh), so the head ran on the CPU.
# Usage: BUILD=<unpacked rig build, fork branch pascal-p106-v2-kquant> MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf>
#        TEXT=<wiki.test.raw> WORK=<dir> [GPU=n] e2e.sh <step>
#   steps: help, ppl-off1, ppl-off1b, ppl-on1, ppl-off4, ppl-on4, gen-off, gen-on (run_all.sh runs them in order)
set -uo pipefail
W=${WORK:?WORK=<work dir>}
BUILD=$(realpath "${BUILD:?BUILD=<unpacked build with bin/>}")
MODEL=$(realpath "${MODEL:?MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wiki.test.raw>}")
P=$(cd "$(dirname "$0")" && pwd)/prompts
mkdir -p $W/e2e $W/kld $W/nvcache
B=/b/bin
M=/m/model.gguf
D=/k
T=/t/wiki.test.raw
COMMON="-m $M -ngl 2 -ot output\.weight=CUDA0 -t 8 -fa on -lm none"
step=$1
run() { # mode, command...
    local mode=$1; shift
    docker run --rm --name q149-e2e --gpus "\"device=${GPU:-0}\"" \
        -v "$BUILD:/b:ro" -v "$MODEL:$M:ro" -v "$TEXT:$T:ro" -v "$W/kld:/k" -v "$P:/p:ro" -v "$W/nvcache:/nv" \
        -e GGML_CUDA_MMVQ_PASCAL=$mode -e CUDA_CACHE_PATH=/nv -e CUDA_CACHE_MAXSIZE=4294967296 \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 900 "$@"
}
PPL="$B/llama-perplexity $COMMON -f $T -c 256 --chunks 1 -b 256"
case $step in
    help)      run 0 $B/llama-completion --help ;;
    ppl-off1)  run 0 $PPL -ub 1 --kl-divergence-base $D/qwen36-off-ub1.kld ;;
    ppl-off1b) run 0 $PPL -ub 1 --kl-divergence-base $D/qwen36-off-ub1.kld --kl-divergence ;;
    ppl-on1)   run 2 $PPL -ub 1 --kl-divergence-base $D/qwen36-off-ub1.kld --kl-divergence ;;
    ppl-off4)  run 0 $PPL -ub 4 --kl-divergence-base $D/qwen36-off-ub4.kld ;;
    ppl-on4)   run 2 $PPL -ub 4 --kl-divergence-base $D/qwen36-off-ub4.kld --kl-divergence ;;
    gen-off|gen-on)
        mode=0; [ $step = gen-on ] && mode=2
        for p in 1 3; do
            run $mode $B/llama-completion $COMMON -f /p/p$p.txt -n 48 --temp 0 --seed 1 -c 1024 -ub 32 \
                -no-cnv --ignore-eos --no-display-prompt > $W/e2e/$step-p$p.txt 2> $W/e2e/$step-p$p.err
            echo "p$p exit $?"
        done ;;
esac
