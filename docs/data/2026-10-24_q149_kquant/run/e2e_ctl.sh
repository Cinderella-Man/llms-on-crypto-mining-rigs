#!/usr/bin/env bash
# Q149 e2e noise-floor controls (2026-10-24), after run_all.sh: the forced run with Q4_K and Q6_K left to llama.cpp
# (GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K), so that only the Q8_0 kernel changes against the base (ppl-skip1); that run's
# logits as a base (ppl-skipbase1); and the fully forced run against it (ppl-on-vs-skip1: the Q4_K and Q6_K kernels only).
# Usage: the environment of e2e.sh (BUILD, MODEL, TEXT, WORK, GPU), then e2e_ctl.sh
W=${WORK:?WORK=<work dir>}
BUILD=$(realpath "${BUILD:?BUILD=<unpacked build with bin/>}")
MODEL=$(realpath "${MODEL:?MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wiki.test.raw>}")
L=$W/e2e/logs
mkdir -p $L $W/kld $W/nvcache
M=/m/model.gguf
PPL="/b/bin/llama-perplexity -m $M -ngl 2 -ot output\.weight=CUDA0 -t 8 -fa on -lm none -f /t/wiki.test.raw -c 256 --chunks 1 -b 256 -ub 1"
run() { # skip list, command...
    local skip=$1; shift
    docker run --rm --name q149-e2e --gpus "\"device=${GPU:-0}\"" \
        -v "$BUILD:/b:ro" -v "$MODEL:$M:ro" -v "$TEXT:/t/wiki.test.raw:ro" -v "$W/kld:/k" -v "$W/nvcache:/nv" \
        -e GGML_CUDA_MMVQ_PASCAL=2 -e GGML_CUDA_MMVQ_PASCAL_SKIP=$skip \
        -e CUDA_CACHE_PATH=/nv -e CUDA_CACHE_MAXSIZE=4294967296 \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 900 "$@"
}
echo "== ppl-skip1 $(date +%T)" >> $L/progress.txt
run q4_K,q6_K $PPL --kl-divergence-base /k/qwen36-off-ub1.kld --kl-divergence > $L/ppl-skip1.log 2>&1
echo "   exit $? $(date +%T)" >> $L/progress.txt
echo "== ppl-skipbase1 $(date +%T)" >> $L/progress.txt
run q4_K,q6_K $PPL --kl-divergence-base /k/qwen36-skip-ub1.kld > $L/ppl-skipbase1.log 2>&1
echo "   exit $? $(date +%T)" >> $L/progress.txt
echo "== ppl-on-vs-skip1 $(date +%T)" >> $L/progress.txt
run "" $PPL --kl-divergence-base /k/qwen36-skip-ub1.kld --kl-divergence > $L/ppl-on-vs-skip1.log 2>&1
echo "   exit $? $(date +%T)" >> $L/progress.txt
