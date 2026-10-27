#!/usr/bin/env bash
# Q163 model checks (2026-10-25) on one GPU that runs the build's compute_61 PTX (a V100 here), all in one container:
# logits over one long chunk with the compute-6.1 prompt kernel forced (GGML_CUDA_FATTN_TILE_PASCAL=2) against the stock
# kernels (=0), as KLD and same-top rates of the kernel's logits against the stock logits. -ub 512, so every micro-batch
# past the first runs the prompt kernel at depth.
#   1. Flint imatU-Q4_0 (head 128, sinks, 513-token window), one chunk of $C tokens: F16 cache; then a Q8_0 cache, kernel
#      off and on, both against the F16 stock logits
#   2. Qwen3.6-35B-A3B Q4_0-noK (head 256 on its 10 attention layers), one chunk of $C tokens, F16 cache
# Usage: BUILD=<test build out dir> FLINT=<Flint .gguf> QWEN=<Qwen3.6 .gguf> TEXT=<wikitext-2 wiki.test.raw> WORK=<dir>
#        [C=16384] [GPU=n] [LOCK=file] t2.sh       (logs in $WORK/logs/t2-*.log)
W=${WORK:?WORK=<work dir>}
BUILD=$(realpath "${BUILD:?BUILD=<test build out dir>}")
FLINT=$(realpath "${FLINT:?FLINT=<Flint .gguf>}")
QWEN=$(realpath "${QWEN:?QWEN=<Qwen3.6-35B-A3B .gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wiki.test.raw>}")
C=${C:-16384}
mkdir -p $W/logs $W/cuda-cache $W/kld
${LOCK:+flock $LOCK} docker run --rm --name q163-t2 --gpus "\"device=${GPU:-0}\"" \
    -v "$BUILD:/b:ro" -v $W/cuda-cache:/cc -v $W/kld:/k -v $W/logs:/l -v "$FLINT:/m/flint.gguf:ro" -v "$QWEN:/m/qwen.gguf:ro" \
    -v "$TEXT:/t/wiki.test.raw:ro" -e LD_LIBRARY_PATH=/b/cuda12-libs -e CUDA_CACHE_PATH=/cc \
    -e CUDA_CACHE_MAXSIZE=4294967296 -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e C=$C nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    TXT=/t/wiki.test.raw
    FLINT=/m/flint.gguf
    QWEN=/m/qwen.gguf
    ppl() { # <log> <kernel 0/2> <model> <extra args...>
        local log=$1 s=$2 m=$3
        shift 3
        GGML_CUDA_FATTN_TILE_PASCAL=$s /b/build/bin/llama-perplexity -m $m -f $TXT -c $C --chunks 1 -b 2048 -ub 512 \
            -ngl 99 -fa on "$@" > /l/$log.log 2>&1 || echo "  $log: exit status $?"
        grep -E "Final estimate|Mean    KLD|Maximum KLD|99.9%   KLD|Same top p|RMS Δp" /l/$log.log | head -7 | sed "s/^/    /"
        grep -m1 "prompt attention kernel" /l/$log.log | sed "s/^/    /"
    }
    echo "=== 1. Flint, F16 cache, stock  $(date +%T)"
    ppl t2-flint-f16-off 0 $FLINT --kl-divergence-base /k/flint-f16.kld
    grep -E "n_head|n_head_kv|n_embd_head_k|n_swa" /l/t2-flint-f16-off.log | head -6 | sed "s/^/    /"
    echo "=== 1. Flint, F16 cache, kernel on against stock  $(date +%T)"
    ppl t2-flint-f16-on 2 $FLINT --kl-divergence-base /k/flint-f16.kld --kl-divergence
    for s in 0 2; do
        echo "=== 1. Flint, Q8_0 cache, kernel $s, against the F16 stock logits  $(date +%T)"
        ppl t2-flint-q8_0-$s $s $FLINT -ctk q8_0 -ctv q8_0 --kl-divergence-base /k/flint-f16.kld --kl-divergence
        grep -E "compute buffer size" /l/t2-flint-q8_0-$s.log | grep CUDA | sed "s/^/    /"
    done
    rm -f /k/flint-f16.kld
    echo "=== 2. Qwen3.6, F16 cache, stock  $(date +%T)"
    ppl t2-qwen-f16-off 0 $QWEN --kl-divergence-base /k/qwen-f16.kld
    grep -E "n_head|n_head_kv|n_embd_head_k" /l/t2-qwen-f16-off.log | head -6 | sed "s/^/    /"
    echo "=== 2. Qwen3.6, F16 cache, kernel on against stock  $(date +%T)"
    ppl t2-qwen-f16-on 2 $QWEN --kl-divergence-base /k/qwen-f16.kld --kl-divergence
    rm -f /k/qwen-f16.kld
'
echo "t2 ended $(date +%T)"
