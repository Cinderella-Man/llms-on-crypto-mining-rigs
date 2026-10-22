#!/usr/bin/env bash
# Compares the quality of quantized GGUF files against a higher-precision reference, on any x86-64 Linux PC with
# Docker and enough RAM for the reference model (it runs on the CPU, not on the rig). Quality depends only on the
# weights, so the rig stays free for speed tests.
#
# For each candidate it reports llama.cpp's perplexity on wikitext-2 and the KL divergence of its next-token
# distribution from the reference's (how far the quantized model's predictions drift from the reference's), plus how
# often both pick the same top token.
#
#   scripts/quality_kld.sh <llama.cpp source dir> <models dir> <work dir> <reference.gguf> <candidate.gguf>...
#   (model paths relative to <models dir>; <work dir> must contain wikitext-2-raw/wiki.test.raw)
#
# Env: CHUNKS (default 80 chunks of 512 tokens), THREADS (default: number of physical cores),
#      CAND_ARGS (extra llama-perplexity options for the candidates only, e.g.
#      "--override-kv flint.expert_used_count=int:4"), TAG (suffix for the candidates' log names).
set -euo pipefail

SRC=$(realpath "$1"); MODELS=$(realpath "$2"); WORK=$(realpath "$3"); REF=$4; shift 4
CHUNKS=${CHUNKS:-80}
THREADS=${THREADS:-$(lscpu -p=Core | grep -v '^#' | sort -u | wc -l)}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}

docker run --rm -e CHUNKS="$CHUNKS" -e THREADS="$THREADS" -e REF="$REF" -e CANDS="$*" -e HOST_UID="$(id -u)" \
    -e CAND_ARGS="${CAND_ARGS:-}" -e TAG="${TAG:-}" \
    -v "$SRC:/src" -v "$MODELS:/models:ro" -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake build-essential > /dev/null
    cmake -S /src -B /src/build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
    cmake --build /src/build-cpu -j "$(nproc)" --target llama-perplexity > /dev/null
    P=/src/build-cpu/bin/llama-perplexity
    TEXT=/work/wikitext-2-raw/wiki.test.raw
    base=/work/$(basename "$REF" .gguf).c512x$CHUNKS.kld

    if [ ! -f "$base" ]; then
        echo "== reference: $REF"
        $P -m "/models/$REF" -f "$TEXT" -c 512 --chunks "$CHUNKS" -t "$THREADS" --kl-divergence-base "$base" \
            > /work/$(basename "$REF" .gguf).ppl.log 2>&1
        grep -E "Final estimate" /work/$(basename "$REF" .gguf).ppl.log
    fi
    for c in $CANDS; do
        echo "== candidate: $c $CAND_ARGS"
        log=/work/$(basename "$c" .gguf)$TAG.kld.log
        $P -m "/models/$c" -f "$TEXT" -c 512 --chunks "$CHUNKS" -t "$THREADS" --kl-divergence-base "$base" \
            --kl-divergence $CAND_ARGS > "$log" 2>&1
        grep -E "^(Mean PPL|Mean ln|Mean +KLD|Median +KLD|99.0%  +KLD|Maximum +KLD|Same top p|RMS Δp)" "$log" || tail -20 "$log"
    done
    chown -R "$HOST_UID" /work /src/build-cpu
'
