#!/usr/bin/env bash
# build.sh: builds ple_tok against a CPU-only libllama (vocabulary loading only needs the CPU) and ple_sim, in Docker.
#   SRC=<llama.cpp git checkout> REF=<commit or branch> OUT=<dir> [JOBS=8] [DOCKER_ARGS=...] bench/ngram_table/build.sh
# Writes OUT/ple_tok, OUT/ple_sim and OUT/BUILDINFO.txt. The tokenizer must come from a tree that knows the model's
# architecture and pre-tokenizer (for Flash-Next: qwen4exp and qwen35).
set -euo pipefail
SRC=${SRC:?a llama.cpp git checkout}
REF=${REF:-HEAD}
OUT=${OUT:?output folder}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
docker run --rm ${DOCKER_ARGS:---cpu-shares=256} -e REF="$REF" -e JOBS="${JOBS:-8}" \
    -v "$(cd "$SRC" && pwd):/src:ro" -v "$HERE:/tool:ro" -v "$(cd "$OUT" && pwd):/out" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null
    mkdir -p /b/llama && cd /b
    git -c safe.directory="*" -C /src archive "$REF" | tar -x -C /b/llama
    sha=$(git -c safe.directory="*" -C /src rev-parse --short=9 "$REF")
    cmake -S llama -B llama/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF > /dev/null
    nice -n 19 cmake --build llama/build -j "$JOBS" --target llama > /dev/null
    libs=$(for l in libllama.a libggml.a libggml-cpu.a libggml-base.a; do find llama/build -name "$l"; done)
    nice -n 19 g++ -O2 -std=c++17 /tool/ple_tok.cpp -I llama/include -I llama/ggml/include -o /out/ple_tok \
        -Wl,--start-group $libs -Wl,--end-group -lpthread -ldl
    nice -n 19 g++ -O3 -std=c++17 -Wall /tool/ple_sim.cpp -o /out/ple_sim
    printf "llama.cpp %s (%s), CPU only\nple_tok.cpp %s\nple_sim.cpp %s\n" "$sha" "$REF" \
        "$(sha256sum /tool/ple_tok.cpp | cut -c1-16)" "$(sha256sum /tool/ple_sim.cpp | cut -c1-16)" > /out/BUILDINFO.txt
    chown -R "$(stat -c %u:%g /out)" /out
'
cat "$OUT/BUILDINFO.txt"
