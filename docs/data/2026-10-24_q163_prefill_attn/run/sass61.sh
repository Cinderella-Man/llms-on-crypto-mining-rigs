#!/usr/bin/env bash
# Q163 (2026-10-24): compile the compute-6.1 prompt attention kernel (fattn-tile-pascal.cu) alone for sm_61 (CPU only,
# low priority) and print ptxas' registers, spills and shared memory per kernel instance; keeps the SASS in
# $WORK/sass/ftp61.sass for sass_stock61.sh's loop comparison.
# Usage: SRC=<llama.cpp tree, fork branch pascal-p106-v2-prefill-attn> WORK=<dir> sass61.sh
SRC=$(realpath "${SRC:?SRC=<llama.cpp source tree>}")
W=${WORK:?WORK=<work dir>}
mkdir -p $W/sass
docker run --rm --name q163-sass --cpu-shares=256 -v "$SRC:/src:ro" -v "$W/sass:/o" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    cd /src/ggml/src/ggml-cuda
    nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_61 -use_fast_math -extended-lambda -I/src/ggml/include -I/src/ggml/src \
        -I/src/ggml/src/ggml-cuda -Xptxas -v -c fattn-tile-pascal.cu -o /o/ftp61.o > /o/ptxas.txt 2>&1
    grep -E "Compiling entry|registers|spill" /o/ptxas.txt | grep -B2 -A0 "registers" | grep -E "flash_attn_tile_pascal|registers|spill" | head -40
    cuobjdump -sass /o/ftp61.o > /o/ftp61.sass
    echo "SASS lines: $(wc -l < /o/ftp61.sass)"
'
