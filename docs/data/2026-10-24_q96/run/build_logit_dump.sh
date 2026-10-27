#!/usr/bin/env bash
# Q96 (2026-10-24): builds llama-logit-dump (bench/ngram_replay/logit_dump/logit_dump.cpp, copied into WORK) against a
# llama.cpp tree's CUDA build (LLAMA_SRC with build-cuda/; here d77256999 + the rescale switch, the build Q51 generated
# its transcripts with), mounted read-only; compiles the one source file and links it the way CMake links
# llama-route-dump there. CPU only.
set -euo pipefail
LLAMA_SRC=${LLAMA_SRC:?LLAMA_SRC=<llama.cpp tree with a CUDA build in build-cuda/>}
WORK=${WORK:?WORK=<work dir holding logit_dump.cpp; the binary goes to bin/>}
docker run --rm --name q96-logit-build --label pascal-quants=1 --cpus 8 \
  -v "$LLAMA_SRC":/llama:ro -v "$WORK":/w \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
  S=/llama; B=$S/build-cuda; C=/usr/local/cuda-12.9/targets/x86_64-linux/lib
  mkdir -p /w/bin
  c++ -O3 -DNDEBUG -std=c++17 -DGGML_USE_CPU -DGGML_USE_CUDA -I$S/common -I$S/vendor -I$S/include -I$S/ggml/include \
    -c /w/logit_dump.cpp -o /w/bin/logit_dump.o
  c++ -O3 /w/bin/logit_dump.o -o /w/bin/llama-logit-dump -Wl,-rpath,$C: $B/common/libllama-common.a $B/src/libllama.a \
    $B/ggml/src/libggml.a -ldl $B/ggml/src/libggml-cpu.a $B/ggml/src/ggml-cuda/libggml-cuda.a $B/ggml/src/libggml-base.a \
    /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a -lm $C/libcudart.so $C/libcublas.so \
    $C/libcublasLt.so $C/libculibos.a /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
    /usr/lib/x86_64-linux-gnu/libnccl.so $B/common/libllama-common-base.a $B/vendor/cpp-httplib/libcpp-httplib.a
  rm /w/bin/logit_dump.o
  cat $B/BUILDINFO.txt > /w/bin/BUILDINFO.txt
  ls -la /w/bin
'
