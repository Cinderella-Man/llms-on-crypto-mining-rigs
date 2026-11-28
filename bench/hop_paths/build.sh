#!/usr/bin/env bash
# Builds vram_write and memops_check in the CUDA 12.9 devel image (Ubuntu 24.04, the rigs' glibc) with the Vulkan
# headers and loader, into OUT (default ./out), with an MD5SUMS file. No GPU is needed to build.
# Env: IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), DOCKER_ARGS (e.g. labels, --cpu-shares).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$here/out}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
docker run --rm ${DOCKER_ARGS:-} -v "$here:/src:ro" -v "$OUT:/out" "$IMAGE" bash -c '
  set -e
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan-dev > /dev/null
  cd /src
  g++ -O2 -std=c++17 -Wall vram_write.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
    -lcudart_static -ldl -lrt -lpthread -lvulkan -o /out/vram_write
  g++ -O2 -Wall memops_check.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -o /out/memops_check
  cd /out && md5sum vram_write memops_check > MD5SUMS && cat MD5SUMS
  objdump -T vram_write memops_check | grep -o "GLIBC_[0-9.]*" | sort -uV | tail -1'
