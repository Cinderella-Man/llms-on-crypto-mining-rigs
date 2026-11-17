#!/usr/bin/env bash
# Builds build/rated_clock: one static x86_64 binary (glibc linked in, so it needs no library on the rig), no AVX
# (the rig's Celeron has none), compiled in a container with no network. Nothing is run; build/BUILDINFO.txt
# describes the binary (md5, ELF header, no program interpreter, AVX count).
#
# With NVSRC set to a checkout of NVIDIA's open-gpu-kernel-modules at tag 580.178.04, it first compiles
# abi_upstream.c (syntax only, nothing is linked or run) so the asserts in abi_check.h hold for NVIDIA's own
# headers as well as for the excerpts in nvidia/. Only headers are read from that checkout.
#
# Usage: bench/rated_clock/build.sh
#   env IMAGE        container image with gcc and a static glibc (default nvidia/cuda:12.9.2-devel-ubuntu24.04,
#                    Ubuntu 24.04, the rigs' release)
#   env NVSRC        optional, the upstream checkout described above
#   env DOCKER_ARGS  extra docker run options, e.g. "--cpu-shares=256"
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
rm -f build/rated_clock build/rated_clock.o build/BUILDINFO.txt
NVMOUNT=()
if [ -n "${NVSRC:-}" ]; then
    grep -qx 'NVIDIA_VERSION = 580.178.04' "$NVSRC/version.mk" || { echo "NVSRC is not at 580.178.04" >&2; exit 1; }
    NVMOUNT=(-v "$(cd "$NVSRC" && pwd):/nvsrc:ro")
fi
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
# shellcheck disable=SC2086
docker run --rm --network none ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE="$IMAGE" \
    -e IMAGE_ID="$IMAGE_ID" -v "$PWD:/w" -w /w "${NVMOUNT[@]}" "$IMAGE" bash -euo pipefail -c '
    FLAGS="-std=gnu11 -O2 -Wall -Wextra -Werror -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-fma"
    UPSTREAM="not run (NVSRC unset)"
    if [ -d /nvsrc ]; then
        gcc $FLAGS -fsyntax-only -I /nvsrc/src/common/sdk/nvidia/inc -I /nvsrc/kernel-open/common/inc \
            -I /nvsrc/src/nvidia/arch/nvalloc/unix/include abi_upstream.c
        UPSTREAM="passed: abi_check.h holds for the upstream headers at $(sed -n "s/^NVIDIA_VERSION = //p" /nvsrc/version.mk)"
    fi
    gcc $FLAGS -c -o build/rated_clock.o rated_clock.c
    gcc $FLAGS -static -o build/rated_clock build/rated_clock.o
    {
        echo "built: $(date -Is) in $IMAGE ($IMAGE_ID)"
        echo "compiler: $(gcc --version | head -n 1); C library: $(ldd --version | head -n 1)"
        echo "flags: $FLAGS -static"
        echo "upstream header check: $UPSTREAM"
        echo "md5: $(md5sum build/rated_clock | cut -d " " -f 1)  size: $(stat -c %s build/rated_clock) bytes"
        echo "ldd: $(ldd build/rated_clock 2>&1 | tr -s " \t" " " || true)"
        echo "program interpreter (INTERP) entries: $(readelf -l build/rated_clock | grep -c INTERP || true)"
        echo "ELF: $(readelf -h build/rated_clock | sed -n "s/^ *\(Class\|Machine\|Type\): *//p" | paste -sd " ")"
        echo "avx (ymm) instructions in the tool own code: $(objdump -d build/rated_clock.o | grep -c ymm || true)" \
             "(the static glibc picks its AVX string functions at run time, only on CPUs that have AVX)"
    } > build/BUILDINFO.txt
    cat build/BUILDINFO.txt
'
