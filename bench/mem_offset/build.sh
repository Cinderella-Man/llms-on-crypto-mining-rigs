#!/usr/bin/env bash
# Builds the two GPU programs of the memory-offset kit for the P106 rig, inside the CUDA 12.9 container (CUDA 13 can no
# longer target Pascal), and packs them with the kit's scripts into one tarball:
#   - mmvq_pascal: bench/mmvq_pascal (streaming read, correctness check and timing of the decode kernels), built as
#     bench/mmvq_pascal/build.sh does;
#   - cuda_memtest: github.com/ComputationalRadiationPhysics/cuda_memtest at a pinned commit, sm_61 code + compute_61 PTX.
# Host code is built without AVX (the rig's Celeron has none) and the CUDA runtime is linked statically, so both need
# only the NVIDIA driver on the rig (cuda_memtest also loads the driver's libnvidia-ml for its temperature monitor).
#
# Usage: bench/mem_offset/build.sh <work dir>       e.g. a scratch folder on a build machine with Docker
# Env: IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), JOBS (default 8), DOCKER_ARGS (e.g. --label key=value),
#      MEMTEST_COMMIT (default below)
# Output: <work dir>/mem_offset-kit-<date>.tar.gz (bin/, the scripts, BUILDINFO.txt, MD5SUMS) and its .md5
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mkdir -p "$1" && cd "$1" && pwd)
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MEMTEST_COMMIT=${MEMTEST_COMMIT:-e94e1ee54e0689c9f154a8a08202554452552ec6}

mkdir -p "$work/src" "$work/kit/bin"
rm -rf "$work/src/mmvq_pascal" && mkdir -p "$work/src/mmvq_pascal" && cp -r "$here/../mmvq_pascal/src" "$work/src/mmvq_pascal/"
[ -d "$work/src/cuda_memtest/.git" ] || git clone -q https://github.com/ComputationalRadiationPhysics/cuda_memtest.git "$work/src/cuda_memtest"
git -C "$work/src/cuda_memtest" checkout -q "$MEMTEST_COMMIT"

docker run --rm ${DOCKER_ARGS:-} -e JOBS="${JOBS:-8}" -v "$work:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake git > /dev/null
    host="-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c"

    nvcc -O3 -std=c++17 -use_fast_math -gencode arch=compute_61,code=[sm_61,compute_61] \
        -Wno-deprecated-gpu-targets -cudart static -Xcompiler "-O2 $host" \
        -o kit/bin/mmvq_pascal src/mmvq_pascal/src/main.cu

    rm -rf src/cuda_memtest/build
    cmake -S src/cuda_memtest -B src/cuda_memtest/build -DCMAKE_CUDA_ARCHITECTURES=61 -DCUDA_MEMTEST_RELEASE=ON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$host" -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -Xcompiler=${host// /,}" \
        > src/cuda_memtest-cmake.log
    cmake --build src/cuda_memtest/build -j "$JOBS" > src/cuda_memtest-build.log
    cp src/cuda_memtest/build/cuda_memtest kit/bin/

    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)"
        echo "cuda_memtest commit: $(git -C src/cuda_memtest rev-parse HEAD)"
        for b in kit/bin/mmvq_pascal kit/bin/cuda_memtest; do
            echo "$b:"
            echo "  avx (ymm) instructions: $(objdump -d $b | grep -c ymm || true)"
            echo "  highest glibc symbol: $(objdump -T $b | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
            echo "  cuda code: $(cuobjdump --list-elf $b 2>/dev/null | grep -o "sm_[0-9]*" | sort -u | tr "\n" " ")$(cuobjdump --list-ptx $b 2>/dev/null | grep -o "sm_[0-9]*" | sort -u | sed "s/^/ptx:/" | tr "\n" " ")"
            echo "  dynamic libraries: $(ldd $b | awk "{print \$1}" | tr "\n" " ")"
        done
    } > kit/BUILDINFO.txt
    chown -R $(stat -c %u:%g /w) kit src
'

cp "$here/nvml_memoc.py" "$here/memoc_steps.sh" "$here/exact_check.sh" "$here/README.md" "$work/kit/"
(cd "$work/kit" && md5sum bin/* *.py *.sh > MD5SUMS)
name=mem_offset-kit-$(date +%Y%m%d)
rm -rf "$work/$name" && cp -r "$work/kit" "$work/$name"
tar czf "$work/$name.tar.gz" -C "$work" "$name"
(cd "$work" && md5sum "$name.tar.gz" > "$name.tar.gz.md5")
cat "$work/kit/BUILDINFO.txt" "$work/kit/MD5SUMS" "$work/$name.tar.gz.md5"
