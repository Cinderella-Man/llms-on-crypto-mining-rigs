#!/usr/bin/env bash
# Builds llama.cpp with its Vulkan backend for the P106 rig on any x86-64 Linux PC with Docker, then packs the binaries
# into a tarball you copy to the rig. It is the Vulkan counterpart of build_llamacpp_docker.sh (CUDA); the two builds
# can sit side by side under /opt for an A/B test (docs/vulkan_on_the_rig.md).
#
# The build targets exactly what the rig can run:
#   - Ubuntu 24.04 userland, so the binaries match the rig's glibc and libstdc++; GCC 14 as in llama.cpp's own
#     Ubuntu 24.04 Vulkan CI;
#   - the LunarG Vulkan SDK for headers and the shader compiler. Ubuntu 24.04's own glslc (shaderc 2023.8) cannot
#     compile GL_EXT_integer_dot_product, and without it llama.cpp leaves out the integer dot-product shaders, the
#     path that uses the cards' fast 8-bit dot product (dp4a). The build fails if that extension is missing;
#   - CPU code without AVX/AVX2/FMA/F16C, because the rig's Celeron has none of them (an AVX binary dies with
#     "Illegal instruction"). The build fails if any shipped binary contains a VEX/EVEX-encoded (AVX-family)
#     instruction;
#   - llama/ggml linked statically, all shaders compiled into the binaries. At run time the binaries need only the
#     Vulkan loader (libvulkan.so.1, Ubuntu package libvulkan1) and the NVIDIA driver's Vulkan ICD, which ships with
#     the 580 driver. A copy of Ubuntu 24.04's loader goes into lib/ of the tarball for rigs without libvulkan1
#     (use it with LD_LIBRARY_PATH=<dir>/lib).
#
# Usage: scripts/build_llamacpp_vulkan_docker.sh [git-ref]
#   git-ref   llama.cpp branch, tag or commit (default: master)
# Env PATCHES: extra patch files (paths relative to $WORK, applied with git apply), e.g. PATCHES=patches/x.diff
#     SRC: source checkout under $WORK (default src-vulkan; kept separate from the CUDA script's src)
#     VULKAN_SDK_VERSION: LunarG SDK release (default 1.4.341.0); the ~290 MB tarball is downloaded once and kept
#     in $WORK/vulkan-sdk/
# Output: $WORK/llama.cpp-<date>-<sha>-vulkan[-<patch>].tar.gz with bin/, lib/ and BUILDINFO.txt (WORK defaults to
# ~/llama-builds). bin/ holds llama-bench, llama-server, llama-cli, llama-completion, llama-perplexity,
# llama-batched-bench, llama-gguf, test-backend-ops (per-operation support, correctness and speed tests) and
# test-export-graph-ops (writes every operation of a model's graph to a file that test-backend-ops --test-file reads).
# On the rig: tar xzf <tarball> -C /opt && NODEVICE_SELECT=1 /opt/<name>/bin/llama-cli --list-devices
# (always pick the cards with GGML_VK_VISIBLE_DEVICES; see the device numbering section of the doc)
set -euo pipefail

REF=${1:-master}
IMAGE=${IMAGE:-ubuntu:24.04}
WORK=${WORK:-$HOME/llama-builds}
SRC=${SRC:-src-vulkan}
VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.341.0}
mkdir -p "$WORK/ccache" "$WORK/vulkan-sdk"

docker run --rm -e REF="$REF" -e PATCHES="${PATCHES:-}" -e SRC="$SRC" -e IMAGE="$IMAGE" \
    -e SDK_VER="$VULKAN_SDK_VERSION" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache CC=gcc-14 CXX=g++-14
    apt-get update -qq && apt-get install -y -qq git cmake build-essential gcc-14 g++-14 ccache curl ca-certificates \
        xz-utils libvulkan1 > /dev/null

    sdk_tar=/work/vulkan-sdk/vulkan_sdk-$SDK_VER.tar.xz
    if [ ! -s "$sdk_tar" ]; then
        curl -fsSL -o "$sdk_tar.part" "https://sdk.lunarg.com/sdk/download/$SDK_VER/linux/vulkan_sdk.tar.xz"
        mv "$sdk_tar.part" "$sdk_tar"
    fi
    mkdir -p /opt/vulkan-sdk && tar xJf "$sdk_tar" -C /opt/vulkan-sdk --strip-components=1
    set +u; source /opt/vulkan-sdk/setup-env.sh > /dev/null; set -u

    src=/work/$SRC
    [ -d "$src/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$src"
    cd "$src"
    git config --global --add safe.directory "$src"
    git fetch -q origin && git checkout -q --force "$REF" && git reset -q --hard
    git clean -qfdx
    [ "$REF" = master ] && git reset -q --hard origin/master
    sha=$(git rev-parse --short=9 HEAD)
    name=llama.cpp-$(date +%Y%m%d)-$sha-vulkan
    for p in $PATCHES; do
        git apply -v "/work/$p"
        name=$name-$(basename "$p" .diff)
    done

    targets="llama-bench llama-server llama-cli llama-completion llama-perplexity llama-batched-bench llama-gguf"
    tests="test-backend-ops test-export-graph-ops"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_VULKAN=ON -DGGML_CUDA=OFF \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_SKIP_BUILD_RPATH=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /tmp/cmake.log
    grep -q "GL_EXT_integer_dot_product supported by glslc" /tmp/cmake.log \
        || { echo "ERROR: glslc cannot compile GL_EXT_integer_dot_product"; exit 1; }
    cmake --build build -j "$(nproc)" --target $targets $tests > /dev/null

    out=/work/$name
    rm -rf "$out" && mkdir -p "$out/bin" "$out/lib"
    for t in $targets $tests; do cp "build/bin/$t" "$out/bin/"; done
    cp -L /usr/lib/x86_64-linux-gnu/libvulkan.so.1 "$out/lib/"

    # AVX, AVX2, FMA, F16C and AVX-512 instructions are all VEX/EVEX-encoded, and objdump prints their mnemonics with
    # a leading "v" (vmovaps, vfmadd231ps, vcvtph2ps, ...). Plain SSE mnemonics never start with "v".
    for f in "$out"/bin/*; do
        echo "  $(basename "$f"): $(objdump -d --no-show-raw-insn "$f" | awk "\$2 ~ /^v/" | wc -l)"
    done > /tmp/avx.txt
    avx_total=$(awk "{s += \$2} END {print s + 0}" /tmp/avx.txt)
    {
        echo "name: $name"
        echo "ref: $REF"
        echo "commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
        echo "patches: ${PATCHES:-none}"
        echo "image: $IMAGE"
        echo "compiler: $($CXX --version | head -1)"
        echo "vulkan sdk: $SDK_VER ($(grep -m1 -o "VK_HEADER_VERSION [0-9]*" "$VULKAN_SDK/include/vulkan/vulkan_core.h"))"
        echo "glslc: $(glslc --version | head -1)"
        echo "glslc shader extensions (from cmake):"; grep "by glslc" /tmp/cmake.log | sed "s/^-- /  /"
        echo "built: $(date -Is)"
        echo "AVX-family (VEX/EVEX) instructions, must all be 0: total $avx_total"
        cat /tmp/avx.txt
        echo "ymm/zmm register uses in llama-bench: $(objdump -d build/bin/llama-bench | grep -cE "%[yz]mm")"
        echo "highest glibc symbol: $(objdump -T build/bin/llama-bench | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        echo "highest libstdc++ symbol: $(objdump -T build/bin/llama-bench | grep -o "GLIBCXX_[0-9.]*" | sort -V | tail -1)"
        echo "dynamic libraries (no RPATH):"; env -u LD_LIBRARY_PATH ldd build/bin/llama-bench | sed "s/^/  /"
        echo "lib/libvulkan.so.1: Ubuntu 24.04 Vulkan loader $(dpkg-query -W -f="\${Version}" libvulkan1), for rigs without libvulkan1"
    } > "$out/BUILDINFO.txt"
    tar czf "/work/$name.tar.gz" -C /work "$name"
    chown -R "$HOST_UID:$HOST_GID" "$src" "$out" "/work/$name.tar.gz" /work/vulkan-sdk
    chown -R "$HOST_UID:$HOST_GID" /work/ccache 2>/dev/null || true
    cat "$out/BUILDINFO.txt"
    if [ "$avx_total" -ne 0 ]; then echo "ERROR: AVX-family instructions found; these binaries crash on the rig"; exit 1; fi
'
