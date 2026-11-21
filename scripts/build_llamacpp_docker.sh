#!/usr/bin/env bash
# Builds llama.cpp for the P106 rig on any x86-64 Linux PC with Docker, then packs the binaries into a tarball you
# copy to the rig. Building on the rig itself works too, but its 2-core Celeron is slow at it.
#
# The build targets exactly what the rig can run:
#   - CUDA 12.9 inside the container (CUDA 13 can't build for Pascal), code for compute capability 6.1 only;
#   - CPU code without AVX/AVX2/FMA/F16C, because the rig's Celeron has none of them (an AVX binary dies with
#     "Illegal instruction");
#   - Ubuntu 24.04 userland, so the binaries match the rig's glibc;
#   - llama/ggml linked statically; only the CUDA runtime libraries (libcudart, libcublas) come from the rig's own
#     CUDA 12.9 installation.
#
# llama-server's web page (its built-in chat UI) is not in recent llama.cpp sources: the build downloads it from
# upstream's archive on Hugging Face by release number (b<N>) and, when that release has no archive, takes the newest
# one, which need not fit an older tree. A fork's own commit count is no upstream release, so the script asks for the
# upstream release the tree sits on (git describe --tags --abbrev=0 --match 'b[0-9]*' on the checked-out tree) and
# prints which one it asked for and which one the build took. A tree whose clone lacks upstream's b<N> tags gets a
# warning and the unpinned behaviour; fetch them with git -C $WORK/src fetch --tags origin. Trees that carry their own
# web page download nothing and are not affected.
#
# Usage: scripts/build_llamacpp_docker.sh [git-ref] [--cuda-graphs-pascal]
#   git-ref               llama.cpp branch, tag or commit (default: master); a branch of a fork as <remote>/<branch>,
#                         e.g. fork/pascal-p106, once the fork is a remote of $WORK/src with an https fetch URL
#                         (git -C ~/llama-builds/src remote add fork https://github.com/<you>/llama.cpp)
#   --cuda-graphs-pascal  also apply llama.cpp PR #27721, which re-enables CUDA graphs on Pascal (compute 6.1)
# Env PATCHES: extra patch files (paths relative to $WORK, applied with git apply), e.g. PATCHES=patches/x.diff
#     TESTS=1: also build test-backend-ops (per-operation correctness and speed tests, e.g. `-o MUL_MAT perf`)
#     EXTRA_TARGETS="t1 t2": also build these targets (tests need TESTS=1) and copy them into bin/, e.g. a test the
#     rig runs on its cards
#     SCHED_COPIES=N: ggml scheduler input copies for pipeline parallelism (default 4); each copy holds the inputs of
#     one ubatch, the attention mask included, so fewer copies leave room for a larger -ub at long context
#     JOBS=N: parallel compile jobs (default: all cores), e.g. on a shared build machine
#     RPC=1: also build ggml-rpc-server with -DGGML_RPC=ON, so llama-server can use the cards of other rigs (--rpc);
#     the fork needs its Q743 branch or later for this
#     HF_UI_VERSION=b<N>: the upstream release whose web page the build embeds, instead of the one the tree sits on;
#     a -e HF_UI_VERSION=... in DOCKER_ARGS wins over this
#     DOCKER_ARGS: extra `docker run` arguments, e.g. a label the build machine's policy asks for
# Output: $WORK/llama.cpp-<date>-<sha>[-graphs61][-<patch>].tar.gz with bin/ and BUILDINFO.txt (WORK defaults to
# ~/llama-builds); BUILDINFO.txt names the web page's release.
# On the rig: tar xzf <tarball> -C /opt && /opt/<name>/bin/llama-server --list-devices
set -euo pipefail

REF=${1:-master}
PATCH=${2:-}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
WORK=${WORK:-$HOME/llama-builds}
mkdir -p "$WORK/ccache"
# an HF_UI_VERSION from the environment goes into the container, unless DOCKER_ARGS passes its own
UI_ENV=()
if [ -n "${HF_UI_VERSION:-}" ] && [[ " ${DOCKER_ARGS:-} " != *HF_UI_VERSION=* ]]; then
    UI_ENV=(-e "HF_UI_VERSION=$HF_UI_VERSION")
fi

docker run --rm ${DOCKER_ARGS:-} ${UI_ENV[@]+"${UI_ENV[@]}"} -e REF="$REF" -e PATCH="$PATCH" -e PATCHES="${PATCHES:-}" -e TESTS="${TESTS:-0}" -e EXTRA_TARGETS="${EXTRA_TARGETS:-}" -e SCHED_COPIES="${SCHED_COPIES:-}" -e RPC="${RPC:-0}" -e JOBS="${JOBS:-}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache curl > /dev/null

    [ -d /work/src/.git ] || git clone -q https://github.com/ggml-org/llama.cpp /work/src
    cd /work/src
    git config --global --add safe.directory /work/src
    git fetch -q origin
    case "$REF" in */*) git fetch -q "${REF%%/*}" ;; esac
    git checkout -q --force "$REF" && git reset -q --hard
    git clean -qfdx
    [ "$REF" = master ] && git reset -q --hard origin/master
    sha=$(git rev-parse --short=9 HEAD)
    name=llama.cpp-$(date +%Y%m%d)-$sha
    case "$REF" in */*) name=$name-${REF##*/} ;; esac

    # the web page: the release given, else the upstream release this tree sits on (see the header)
    if [ -n "${HF_UI_VERSION:-}" ]; then
        ui_from="given"
    elif HF_UI_VERSION=$(git describe --tags --abbrev=0 --match "b[0-9]*" HEAD 2>/dev/null); then
        ui_from="the upstream release this tree sits on"
    else
        HF_UI_VERSION=
        ui_from="no b<N> tag in this tree"
        echo "WARNING: no upstream release tag (b<N>) in the history of $REF, so the web page is not pinned: the build" \
            "asks for b<its commit count>, then the newest archive, which may not fit this tree. Fetch the tags" \
            "(git -C \$WORK/src fetch --tags origin) or pass HF_UI_VERSION=b<N>." >&2
    fi
    if [ -n "$HF_UI_VERSION" ]; then export HF_UI_VERSION; else unset HF_UI_VERSION; fi
    echo "web page: HF_UI_VERSION=${HF_UI_VERSION:-unset} ($ui_from)"

    if [ "$PATCH" = --cuda-graphs-pascal ]; then
        curl -sL https://patch-diff.githubusercontent.com/raw/ggml-org/llama.cpp/pull/27721.diff | git apply -v \
            || sed -i "s/devices\[cuda_ctx->device\].cc < GGML_CUDA_CC_VOLTA) {/devices[cuda_ctx->device].cc < GGML_CUDA_CC_DP4A) {/" ggml/src/ggml-cuda/ggml-cuda.cu
        grep -q "cc < GGML_CUDA_CC_DP4A) {" ggml/src/ggml-cuda/ggml-cuda.cu || { echo "patch failed"; exit 1; }
        name=$name-graphs61
    fi
    [ -n "$SCHED_COPIES" ] && name=$name-sched$SCHED_COPIES
    [ "$RPC" = 1 ] && case "$name" in *rpc*) ;; *) name=$name-rpc ;; esac
    for p in $PATCHES; do
        git apply -v "/work/$p"
        name=$name-$(basename "$p" .diff)
    done

    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=$([ "$TESTS" = 1 ] && echo ON || echo OFF) \
        $([ -n "$SCHED_COPIES" ] && echo -DGGML_SCHED_MAX_COPIES=$SCHED_COPIES) \
        $([ "$RPC" = 1 ] && echo -DGGML_RPC=ON) \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
    cmake --build build -j "${JOBS:-$(nproc)}" --target llama-bench llama-server llama-cli llama-completion \
        llama-perplexity llama-batched-bench llama-gguf $([ "$TESTS" = 1 ] && echo test-backend-ops) $([ "$RPC" = 1 ] && echo ggml-rpc-server) \
        $([ -d examples/pipespec ] && echo llama-pipespec) $([ -d examples/screen-check ] && echo llama-screen-check) \
        $(grep -qs llama-pipe-drafter-check examples/pipespec/CMakeLists.txt && echo llama-pipe-drafter-check) $EXTRA_TARGETS \
        | { grep "^-- UI:" || true; } | tee /tmp/ui.log
    # the release whose archive the build took ("UI: HF download succeeded, stamp updated (<bucket>|<asked>, resolved: <took>)")
    ui_took=$(sed -n "s/.*, resolved: \(.*\))$/\1/p" /tmp/ui.log | tail -n 1)
    if [ ! -s /tmp/ui.log ]; then
        ui_page="none downloaded (the tree carries its own)"
    else
        ui_page="asked ${HF_UI_VERSION:-nothing} ($ui_from), took ${ui_took:-none}"
        [ "$ui_took" = "${HF_UI_VERSION:-}" ] || echo "WARNING: web page: $ui_page" >&2
    fi
    echo "web page: $ui_page"

    out=/work/$name
    rm -rf "$out" && mkdir -p "$out/bin"
    cp build/bin/llama-* "$out/bin/"
    [ "$TESTS" = 1 ] && cp build/bin/test-backend-ops "$out/bin/"
    for t in $EXTRA_TARGETS; do cp "build/bin/$t" "$out/bin/"; done
    [ "$RPC" = 1 ] && cp build/bin/ggml-rpc-server "$out/bin/"
    {
        echo "name: $name"
        echo "ref: $REF"
        echo "commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
        echo "patch: ${PATCH:-none}"
        echo "web page: $ui_page"
        echo "image: '"$IMAGE"'"
        echo "built: $(date -Is)"
        echo "avx instructions in llama-bench: $(objdump -d build/bin/llama-bench | grep -c ymm)"
        echo "highest glibc symbol: $(objdump -T build/bin/llama-bench | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        echo "dynamic libraries:"; ldd build/bin/llama-bench | sed "s/^/  /"
    } > "$out/BUILDINFO.txt"
    tar czf "/work/$name.tar.gz" -C /work "$name"
    chown -R "$HOST_UID:$HOST_GID" /work
    cat "$out/BUILDINFO.txt"
'
