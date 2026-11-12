#!/usr/bin/env bash
# Build edge0-strata (CUDA 13, the engine for Edge0 145B and 120B) and edge0-infer (its launcher, CPU preset) from one
# pinned commit of https://github.com/Edge0-AI/Edge0, inside a container. The build needs no GPU: the card's
# architecture is given, not detected. The CUDA 13 toolkit is either the image's own or a host directory mounted
# read-only (CUDA_DIR), so nothing is installed on the host.
#
#   WORK=/path/to/edge0 CUDA_DIR=/usr/local/cuda-13.4 bash build_edge0.sh
#
# Optional: COMMIT (the Edge0 commit that was read before building), ARCH (86 = RTX 30; 75, 89, 120 for others),
#           JOBS (default 8), LOCK (a file: the compile waits for it with flock),
#           BASE (image the build image is made from, default ubuntu:24.04), IMAGE (name of the build image).
# Output:   $WORK/src/Edge0/edge0-strata/build/strata, $WORK/src/Edge0/edge0-strata/.venv,
#           $WORK/src/Edge0/edge0-infer/build/cpu/bin/edge0-infer, $WORK/build/build.log, $WORK/build/BUILDINFO.txt,
#           $WORK/build/done ("OK <date>" or "FAIL <date> <step>")
set -u
WORK=${WORK:?set WORK}
COMMIT=${COMMIT:-b7900cbb322926ccebdcd0c0c5a1280497ecff22}
LLAMA=3cf03257f219afbe7334045ff7c6a06ac68c627d     # the ggml source edge0-strata pins (llama.cpp tag b11062)
ARCH=${ARCH:-86}; JOBS=${JOBS:-8}; LOCK=${LOCK:-}; CUDA_DIR=${CUDA_DIR:-}
BASE=${BASE:-ubuntu:24.04}; IMAGE=${IMAGE:-edge0-build:24.04}
SRC=$WORK/src/Edge0; OUT=$WORK/build; LOG=$OUT/build.log
mkdir -p "$WORK/src" "$OUT"; rm -f "$OUT/done"
exec >> "$LOG" 2>&1
fail() { echo "== $(date '+%F %T') FAIL: $*"; echo "FAIL $(date '+%F %T') $*" > "$OUT/done"; exit 1; }
echo "== $(date '+%F %T') start: Edge0 $COMMIT, sm_$ARCH, base $BASE, CUDA ${CUDA_DIR:-from the image}"

# 1. the source, at the commit that was read
[ -d "$SRC/.git" ] || git clone -q https://github.com/Edge0-AI/Edge0.git "$SRC" || fail clone
git -C "$SRC" fetch -q origin || fail fetch
git -C "$SRC" -c advice.detachedHead=false checkout -q "$COMMIT" || fail checkout
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || fail "HEAD is not $COMMIT"
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || fail "source tree is modified"

# 2. ggml, from llama.cpp at the pinned commit (the archive of a commit hash is that commit's tree)
G=$SRC/edge0-strata/third_party/llama.cpp
if [ ! -f "$G/ggml/CMakeLists.txt" ]; then
  mkdir -p "$G"
  curl -fsSL --retry 5 "https://github.com/ggml-org/llama.cpp/archive/$LLAMA.tar.gz" | tar -xz --strip-components=1 -C "$G" || fail "ggml source"
fi

# 3. the build image: compilers, CMake, Ninja and Python's venv on top of BASE
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  docker build --label pascal-quants=1 -t "$IMAGE" - <<EOF || fail "image"
FROM $BASE
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ca-certificates curl cmake ninja-build python3 python3-venv python3-pip build-essential \
 && rm -rf /var/lib/apt/lists/*
EOF
fi
echo "== $(date '+%F %T') image ready: $(docker image inspect -f '{{.Id}}' "$IMAGE")"

# 4. compile, one build at a time when LOCK is given, at a low CPU share
STEPS='
set -eu
[ -z "${CUDA_DIR:-}" ] || export PATH="$CUDA_DIR/bin:$PATH" LD_LIBRARY_PATH="$CUDA_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version | tail -n 2
git config --global --add safe.directory "*"
cd edge0-strata
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build --target strata -j "$JOBS"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install --no-cache-dir --disable-pip-version-check -q -r requirements.txt
.venv/bin/pip freeze > build/pip-freeze.txt
cd ../edge0-infer
cmake --preset cpu
cmake --build --preset cpu -j "$JOBS"
build/cpu/bin/edge0-infer info || true
'
compile() {
  docker run --rm --name edge0-build --label pascal-quants=1 --label edge0-build=1 --cpu-shares=256 --entrypoint bash \
    -v "$WORK:$WORK" ${CUDA_DIR:+-v "$CUDA_DIR:$CUDA_DIR:ro"} -e CUDA_DIR="$CUDA_DIR" -e ARCH="$ARCH" -e JOBS="$JOBS" \
    -w "$SRC" "$IMAGE" -c "$STEPS"
}
if [ -n "$LOCK" ]; then
  echo "== $(date '+%F %T') waiting for $LOCK"
  ( flock 9; echo "== $(date '+%F %T') lock taken"; compile ) 9> "$LOCK" || fail compile
else
  compile || fail compile
fi

# 5. what was built
[ -x "$SRC/edge0-strata/build/strata" ] && [ -x "$SRC/edge0-infer/build/cpu/bin/edge0-infer" ] || fail "binaries missing"
{
  echo "Edge0 commit:   $COMMIT"
  echo "ggml source:    llama.cpp $LLAMA"
  echo "CUDA arch:      sm_$ARCH"
  echo "CUDA toolkit:   ${CUDA_DIR:-from the image}"
  echo "image:          $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE")) from $BASE"
  echo "built:          $(date '+%F %T')"
  sha256sum "$SRC/edge0-strata/build/strata" "$SRC/edge0-infer/build/cpu/bin/edge0-infer"
} > "$OUT/BUILDINFO.txt" || fail buildinfo
echo "== $(date '+%F %T') done"; echo "OK $(date '+%F %T')" > "$OUT/done"
