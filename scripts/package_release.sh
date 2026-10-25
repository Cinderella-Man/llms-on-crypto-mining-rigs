#!/usr/bin/env bash
# package_release.sh: the engine's release assets for the fork's GitHub Releases (docs/research/notes/release_v2.md,
# section 3). Builds the given fork commit with scripts/build_llamacpp_docker.sh (compute capability 6.1, no AVX,
# Ubuntu 24.04 userland), or takes an existing build folder, and packs:
#   <PREFIX>-<VERSION>-bin-ubuntu-cuda12.9-sm61.tar.gz     the llama.cpp binaries with RPATH $ORIGIN, LICENSE,
#                                                          BUILDINFO.txt and THIRD_PARTY_NOTICES.md
#   cudart-<PREFIX>-<VERSION>-bin-ubuntu-cuda12.9.tar.gz   NVIDIA's redistributable runtime libraries the binaries load
#                                                          (libcudart, libcublas, libcublasLt), with the EULA notice
#   <PREFIX>-<VERSION>-tests-ubuntu-cuda12.9-sm61.tar.gz   test-backend-ops, when the build has it (TESTS=1)
#   SHA256SUMS, RELEASE_NOTES.md (from release/notes-<VERSION>.md with the commit, names, sizes and hashes filled in)
# All three unpack into one folder, <PREFIX>-<VERSION>/, binaries and libraries side by side, so the rig needs only the
# NVIDIA driver: no CUDA toolkit. Nothing is uploaded or published.
#
# Usage, on the build machine (any x86-64 Linux PC with Docker):
#   scripts/package_release.sh <git-ref> <version> [expected-commit]
#       git-ref: as build_llamacpp_docker.sh takes it, e.g. fork/pascal-p106-v2-rc (a branch of the fork remote);
#       expected-commit: the build stops unless the ref resolves to it (a prefix is enough)
#   BUILD_DIR=<WORK>/llama.cpp-<date>-<sha>-<name> scripts/package_release.sh - <version>
#       packages a build that already exists
# Env [defaults]: WORK [~/llama-builds] (build_llamacpp_docker.sh's), OUT [WORK/release-<PREFIX>-<VERSION>],
#   PREFIX [llama-p106], IMAGE [nvidia/cuda:12.9.2-devel-ubuntu24.04], DOCKER_ARGS (e.g. a label the machine's policy
#   asks for), TESTS [1], JOBS, NOTES [release/notes-<VERSION>.md in this repository, if present], SRC [the llama.cpp
#   checkout the build came from: the build folder's sibling src/], NOTES_ONLY [0; 1 rewrites only RELEASE_NOTES.md of
#   the assets already in OUT, e.g. once the measured numbers are in the template], NOTES_VARIANT [none]: the engine
#   variant of the notes, as scripts/release_variant.sh takes it (rc4, rc3, rc1 or a list of tags); tagged lines and
#   blocks of other variants are removed, DRAFTS: drafter side files (export_gguf.py's GGUFs) as "name=file ...",
#   each copied unchanged into OUT under its name and added to SHA256SUMS and to the notes' asset table
#   ({{DRAFT_ROWS}}); DRAFT=<file> adds one as DRAFT_NAME [flint-staircase-drafter-<VERSION>.gguf]
# Exit status: 0 assets written and checked, 1 a step failed, 2 usage.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REF=${1:-}
VERSION=${2:-}
EXPECT=${3:-}
[ -n "$REF" ] && [ -n "$VERSION" ] || { sed -n '2,/^# Exit status/p' "$0"; exit 2; }
PREFIX=${PREFIX:-llama-p106}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
WORK=${WORK:-$HOME/llama-builds}
OUT=${OUT:-$WORK/release-$PREFIX-$VERSION}
NOTES=${NOTES:-$here/../release/notes-$VERSION.md}
BIN_ASSET=$PREFIX-$VERSION-bin-ubuntu-cuda12.9-sm61.tar.gz
CUDART_ASSET=cudart-$PREFIX-$VERSION-bin-ubuntu-cuda12.9.tar.gz
TESTS_ASSET=$PREFIX-$VERSION-tests-ubuntu-cuda12.9-sm61.tar.gz
DRAFT_NAME=${DRAFT_NAME:-flint-staircase-drafter-$VERSION.gguf}

if [ "${NOTES_ONLY:-0}" = 1 ]; then  # refill RELEASE_NOTES.md from the template, for assets already in $OUT
    [ -f "$OUT/SHA256SUMS" ] && [ -f "$OUT/$BIN_ASSET" ] || { echo "no assets in $OUT"; exit 1; }
    commit=$(tar xzf "$OUT/$BIN_ASSET" -O "$PREFIX-$VERSION/BUILDINFO.txt" | sed -n 's/^commit: \([0-9a-f]*\).*/\1/p')
else
if [ -z "${BUILD_DIR:-}" ]; then
    [ "$REF" != - ] || { echo "give a git ref, or BUILD_DIR"; exit 2; }
    log=$(mktemp)
    TESTS=${TESTS:-1} WORK=$WORK IMAGE=$IMAGE "$here/build_llamacpp_docker.sh" "$REF" | tee "$log"
    name=$(sed -n 's/^name: //p' "$log" | tail -1); rm -f "$log"
    BUILD_DIR=$WORK/$name
fi
[ -f "$BUILD_DIR/BUILDINFO.txt" ] && [ -x "$BUILD_DIR/bin/llama-server" ] || { echo "not a build folder: $BUILD_DIR"; exit 1; }
commit=$(sed -n 's/^commit: \([0-9a-f]*\).*/\1/p' "$BUILD_DIR/BUILDINFO.txt")
if [ -n "$EXPECT" ] && [ "${commit#"$EXPECT"}" = "$commit" ]; then
    echo "the build is at $commit, not at $EXPECT: stop"; exit 1
fi
[ -d "$OUT" ] && { echo "$OUT exists: move it away first"; exit 1; }
mkdir -p "$OUT"
echo "packaging $BUILD_DIR (commit $commit) as $PREFIX $VERSION into $OUT"

# the source tree's licence files, from the commit that was built (the build's own WORK/src, unless SRC is given)
src=${SRC:-$(dirname "$BUILD_DIR")/src}
lic=$(mktemp -d)
git -C "$src" show "$commit:LICENSE" > "$lic/LICENSE" 2>/dev/null || { echo "LICENSE of $commit not found in $src"; exit 1; }
if git -C "$src" cat-file -e "$commit:licenses" 2>/dev/null; then
    mkdir -p "$lic/licenses"
    for f in $(git -C "$src" ls-tree --name-only "$commit" licenses/); do git -C "$src" show "$commit:$f" > "$lic/$f"; done
fi

docker run --rm ${DOCKER_ARGS:-} -e P="$PREFIX-$VERSION" -e COMMIT="$commit" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e BIN_ASSET="$BIN_ASSET" -e CUDART_ASSET="$CUDART_ASSET" -e TESTS_ASSET="$TESTS_ASSET" \
    -v "$BUILD_DIR:/in:ro" -v "$lic:/lic:ro" -v "$OUT:/out" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq patchelf binutils > /dev/null
    s=/out/stage; b=$s/bin/$P; c=$s/cudart/$P; t=$s/tests/$P
    mkdir -p "$b" "$c" "$t"
    for f in /in/bin/*; do
        case $(basename "$f") in test-*) cp "$f" "$t/";; *) cp "$f" "$b/";; esac
    done
    for f in "$b"/* "$t"/*; do [ -f "$f" ] && patchelf --set-rpath "\$ORIGIN" "$f"; done
    cp /in/BUILDINFO.txt "$b/"; cp -r /lic/. "$b/"
    # the CUDA libraries llama-server loads, by soname, resolved in the build image
    ldd /in/bin/llama-server | awk "/libcu(dart|blas|blasLt)\\.so/ { print \$1, \$3 }" | while read -r so path; do
        cp -L "$path" "$c/$so"
    done
    ls "$c" | grep -q libcudart || { echo "no libcudart found through ldd"; exit 1; }
    cat > "$c/NVIDIA_CUDA_RUNTIME_NOTICE.md" <<EOF
# NVIDIA CUDA runtime libraries

$(ls "$c" | grep "^lib" | sed "s/^/- /")

These files are NVIDIA CUDA ${CUDA_VERSION:-12.9} runtime libraries, redistributed unmodified with the binaries of this
release as permitted by the NVIDIA CUDA Toolkit End User License Agreement, which lists libcudart, libcublas and
libcublasLt among the redistributable files (Attachment A): https://docs.nvidia.com/cuda/eula/index.html
They remain the property of NVIDIA under that agreement; the MIT licence of the binaries does not cover them.
EOF
    cat > "$b/THIRD_PARTY_NOTICES.md" <<EOF
# Third-party notices

- **llama.cpp and ggml** (https://github.com/ggml-org/llama.cpp): MIT licence, copyright the ggml authors; see
  LICENSE$( [ -d /lic/licenses ] && echo " and licenses/").
- **The Pascal changes of this fork** (commit $COMMIT): MIT licence, the same terms as llama.cpp.
- **NVIDIA CUDA runtime libraries**: shipped in the separate asset $CUDART_ASSET under the NVIDIA CUDA Toolkit
  End User License Agreement; see NVIDIA_CUDA_RUNTIME_NOTICE.md there.
EOF
    # checks: no AVX, glibc floor, every library found from the folder alone (system CUDA libraries removed)
    avx=$(objdump -d "$b/llama-server" | grep -c ymm || true)
    glibc=$(objdump -T "$b/llama-server" | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)
    rm -f /usr/local/cuda*/targets/*/lib/libcu* /usr/local/cuda*/lib64/libcu* 2>/dev/null; ldconfig
    mkdir -p /tmp/u && cp -a "$b"/. "$c"/. /tmp/u/
    missing=$(ldd /tmp/u/llama-server | grep "not found" | grep -v "libcuda.so.1" || true)
    {
        echo "avx instructions in llama-server: $avx"
        echo "highest glibc symbol: $glibc"
        echo "libraries not found with both assets unpacked (libcuda.so.1 comes from the driver): ${missing:-none}"
        ldd /tmp/u/llama-server | sed "s/^/  /"
    } > /out/CHECKS.txt
    cat /out/CHECKS.txt
    [ "$avx" = 0 ] && [ -z "$missing" ] || { echo "checks failed"; exit 1; }
    # deterministic archives: the same files give the same bytes (sorted names, fixed times and owners, no gzip time)
    pack() { tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - -C "$1" "$P" | gzip -n -9 > "$2"; }
    pack "$s/bin" "/out/$BIN_ASSET"
    pack "$s/cudart" "/out/$CUDART_ASSET"
    [ -n "$(ls "$t")" ] && pack "$s/tests" "/out/$TESTS_ASSET"
    rm -rf "$s"
    (cd /out && sha256sum *.tar.gz > SHA256SUMS)
    chown -R "$HOST_UID:$HOST_GID" /out
'
rm -rf "$lic"
fi

# the drafter side files, copied unchanged next to the archives (DRAFT=<file> is DRAFTS="$DRAFT_NAME=<file>")
[ -n "${DRAFT:-}" ] && DRAFTS="${DRAFTS:+$DRAFTS }$DRAFT_NAME=$DRAFT"
for d in ${DRAFTS:-}; do
    name=${d%%=*} file=${d#*=}
    [ "$(head -c 4 "$file")" = GGUF ] || { echo "not a GGUF file: $file"; exit 1; }
    cp "$file" "$OUT/$name"
    (cd "$OUT" && grep -v " $name\$" SHA256SUMS > SHA256SUMS.new; sha256sum "$name" >> SHA256SUMS.new; mv SHA256SUMS.new SHA256SUMS)
    echo "drafter: $file -> $OUT/$name"
done

size() { du -m "$OUT/$1" | cut -f1; }
sha() { grep " $1\$" "$OUT/SHA256SUMS" | cut -d' ' -f1; }
if [ -f "$NOTES" ]; then
    # one asset row per drafter file in OUT, in place of the template line {{DRAFT_ROWS}}
    rows=$(mktemp)
    for f in "$OUT"/*.gguf; do
        [ -f "$f" ] || continue
        n=$(basename "$f"); echo "| \`$n\` | $(size "$n") MB | \`$(sha "$n")\` |" >> "$rows"
    done
    sed -e "s|{{VERSION}}|$VERSION|g" -e "s|{{COMMIT}}|$commit|g" -e "s|{{PREFIX}}|$PREFIX|g" \
        -e "s|{{BIN_ASSET}}|$BIN_ASSET|g" -e "s|{{BIN_MB}}|$(size "$BIN_ASSET")|g" -e "s|{{BIN_SHA256}}|$(sha "$BIN_ASSET")|g" \
        -e "s|{{CUDART_ASSET}}|$CUDART_ASSET|g" -e "s|{{CUDART_MB}}|$(size "$CUDART_ASSET")|g" \
        -e "s|{{CUDART_SHA256}}|$(sha "$CUDART_ASSET")|g" -e "/^{{DRAFT_ROWS}}\$/{r $rows" -e "d}" "$NOTES" \
        | bash "$here/release_variant.sh" "${NOTES_VARIANT:-none}" - > "$OUT/RELEASE_NOTES.md"
    rm -f "$rows"
    left=$(grep -o '{{[A-Z_0-9]*}}' "$OUT/RELEASE_NOTES.md" | sort -u | tr '\n' ' ' || true)
    [ -n "$left" ] && echo "WARNING: unfilled fields in the notes: $left"
    echo "release notes: $OUT/RELEASE_NOTES.md (placeholders left: $(grep -c 'RIG-TBD' "$OUT/RELEASE_NOTES.md" || true))"
else
    echo "no notes template at $NOTES: RELEASE_NOTES.md not written"
fi
echo "assets in $OUT:"; ls -l "$OUT"
