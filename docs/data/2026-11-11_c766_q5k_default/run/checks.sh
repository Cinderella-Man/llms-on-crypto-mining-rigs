#!/usr/bin/env bash
# C766 (Q1347): C690's Q5_K multi-token kernel (GGML_CUDA_MMVQ_PASCAL_Q5K_MT) at one row per lane group and without the
# register prefetch by default, checked on a build machine without a GPU: every build and test in a container (the CPU
# ones without network) at low CPU priority, one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   build  the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then test-llama-archs
#          from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   sass   CUDA 12.9's cuobjdump (-sass, -res-usage, -ptx) of every CUDA object of the branch's build tree and of the
#          base's (BASE_OBJ, the same build of the base commit), and -sass of both tarballs' bin/llama-server (BASE_BIN);
#          then run/compare_sass.py: every function's code the same in both builds, the 12 instantiations of
#          mul_mat_vec_q5_K_pascal_mt<NT, R, PF> and which one each setting of the switches launches in each build; and
#          C690's sass_check.sh (instruction counts, registers, the last two FFMA) on both builds' Q5_K object
#   cpu    a static native CPU build (GGML_CUDA=OFF) of the branch, every target, at $WORK/cpu/tip/tree/{src,build} with
#          its ccache in $WORK/cpu/ccache (relative paths); test-llama-archs kept in $WORK/cpu/tip/bin
#   off    test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, the branch's compares
#          them with every switch off
#   ctest  ctest in the branch's CPU build tree, without the tests that need a download, the network or torch (as C565)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE_CPU=<bin/ of the base's static native CPU build>
#   BASE_OBJ=<the base's rig build tree>/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
#   BASE_BIN=<the base's rig tarball tree>/bin [REMOTE=fork] [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...]
#   [IMAGE=pq-c565-cpu:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   $WORK/rig is scripts/build_llamacpp_docker.sh's work folder: its llama.cpp clone in $WORK/rig/src has the fork as
#   remote REMOTE with an https fetch URL, its ccache in $WORK/rig/ccache
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/sass/*.tsv, logs in $WORK/logs and $WORK/checks; done when
#   $WORK/C766_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
BASE_OBJ=$(realpath "${BASE_OBJ:?BASE_OBJ=<the base build tree's ggml-cuda.dir>}")
BASE_BIN=$(realpath "${BASE_BIN:?BASE_BIN=<the base tarball tree's bin/>}")
: "${BRANCH:?BRANCH=<fork branch>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass cpu off ctest}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
SRC=$WORK/rig/src
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C766_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$REPO:$REPO:ro" "$IMAGE" "$@"; }
tarball_tree() { local t; t=$(ls -t "$WORK"/rig/llama.cpp-*-"$BRANCH".tar.gz 2> /dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/rig/$(basename "$t" .tar.gz)"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c766 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi

if step build; then
    echo "=== build $(date -Is)"
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

if step sass; then
    echo "=== sass $(date -Is)"
    T=$(tarball_tree)
    S=$WORK/sass
    TIP_OBJ=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
    rm -rf "$S" && mkdir -p "$S"
    if [ -z "$T" ] || [ ! -d "$TIP_OBJ" ]; then
        verdict sass FAIL "no tarball tree or build tree of $BRANCH"
    else
        # shellcheck disable=SC2086
        locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_OBJ:$BASE_OBJ:ro" -v "$BASE_BIN:$BASE_BIN:ro" \
            -v "$REPO:$REPO:ro" "$CUDA_IMAGE" bash -c '
            set -u
            dump() { # <out> <objects dir> <bin dir>
                local out=$1 obj=$2 bin=$3 o n
                mkdir -p "$out/obj" "$out/sasscheck"
                (cd "$obj" && find . -name "*.cu.o" | sort) > "$out/objects.txt"
                while read -r o; do
                    n=$(basename "$o" .o)
                    nice -n 19 cuobjdump -sass "$obj/$o" > "$out/obj/$n.sass" 2>> "$out/errors.txt"
                    nice -n 19 cuobjdump -res-usage "$obj/$o" > "$out/obj/$n.res" 2>> "$out/errors.txt"
                    nice -n 19 cuobjdump -ptx "$obj/$o" > "$out/obj/$n.ptx" 2>> "$out/errors.txt"
                done < "$out/objects.txt"
                nice -n 19 cuobjdump -sass "$bin/llama-server" > "$out/llama-server.sass" 2>> "$out/errors.txt"
                md5sum "$obj/mmvq-pascal-q5k.cu.o" "$bin/llama-server" > "$out/md5.txt"
                # C690s sass_check.sh reads these three
                cp "$out/obj/mmvq-pascal-q5k.cu.res" "$out/sasscheck/q5k_res.txt"
                cp "$out/obj/mmvq-pascal-q5k.cu.sass" "$out/sasscheck/q5k.sass"
                gzip -c "$out/obj/mmvq-pascal.cu.sass" > "$out/sasscheck/mmvqp.sass.gz"
                SASS="$out/sasscheck" bash '"$REPO"'/docs/data/2026-11-11_c690_head_four_rows/run/sass_check.sh \
                    > "$out/sass_counts.tsv" 2>> "$out/errors.txt"
            }
            cuobjdump --version | tail -n 1 > '"$S"'/cuobjdump.txt
            dump '"$S/base"' '"$BASE_OBJ"' '"$BASE_BIN"'
            dump '"$S/tip"' '"$TIP_OBJ"' '"$T/bin"'
            ' > "$WORK/logs/sass.log" 2>&1
        rc=$?
        box bash -c "nice -n 19 python3 -I $REPO/docs/data/2026-11-11_c766_q5k_default/run/compare_sass.py $S/base $S/tip $S" \
            > "$K/compare_sass.txt" 2>&1
        rc2=$?
        cmp -s "$S/base/sass_counts.tsv" "$S/tip/sass_counts.tsv" && counts=same || counts=differ
        verdict sass "$([ $rc = 0 ] && [ $rc2 = 0 ] && [ "$counts" = same ] && echo SAME || echo DIFFERS)" \
            "rc=$rc/$rc2; $(tail -n 1 "$K/compare_sass.txt"); sass_check.sh counts $counts; $(cat "$S/cuobjdump.txt" 2> /dev/null); $(wc -l < "$S/base/errors.txt" 2> /dev/null)/$(wc -l < "$S/tip/errors.txt" 2> /dev/null) error lines"
    fi
fi

if step cpu; then
    echo "=== cpu $(date -Is)"
    R=$C/tip T=$C/tip/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
    rc=$?
    cp "$T/build/bin/test-llama-archs" "$R/bin/" 2> /dev/null
    verdict cpu-tip "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); all targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$K/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env> [args]: rc
        local tag=$1 bin=$2 envs=$3
        shift 3
        box bash -c 'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 '"$*"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$BASE_CPU" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; ${BASE_CPU}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    rc=$(archs tip "$C/tip/bin" '')
    n=$(grep -ac 'logits against' "$A/tip.log")
    grep -a 'logits against' "$A/tip.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/tip.differ.txt"
    bad=$(cut -d: -f1 "$A/tip.differ.txt" | tr '\n' ' ')
    verdict off-tip "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; no switch; $n compared, $(wc -l < "$A/tip.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/tip.log"); $(grep -a -m 1 '^Summary' "$A/tip.log")"
    rm -f "$A"/*.f32
fi

if step ctest; then
    echo "=== ctest $(date -Is)"
    B=$C/tip/tree/build
    box bash -c "cd $B && nice -n 19 ctest -j $JOBS --timeout 1800 --output-on-failure -E '$CTEST_EXCLUDE' --output-junit $K/ctest.xml" \
        > "$K/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$K/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$K/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}; not run: $CTEST_EXCLUDE"
fi

echo "=== done $(date -Is)"
touch "$WORK/C766_DONE"
