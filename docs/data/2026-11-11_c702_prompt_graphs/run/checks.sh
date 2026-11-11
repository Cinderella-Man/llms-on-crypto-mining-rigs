#!/usr/bin/env bash
# C702 (Q1240): CUDA graphs for prompt micro-batches (GGML_CUDA_GRAPHS_FORCE) on a fork branch, checked on a build
# machine, CPU only. Every step that builds or tests takes LOCK, so that one runs at a time on a shared machine.
# Steps:
#   build   the rig tarball of the branch (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code, no AVX)
#   tests   from the same CUDA build tree: test-cuda-graph-force (the switch's choice of graphs and its capture
#           decisions, in a model of the whole warm-up and capture logic) and test-cuda-profile-light (the stock
#           logic's model, a neighbour), run in a container without a GPU against CUDA's stub driver
# Usage: WORK=<dir> BUILD=<scripts/build_llamacpp_docker.sh> BRANCH=<fork branch> [REMOTE=fork] [JOBS=8]
#   [LOCK=<flock file>] [DOCKER_ARGS=...] [STEPS='build tests'] bash checks.sh
#   WORK   the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as remote
#          REMOTE with an https fetch URL; $WORK/ccache
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/md5.txt, logs in $WORK/logs, test output in $WORK/tests;
#   done when $WORK/C702_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BUILD=$(realpath "${BUILD:?BUILD=<scripts/build_llamacpp_docker.sh>}")
: "${BRANCH:?BRANCH=<fork branch>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
STEPS=${STEPS:-build tests}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
TESTS_RUN="test-cuda-graph-force test-cuda-profile-light"
mkdir -p "$WORK/logs" "$WORK/tests"
rm -f "$WORK/C702_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
# a container without a GPU: the binaries link the CUDA driver library, for which the toolkit's stub stands in
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
echo "=== c702 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$BUILD" "$REMOTE/$BRANCH" > "$WORK/logs/rig.log" 2>&1
    rc=$?
    T=$(tarball_tree)
    if [ $rc = 0 ] && [ -n "$T" ]; then
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        verdict build-rig OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/rig.log") error lines in logs/rig.log"
    fi
fi

if step tests; then
    echo "=== tests $(date -Is)"
    # the tests from the rig build's tree (its cache: CUDA on, sm_61, no AVX), at the same commit
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e TESTS_RUN="$TESTS_RUN" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target $TESTS_RUN > /work/logs/tests-build.log 2>&1 \
            || { tail -40 /work/logs/tests-build.log; exit 1; }
        rm -rf /work/tests/bin && mkdir -p /work/tests/bin
        for t in $TESTS_RUN; do cp build/bin/$t /work/tests/bin/; done
        chown -R "$HOST_UID:$HOST_GID" /work/tests /work/logs /work/ccache' > "$WORK/logs/tests.log" 2>&1
    rc=$?
    verdict tests-build "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $TESTS_RUN at ${TIP:0:9}"
    for t in $TESTS_RUN; do
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$WORK:$WORK" "$IMAGE" bash -c "$STUB"'
            nice -n 19 '"$WORK/tests/bin/$t"' > '"$WORK/tests/$t.log"' 2>&1; echo "rc=$?"; chown '"$(id -u):$(id -g)"' '"$WORK/tests/$t.log" \
            > "$WORK/tests/$t.rc" 2>&1
        rc=$(sed -n 's/^rc=//p' "$WORK/tests/$t.rc")
        verdict "$t" "$([ "$rc" = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(grep -h -E '^(random runs|test-cuda)' "$WORK/tests/$t.log" | tr '\n' ' ')"
    done
fi

echo "=== done $(date -Is)"
touch "$WORK/C702_DONE"
