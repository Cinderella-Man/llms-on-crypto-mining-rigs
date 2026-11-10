#!/usr/bin/env bash
# C675 (Q1187): the CPU check of LLAMA_SPEC_REJECT_STOP on a build machine, CPU only, every test in a container without
# network. Two trees, built one after the other at the same paths ($WORK/cpu/tree/{src,build}), so ccache hits:
#   (ccache: $WORK/ccache, the one scripts/build_llamacpp_docker.sh fills)
#   tree    git archive of TIP (the branch with the switch) or BASE (the branch it was cut from) into the tree
#   build   a static CPU build (GGML_CUDA=OFF, GGML_NATIVE=ON, LLAMA_BUILD_TESTS=ON) of the targets the checks run, under
#           LOCK; the binaries are kept in $WORK/cpu/<tag>/bin
#   ctest   TIP only: test-spec-reject-stop (the switch at its default, off), test-spec-reject-stop-env (on from the
#           environment), test-speculative-rejection-sampling (the target distribution preserved) and test-sampling,
#           with the model the fixture test-generate-speculative-target writes
#   ident   test-speculative-rejection-sampling of BASE and of TIP, TIP once more with LLAMA_SPEC_REJECT_STOP=1, on the
#           ctest step's model, N_TRIALS trials each: every printed line of the distribution the same in the three
#           runs (its drafts hold one token, which the switch never cuts)
# Usage: WORK=<dir> SRC=<llama.cpp clone holding both commits> TIP=<commit> BASE=<commit> [JOBS=8] [THREADS=4]
#   [N_TRIALS=20000] [LOCK=<flock file>] [DOCKER_ARGS=...] [BASE_IMAGE=ubuntu:24.04] [IMAGE=pq-c565-cpu:24.04] bash cpu_check.sh
# Output: $WORK/cpu/<tag>/ (tree.txt, cmake.log, build.log, bin/), $WORK/cpu/ctest.log, $WORK/cpu/ident/*.log,
#   $WORK/cpu/summary.tsv (tree, check, verdict, detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${TIP:?TIP=<commit>}" "${BASE:?BASE=<commit>}"
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
N_TRIALS=${N_TRIALS:-20000}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
C=$WORK/cpu
T=$C/tree
B=$T/build
D="--rm --label pascal-quants=1 --cpu-shares=256 --network none ${DOCKER_ARGS:-}"
mkdir -p "$C" "$T" "$WORK/ccache" "$C/ident"
rm -f "$C/DONE"
printf 'tree\tcheck\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$C/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
# shellcheck disable=SC2086
box()     { docker run $D -v "$WORK:$WORK" "$IMAGE" "$@"; }

# the image: the compiler and the tests' tools, nothing else
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        "$BASE_IMAGE" | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$C/image.log" 2>&1 || { echo "image build failed: $C/image.log"; exit 1; }
fi
echo "=== c675 cpu check started $(date -Is), TIP ${TIP:0:9}, BASE ${BASE:0:9}"

# <tag> <commit> <targets>: the tree, its build, its binaries kept
build() {
    local tag=$1 commit=$2 targets=$3 R=$C/$1 rc=0
    mkdir -p "$R"
    rm -rf "$T/src" "$B" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$commit" | tar -x -C "$T/src" || rc=$?
    echo "commit $(git -C "$SRC" rev-parse "$commit") $(git -C "$SRC" log -1 --format=%s "$commit")" > "$R/tree.txt"
    verdict "$tag" tree "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(cat "$R/tree.txt")"
    [ $rc = 0 ] || return 1
    # shellcheck disable=SC2086
    locked docker run $D -v "$WORK:$WORK" -e CCACHE_DIR="$WORK/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$B"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
            -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$B"' -j '"$JOBS"' --target '"$targets"' > '"$R/build.log"' 2>&1'
    rc=$?
    verdict "$tag" build "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; targets $targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
    for t in $targets; do [ -x "$B/bin/$t" ] && cp "$B/bin/$t" "$R/bin/"; done
    return $rc
}

TESTS_RE='^(test-generate-speculative-target|test-spec-reject-stop|test-spec-reject-stop-env|test-speculative-rejection-sampling|test-sampling)$'
if build tip "$TIP" "test-llama-archs test-spec-reject-stop test-speculative-rejection-sampling test-sampling"; then
    box bash -c "cd $B && nice -n 19 ctest --timeout 1800 --output-on-failure -V -R '$TESTS_RE' --output-junit $C/ctest.xml" > "$C/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$C/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$C/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict tip ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}"
    # each run of the new test: its random trials' counts
    grep -a 'random: ' "$C/ctest.log" | sed -E 's/^[0-9]+: //' > "$C/reject_stop_counts.txt"
    cp "$B/tests/test-speculative-models/target/llama-dense.gguf" "$C/ident/llama-dense.gguf"
fi

build base "$BASE" "test-speculative-rejection-sampling"

if [ -x "$C/tip/bin/test-speculative-rejection-sampling" ] && [ -x "$C/base/bin/test-speculative-rejection-sampling" ] && [ -f "$C/ident/llama-dense.gguf" ]; then
    I=$C/ident
    run() { box env $2 nice -n 19 "$C/$3/bin/test-speculative-rejection-sampling" -m "$I/llama-dense.gguf" -t "$THREADS" \
            --n-trials "$N_TRIALS" > "$I/$1.log" 2>&1; echo "$1 rc=$?" >> "$I/rc.txt"; grep -aE '^(token |Empirical|All tests|Some tests)' "$I/$1.log" > "$I/$1.dist"; }
    rm -f "$I/rc.txt"
    run base     ''                       base
    run tip      ''                       tip
    run tip-on   LLAMA_SPEC_REJECT_STOP=1 tip
    n=$(grep -ac '^token ' "$I/base.dist")
    same=1
    for r in tip tip-on; do cmp -s "$I/base.dist" "$I/$r.dist" || same=0; done
    verdict ident distribution "$([ $same = 1 ] && [ "$n" -gt 0 ] && ! grep -qv ' rc=0$' "$I/rc.txt" && echo SAME || echo DIFFERS)" \
        "$n token lines, $N_TRIALS trials; $(tr '\n' ' ' < "$I/rc.txt")"
fi

date -Is > "$C/DONE"
echo "=== done $(date -Is)"
cat "$C/summary.tsv"
