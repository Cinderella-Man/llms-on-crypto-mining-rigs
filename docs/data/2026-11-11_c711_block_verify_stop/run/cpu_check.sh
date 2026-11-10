#!/usr/bin/env bash
# C711 (Q1263): the CPU check of LLAMA_SPEC_REJECT_STOP in block verification on a build machine, CPU only, every test
# in a container without network. Two trees, built one after the other at the same paths ($WORK/cpu/tree/{src,build}),
# so ccache hits (ccache: $WORK/ccache, the one scripts/build_llamacpp_docker.sh fills):
#   tree    git archive of TIP (the branch with the cut) or BASE (C675's branch, which it was cut from) into the tree;
#           BASE gets TIP's tests/test-spec-reject-stop.cpp laid over it, so both build the same test
#   build   a static CPU build (GGML_CUDA=OFF, GGML_NATIVE=ON, LLAMA_BUILD_TESTS=ON) of the targets the checks run; the
#           binaries are kept in $WORK/cpu/<tag>/bin
#   ctest   TIP only: test-spec-reject-stop (the switch at its default, off), test-spec-reject-stop-env (on from the
#           environment), test-spec-block (block verification's arithmetic, exact by enumeration, the cut included),
#           test-speculative-rejection-sampling and test-sampling, with the model the fixture
#           test-generate-speculative-target writes
#   ident   with the switch off the block path returns what BASE's returns: test-spec-reject-stop of BASE (TIP's test on
#           BASE's library; its checks of the switch on fail there, as BASE has no cut) and of TIP, N_TRIALS random drafts
#           each, the digest of the switch-off block results the same; and test-spec-block of BASE and TIP print the same
#           line for every configuration BASE has
# Usage: WORK=<dir> SRC=<llama.cpp clone holding both commits> TIP=<commit> BASE=<commit> [JOBS=8] [THREADS=4]
#   [N_TRIALS=4000] [LOCK=<flock file>] [DOCKER_ARGS=...] [BASE_IMAGE=ubuntu:24.04] [IMAGE=pq-c565-cpu:24.04] bash cpu_check.sh
# Output: $WORK/cpu/<tag>/ (tree.txt, cmake.log, build.log, bin/), $WORK/cpu/ctest.log, $WORK/cpu/counts.txt,
#   $WORK/cpu/ident/*.log, $WORK/cpu/summary.tsv (tree, check, verdict, detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${TIP:?TIP=<commit>}" "${BASE:?BASE=<commit>}"
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
N_TRIALS=${N_TRIALS:-4000}
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
echo "=== c711 cpu check started $(date -Is), TIP ${TIP:0:9}, BASE ${BASE:0:9}"

# <tag> <commit> <targets> [<file of TIP to lay over the tree>...]: the tree, its build, its binaries kept
build() {
    local tag=$1 commit=$2 targets=$3 R=$C/$1 rc=0 f
    shift 3
    mkdir -p "$R"
    rm -rf "$T/src" "$B" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$commit" | tar -x -C "$T/src" || rc=$?
    echo "commit $(git -C "$SRC" rev-parse "$commit") $(git -C "$SRC" log -1 --format=%s "$commit")" > "$R/tree.txt"
    for f in "$@"; do
        git -C "$SRC" show "$TIP:$f" > "$T/src/$f" || rc=$?
        echo "laid over: $f from $(git -C "$SRC" rev-parse --short=9 "$TIP")" >> "$R/tree.txt"
    done
    verdict "$tag" tree "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(tr '\n' ';' < "$R/tree.txt")"
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

I=$C/ident
TESTS_RE='^(test-generate-speculative-target|test-spec-reject-stop|test-spec-reject-stop-env|test-spec-block|test-speculative-rejection-sampling|test-sampling)$'
if build tip "$TIP" "test-llama-archs test-spec-reject-stop test-spec-block test-speculative-rejection-sampling test-sampling"; then
    box bash -c "cd $B && nice -n 19 ctest --timeout 1800 --output-on-failure -V -R '$TESTS_RE' --output-junit $C/ctest.xml" > "$C/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$C/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$C/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict tip ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}"
    # each run's counts: the random drafts of both paths, the grammar case, block verification's enumeration
    grep -aE 'random: |block: |block off digest|grammar (rejection|block)|instances: max' "$C/ctest.log" | sed -E 's/^[0-9]+: //' > "$C/counts.txt"
    cp "$B/tests/test-speculative-models/target/llama-dense.gguf" "$I/llama-dense.gguf"
fi

build base "$BASE" "test-spec-reject-stop test-spec-block" tests/test-spec-reject-stop.cpp

if [ -f "$I/llama-dense.gguf" ]; then
    rm -f "$I/rc.txt"
    for r in base tip; do
        if [ -x "$C/$r/bin/test-spec-reject-stop" ]; then
            box nice -n 19 "$C/$r/bin/test-spec-reject-stop" -m "$I/llama-dense.gguf" -t "$THREADS" --n-trials "$N_TRIALS" \
                > "$I/reject-stop-$r.log" 2>&1
            echo "reject-stop-$r rc=$?" >> "$I/rc.txt"
        fi
        if [ -x "$C/$r/bin/test-spec-block" ]; then
            box nice -n 19 "$C/$r/bin/test-spec-block" > "$I/spec-block-$r.log" 2>&1
            echo "spec-block-$r rc=$?" >> "$I/rc.txt"
        fi
    done
    db=$(grep -a 'block off digest' "$I/reject-stop-base.log" 2>/dev/null | sed -E 's/.*digest //')
    dt=$(grep -a 'block off digest' "$I/reject-stop-tip.log" 2>/dev/null | sed -E 's/.*digest //')
    verdict ident block-off "$([ -n "$db" ] && [ "$db" = "$dt" ] && echo SAME || echo DIFFERS)" \
        "digest base ${db:-none}, tip ${dt:-none}; $(tr '\n' ' ' < "$I/rc.txt")"
    # every enumeration line of BASE's configurations, as TIP prints it
    grep -a 'instances: max' "$I/spec-block-base.log" > "$I/spec-block-base.lines" 2>/dev/null
    n=$(wc -l < "$I/spec-block-base.lines")
    miss=$(grep -cvxFf <(grep -a 'instances: max' "$I/spec-block-tip.log" 2>/dev/null) "$I/spec-block-base.lines")
    verdict ident spec-block "$([ "$n" -gt 0 ] && [ "$miss" = 0 ] && echo SAME || echo DIFFERS)" \
        "$n configuration lines of base, $miss not printed alike by tip; tip prints $(grep -ac 'instances: max' "$I/spec-block-tip.log") lines"
fi

date -Is > "$C/DONE"
echo "=== done $(date -Is)"
cat "$C/summary.tsv"
