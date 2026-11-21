#!/usr/bin/env bash
# C880 (2026-11-21): the fork's own tools and tests moved off upstream's deprecated batch API (fork branch
# pascal-p106-v5-batch-api), built and checked on a build machine's CPU against their base, pascal-p106-v5. No GPU work.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; both compiler caches hard-linked from CPU_CCACHE and RIG_CCACHE
#   quick   fail-fast: cpu_tests.sh builds only the head's eight changed targets (at the later steps' paths and
#           compiler cache, so the head step reuses the objects); a build error or a deprecation warning left in the
#           eight files stops here
#   base    cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the base, its build kept: a CPU build of every
#           target, ctest without network, test-llama-archs (plain, write, compare), the hand-run tests
#   models  with the base's test-llama-archs, seed 1: a random qwen4exp MoE model (4,096-token vocabulary) and the
#           same with 32 experts, 8 used (test-moe-ids' shape); add_screen_head.py adds a screened head to the first
#           (qwen4exp-moe-screen.gguf); prompts.txt, three prompts for llama-screen-check
#   tbase   tools.sh with the base's binaries (before the head's build replaces them) into $W/tools/base
#   head    cpu_tests.sh on the head, ident against the base's logits files, its build kept
#   thead   tools.sh with the head's binaries into $W/tools/head
#   rig     the rig build (scripts/build_llamacpp_docker.sh REF, TESTS=1, test-llama-archs and test-cuda-keepalive
#           added, as BASE_RIG_LOG's build), under LOCK, for its warnings only: the tarball is deleted
#   compare the two CPU builds' warnings, their ctest results test by test, the hand-run tests' verdicts (their
#           random models are new on every run: test-llama-archs seeds from std::random_device without -s), the rig
#           build's warnings against BASE_RIG_LOG, and compare.py on the two tools.sh outputs (tools.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> REPO=<folder with build_llamacpp_docker.sh
#   cpu_tests.sh tools.sh add_screen_head.py compare.py and pipespec/ (bench/pipespec)> [BASE_RIG_LOG=<the base's rig
#   build log>] [CPU_CCACHE=<dir>] [RIG_CCACHE=<dir>] [LOCK=<flock file>] [STEPS=...] [JOBS=8] [THREADS=6] bash run.sh
# Every container: labels pascal-quants=1 and c880=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
REPO=${REPO:?REPO=<folder with the scripts>}
BASE_RIG_LOG=${BASE_RIG_LOG:-}
REF=${REF:-fork/pascal-p106-v5-batch-api}
HEAD_SHA=${HEAD_SHA:-aebff903eae9e205325795ff1486a077edccb543}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src quick base models tbase head thead rig compare}
CHANGED="llama-pipespec llama-screen-check llama-pipe-recurrent-check test-moe-ids test-kpool-dev test-hop-cast test-pool-mask-dev test-kpool-stage"
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c880=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/cpu" "$W/rig" "$W/logs" "$W/checks" "$W/tools"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c880=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
export LC_ALL=C # sort and comm in one collation
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
deprecated() { grep -c "is deprecated: use llama_" "$1"; }
cbox() { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:$REPO:ro" "$CPU_IMAGE" "$@"; }
B=$W/cpu/tree/build/bin
MD=$W/models
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

cpu_tests() { # <trees> <steps> <log>
    WORK="$W/cpu" SRC="$W/rig/src" TREES="$1" STEPS="$2" JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" \
        DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$3" 2>&1
}

if step quick; then
    cpu_tests "quick=$HEAD_SHA:${CHANGED// /,}" "tree build" "$W/logs/quick.log"
    echo "quick build rc=$?"
    grep -P '^quick\t' "$W/cpu/summary.tsv"
    missing=""; for t in $CHANGED; do [ -x "$B/$t" ] || missing="$missing $t"; done
    nd=$(grep -ac "is deprecated: use llama_" "$W/cpu/quick/build.log")
    echo "quick: missing binaries:${missing:- none}; deprecated batch API warnings: $nd"
    grep -aE "error:|is deprecated: use llama_" "$W/cpu/quick/build.log" | head -n 20
    [ -z "$missing" ] && [ "$nd" = 0 ] || { echo "quick check failed"; finish 1; }
    echo "=== quick done $(date -Is)"
fi

if step base; then
    cpu_tests "base=$BASE_SHA" "tree build ctest archs hand" "$W/logs/cpu-base.log"
    echo "base cpu tests rc=$?"
    grep -P '^base\t' "$W/cpu/summary.tsv"
    [ -x "$B/llama-pipespec" ] || { echo "no base build"; finish 1; }
    echo "=== base done $(date -Is)"
fi

if step models; then
    rm -rf "$MD" && mkdir -p "$MD/e32"
    cbox bash -c "export LLAMA_TEST_ARCHS_N_VOCAB=4096
        nice -n 19 $B/test-llama-archs -a '^qwen4exp\$' -s 1 -o $MD > $MD/gen.log 2>&1; echo \"qwen4exp rc=\$?\"
        LLAMA_TEST_ARCHS_N_EXPERT=32,8 nice -n 19 $B/test-llama-archs -a '^qwen4exp\$' -s 1 -o $MD/e32 > $MD/e32/gen.log 2>&1
        echo \"qwen4exp e32 rc=\$?\""
    cbox python3 -I "$REPO/add_screen_head.py" --top-m 64 "$MD/qwen4exp-moe.gguf" "$MD/qwen4exp-moe-screen.gguf" \
        > "$MD/screen.log" 2>&1
    echo "screen head rc=$?"; cat "$MD/screen.log"
    printf '%s\n' 'The quick brown fox jumps over the lazy dog.' \
        'def add(a, b):\n    return a + b\n\nprint(add(2, 3))' \
        'Write a short poem about rain on a tin roof.' > "$MD/prompts.txt"
    (cd "$MD" && md5sum qwen4exp-moe.gguf qwen4exp-moe-screen.gguf e32/qwen4exp-moe.gguf) | tee "$MD/md5.txt"
    [ -f "$MD/qwen4exp-moe.gguf" ] && [ -f "$MD/qwen4exp-moe-screen.gguf" ] && [ -f "$MD/e32/qwen4exp-moe.gguf" ] \
        || { echo "test models missing"; finish 1; }
    echo "=== models done $(date -Is)"
fi

tools() { # <tag>
    rm -rf "$W/tools/$1" && mkdir -p "$W/tools/$1"
    cbox env BIN="$B" MD="$MD" PS="$REPO/pipespec" OUT="$W/tools/$1" THREADS="$THREADS" \
        nice -n 19 bash "$REPO/tools.sh" > "$W/logs/tools-$1.log" 2>&1
    echo "tools $1 rc=$?"
    cat "$W/tools/$1/rc.tsv" | tr '\n' ' '; echo
}

if step tbase; then
    grep -q "commit $BASE_SHA" "$W/cpu/base/tree.txt" || { echo "the build tree is not the base's"; finish 1; }
    tools base
    echo "=== tbase done $(date -Is)"
fi

if step head; then
    cpu_tests "c880=$HEAD_SHA@base" "tree build ctest archs ident hand" "$W/logs/cpu-head.log"
    echo "head cpu tests rc=$?"
    grep -P '^c880\t' "$W/cpu/summary.tsv"
    echo "=== head done $(date -Is)"
fi

if step thead; then
    grep -q "commit $HEAD_SHA" "$W/cpu/c880/tree.txt" || { echo "the build tree is not the head's"; finish 1; }
    tools head
    echo "=== thead done $(date -Is)"
fi

if step rig; then
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rig build rc=$?"
    # the warnings are all this build is for: no tarball from this row
    rm -rf "$W"/rig/llama.cpp-*
    echo "=== rig done $(date -Is)"
fi

if step compare; then
    K=$W/checks
    warnings "$W/cpu/base/build.log" > "$K/cpu-warnings-base.txt"
    warnings "$W/cpu/c880/build.log" > "$K/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$K/cpu-warnings-base.txt") lines ($(deprecated "$K/cpu-warnings-base.txt") deprecated batch API),"\
        "head $(wc -l < "$K/cpu-warnings-head.txt") ($(deprecated "$K/cpu-warnings-head.txt")), only in head: $(comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt"
    if [ -n "$BASE_RIG_LOG" ] && [ -f "$W/logs/rig.log" ]; then
        warnings "$BASE_RIG_LOG" > "$K/rig-warnings-base.txt"
        warnings "$W/logs/rig.log" > "$K/rig-warnings-head.txt"
        echo "rig warnings: base $(wc -l < "$K/rig-warnings-base.txt") lines ($(deprecated "$K/rig-warnings-base.txt") deprecated batch API),"\
            "head $(wc -l < "$K/rig-warnings-head.txt") ($(deprecated "$K/rig-warnings-head.txt")), only in head: $(comm -13 "$K/rig-warnings-base.txt" "$K/rig-warnings-head.txt" | wc -l)"
        comm -13 "$K/rig-warnings-base.txt" "$K/rig-warnings-head.txt"
    fi
    for t in base c880; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$K/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$K/ctest-$t.tsv"
    done
    echo "ctest, test by test: $(diff -q "$K/ctest-base.tsv" "$K/ctest-c880.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$K/ctest-base.tsv" "$K/ctest-c880.tsv"
    cat "$K/ctest-failed-c880.txt"
    # the hand-run tests' random models differ per run, so their verdicts are compared, not their logs
    grep -P '^(base|c880)\thand ' "$W/cpu/summary.tsv" | cut -f1-3 | sort -t$'\t' -k2,2 -k1,1 > "$K/hand-verdicts.tsv"
    echo "hand-run tests, verdicts: $(diff -q <(grep -P '^base\t' "$K/hand-verdicts.tsv" | cut -f2-3) \
        <(grep -P '^c880\t' "$K/hand-verdicts.tsv" | cut -f2-3) > /dev/null && echo same || echo DIFFERENT)"
    cbox python3 -I "$REPO/compare.py" "$W/tools/base" "$W/tools/head" "$K/tools.tsv"
    echo "compare rc=$?"
    echo "=== compare done $(date -Is)"
fi
finish 0
