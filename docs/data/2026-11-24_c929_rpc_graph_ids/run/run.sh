#!/usr/bin/env bash
# C929 (2026-11-24): ggml's RPC client forgetting the graphs its server dropped at a FREE_BUFFER. Processor only, no GPU.
# Three trees of Miner Llama:
#   base  TEST: pascal-p106-v5 with the new test only (test-rpc-graph-cache), the client as it was
#   head  COMMIT: TEST plus the client's fix
#   c901  C901's commit (its pipe-slot re-plan count and LLAMA_PIPE_SLOT_PLAN) with TEST's and COMMIT's patches on top,
#         the build C901's server check ran on, fixed
# Steps (STEPS, in this order):
#   src     a local clone of SEED, the fork's branches BRANCH and C901_BRANCH fetched and checked against COMMIT and
#           C901; TEST must be COMMIT's parent; the compiler cache hard-linked from CPU_CCACHE, its counters zeroed
#   cpu     cpu_tests.sh (REPO) on base and head@base with GGML_RPC on, static: a CPU build of every target under LOCK,
#           ctest without network (with the RPC tests, the new one among them), test-llama-archs (plain, write, compare)
#           and head's logits against base's (ident); then ccache's counters
#   rpc     for each of base, head and c901: llama-server, ggml-rpc-server and test-llama-archs, built with the cpu
#           step's options and compiler cache (static, native, GGML_RPC on, tests on) in $W/rpc-<tree>
#   server  c929_server.sh (REPO) on each rpc build, one at a time under LOCK, all on head's random Flint model:
#             head  cpu_off rpc_noreuse_off rpc_reuse_off
#             base  rpc_noreuse_off rpc_reuse_off  (the reuse arm should show the abort)
#             c901  cpu_off cpu_on rpc_noreuse_off rpc_noreuse_on rpc_reuse_off rpc_reuse_on
#   ctest   ctest test by test for base and head ($W/checks/ctest.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone> REPO=<folder with cpu_tests.sh and c929_server.sh> (under W)
#   [TEST=<sha>] [COMMIT=<sha>] [C901=<sha>] [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS='src cpu rpc server ctest'] [JOBS=8]
#   bash run.sh
# Every container: labels pascal-quants=1 and c929=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh and c929_server.sh>}
TEST=${TEST:-3c8f51216d86a11ca581dc1d03735a8b5b7e4c81}
COMMIT=${COMMIT:-60c8ad7884fe9734adf93568e0be6e18a3dc0f02}
C901=${C901:-19048bb6d}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5-rpc-graph-ids}
C901_BRANCH=${C901_BRANCH:-pascal-p106-v5-pipe-slot-plan}
STEPS=${STEPS:-src cpu rpc server ctest}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-c735-srv:24.04}
LBL="--label pascal-quants=1 --label c929=1"
MEM="--memory=16g --memory-swap=16g"
D="--rm $LBL --cpu-shares=256 $MEM --network none -v $W:$W"
CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DGGML_RPC=ON"
CC=$W/cpu/ccache
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c929=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run $D -e CCACHE_DIR="$CC" "$CPU_IMAGE" ccache "$@"; }
export LC_ALL=C
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
echo "=== start $(date -Is): base $TEST, head $COMMIT, c901 $C901 + both; steps: $STEPS"

S=$W/src
if step src; then
    [ -d "$S/.git" ] || git clone -q --no-checkout "$SEED" "$S" || finish 1
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" "$C901_BRANCH:refs/remotes/fork/$C901_BRANCH" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/fork/$BRANCH")
    echo "fork/$BRANCH = $got"
    [ "$got" = "$(git -C "$S" rev-parse "$COMMIT")" ] || { echo "$BRANCH is not at $COMMIT: stop"; finish 1; }
    [ "$(git -C "$S" rev-parse "$COMMIT^")" = "$(git -C "$S" rev-parse "$TEST")" ] || { echo "$TEST is not $COMMIT's parent: stop"; finish 1; }
    git -C "$S" merge-base --is-ancestor "$C901" "refs/remotes/fork/$C901_BRANCH" || { echo "$C901 is not in $C901_BRANCH: stop"; finish 1; }
    mkdir -p "$W/cpu"
    [ -d "$CC" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$CC"; }
    [ -d "$CC" ] && cc -z > /dev/null
    echo "=== sources ready $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$S" TREES="base=$TEST head=$COMMIT@base" STEPS="tree build ctest archs ident" SHARED=OFF \
        CMAKE_ARGS="-DGGML_RPC=ON" JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" \
        IMAGE="$CPU_IMAGE" bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu tests rc=$?"
    cat "$W/cpu/summary.tsv"
    cc -s > "$W/checks/ccache-cpu.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-cpu.txt"
    echo "=== cpu done $(date -Is)"
fi

if step rpc; then
    for t in base head c901; do
        R=$W/rpc-$t
        rm -rf "$R" && mkdir -p "$R/src"
        case $t in
            base) git -C "$S" archive "$TEST" | tar -x -C "$R/src"; echo "commit $(git -C "$S" rev-parse "$TEST")" > "$R/tree.txt" ;;
            head) git -C "$S" archive "$COMMIT" | tar -x -C "$R/src"; echo "commit $(git -C "$S" rev-parse "$COMMIT")" > "$R/tree.txt" ;;
            c901) git -C "$S" archive "$C901" | tar -x -C "$R/src"
                  rc=0
                  for p in "$TEST" "$COMMIT"; do
                      git -C "$S" format-patch -1 --stdout "$p" | GIT_CEILING_DIRECTORIES=$R git -C "$R/src" apply --whitespace=nowarn - || rc=$?
                  done
                  { echo "commit $(git -C "$S" rev-parse "$C901")"; echo "patch $(git -C "$S" rev-parse "$TEST")"; echo "patch $(git -C "$S" rev-parse "$COMMIT")"; } > "$R/tree.txt"
                  [ $rc = 0 ] || { echo "c901: a patch did not apply"; continue; } ;;
        esac
        s=$(date +%s)
        locked docker run $D -e CCACHE_DIR="$CC" -e CCACHE_BASEDIR="$R" "$CPU_IMAGE" bash -c "
            cmake -S $R/src -B $R/build $CMAKE_OPTS > $R/cmake.log 2>&1 &&
            nice -n 19 cmake --build $R/build -j $JOBS --target llama-server ggml-rpc-server test-llama-archs > $R/build.log 2>&1"
        echo "rpc-$t: build rc=$? s=$(( $(date +%s) - s )); $(tr '\n' ' ' < "$R/tree.txt")" | tee "$R/RC"
    done
    cc -s > "$W/checks/ccache-rpc.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-rpc.txt"
    echo "=== rpc done $(date -Is)"
fi

if step server; then
    cp "$REPO/c929_server.sh" "$W/checks/"
    model=""
    for t in head base c901; do
        O=$W/server-$t
        rm -rf "$O" && mkdir -p "$O"
        case $t in
            head) arms="cpu_off rpc_noreuse_off rpc_reuse_off" ;;
            base) arms="rpc_noreuse_off rpc_reuse_off" ;;
            c901) arms="cpu_off cpu_on rpc_noreuse_off rpc_noreuse_on rpc_reuse_off rpc_reuse_on" ;;
        esac
        [ -x "$W/rpc-$t/build/bin/llama-server" ] || { echo "server-$t: no build"; continue; }
        s=$(date +%s)
        locked docker run $D -w /tmp -e BIN="$W/rpc-$t/build/bin" -e VOCAB="$W/rpc-head/src/models/ggml-vocab-gpt-2.gguf" \
            -e OUT="$O" -e ARMS="$arms" ${model:+-e MODEL="$model"} "$SRV_IMAGE" timeout 2400 bash "$REPO/c929_server.sh" \
            > "$O/console.txt" 2>&1
        echo "server-$t: rc=$? s=$(( $(date +%s) - s )); arms: $arms"
        [ -n "$model" ] || model=$(ls "$W/server-head/model/"*.gguf 2>/dev/null | head -n 1)
        grep -aE '^(model:|.*alive after|.*client aborts|.*rpc server [0-9]+:|Q[1-4] )' "$O/console.txt"
    done
    echo "=== server done $(date -Is)"
fi

if step ctest; then
    ctest_tsv "$W/cpu/base/ctest.log" > "$W/checks/ctest-base.tsv"
    ctest_tsv "$W/cpu/head/ctest.log" > "$W/checks/ctest-head.tsv"
    join -t "$(printf '\t')" -a 1 -a 2 -e missing -o 0,1.2,2.2 "$W/checks/ctest-base.tsv" "$W/checks/ctest-head.tsv" \
        | awk -F'\t' 'BEGIN { OFS = "\t"; print "test", "base", "head", "same" } { print $1, $2, $3, ($2 == $3 ? "yes" : "no") }' \
        > "$W/checks/ctest.tsv"
    echo "ctest: $(tail -n +2 "$W/checks/ctest.tsv" | wc -l) tests; not the same in base and head:"
    tail -n +2 "$W/checks/ctest.tsv" | awk -F'\t' '$4 == "no"'
    echo "passed: base $(grep -c 'Passed' "$W/checks/ctest-base.tsv"), head $(grep -c 'Passed' "$W/checks/ctest-head.tsv")"
fi
finish 0
