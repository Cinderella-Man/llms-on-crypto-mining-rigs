#!/usr/bin/env bash
# C741 (Q1310): fewer yields to the task queue in llama-server's speculative step (LLAMA_SERVER_YIELD), checked on a
# build machine, CPU only: every build and test in a container (without network but for the rig build) at low CPU
# priority, one at a time under LOCK. Made from C738's checks.sh (docs/data/2026-11-11_c738_verify_prepare).
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of the branch's tip, every target, at $WORK/cpu/tip/tree/{src,build}
#            with one ccache (relative paths); test-llama-archs, llama-server and test-server-component kept in
#            $WORK/cpu/tip/bin
#   unit     test-server-component server_queue_yield, five times: the queue's yields with a wake every yield and with
#            the lazy wake (the counts of yields and wakes; reads answered in the yield; a completion declined in the
#            yield and taken at the next step)
#   server   server_check.sh on the qwen4exp model with an MTP block: BASE_SRV's llama-server against TIP's, the switch
#            unset, =1 and =2, with and without the backend sampler, one slot; and two slots with a second request
#            arriving mid-reply and reads of /slots and /metrics all along
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with the
#            switch unset and =2 (llama-server alone reads it: the variable must change nothing there)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), logs in $WORK/logs and $WORK/checks; done when $WORK/C741_DONE
#   exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
BASE_SRV=$(realpath "${BASE_SRV:?BASE_SRV=<dir with the base llama-server>}")
: "${BRANCH:?BRANCH=<fork branch>}"
RUN=$(dirname "$(realpath "$0")")
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-cpu unit server off ctest build}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C741_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c741 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step cpu; then
    R=$C/tip T=$C/tip/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
    echo "=== cpu $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
    rc=$?
    for b in test-llama-archs llama-server test-server-component; do
        cp "$T/build/bin/$b" "$R/bin/" 2> /dev/null
    done
    verdict cpu-tip "$([ $rc = 0 ] && [ -x "$R/bin/llama-server" ] && [ -x "$R/bin/test-server-component" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); every target; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

if step unit; then
    echo "=== unit $(date -Is)"
    U=$K/unit
    rm -rf "$U" && mkdir -p "$U"
    for i in 1 2 3 4 5; do
        # the filter is matched against the whole name, the subtests' included
        box bash -c "LLAMA_TEST_VERBOSE=1 timeout 300 nice -n 19 $C/tip/bin/test-server-component 'server_queue_yield.*' > $U/run$i.log 2>&1"
        rc=$?
        f=$(grep -a '^failures' "$U/run$i.log" | tr -s ' ')
        a=$(grep -a '^assertions' "$U/run$i.log" | tr -s ' ')
        n=$(grep -a -c -E '^ +(a wake every yield|lazy wake) \([0-9]+ assertion\(s\)\) +\[PASS\]' "$U/run$i.log")
        verdict "unit-$i" "$([ $rc = 0 ] && [[ "$f" == "failures : 0" ]] && [[ "$a" != "assertions : 0" ]] && [ "$n" = 2 ] && echo OK || echo FAIL)" \
            "rc=$rc; ${a:-no assertions line}; ${f:-no failures line}; $n of 2 subtests passed"
    done
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    rm -rf "$S" && mkdir -p "$S/models"
    # the random qwen4exp MoE model with one MTP block, written by TIP's test-llama-archs
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp > $S/models/save.log 2>&1"
    M=$S/models/qwen4exp-moe-mtp.gguf
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin MODEL=$M OUT=$S/out THREADS=$THREADS \
        SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    verdict server "$([ "$rc" = 0 ] && grep -q '^PASS' "$S/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -v '^PASS\|^FAIL' "$S/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-2000)"
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
    cmp_arm() { # <tag> <env> [args]
        local tag=$1 envs=$2 rc n bad
        shift 2
        rc=$(archs "$tag" "$C/tip/bin" "$envs" "$@")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    }
    cmp_arm tip ''
    cmp_arm y2 'LLAMA_SERVER_YIELD=2'
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

if step build; then
    echo "=== build $(date -Is)"
    mkdir -p "$WORK/rig"
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

echo "=== done $(date -Is)"
touch "$WORK/C741_DONE"
