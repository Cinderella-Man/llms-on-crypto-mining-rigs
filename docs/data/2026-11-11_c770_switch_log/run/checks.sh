#!/usr/bin/env bash
# C770 (Q1355): llama-server's one line at load with every LLAMA_ and GGML_ switch set, at the server's default
# verbosity, checked on a build machine, CPU only: every build and test in a container (without network but for the
# rig build) at low CPU priority, one at a time under LOCK. Made from C771's checks.sh
# (docs/data/2026-11-11_c771_draft_kpool_split).
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of TIP, every target, at $WORK/cpu/tip/tree/{src,build}
#   names    names.sh: the branch's scan (common/switch-names.cmake) on BASE, TIP, DAY and each of BRANCHES, and the
#            names documented in DOCS against them
#   server   server_check.sh: BASE_SRV's llama-server against TIP's on the CPU at the default verbosity, with no
#            switch, with five the branch reads, and with those and three it does not; the random qwen4exp model with
#            one MTP block and its MTP-only draft (MODELS)
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   rescan   a later switch with no edit to the list: a comment naming LLAMA_C770_LATER_SWITCH added to
#            src/llama-context.cpp in TIP's tree, llama-server rebuilt, its line with that variable set, then the
#            comment removed and llama-server rebuilt again
#   build   the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt;
#            the tarball's libraries searched for the line's text
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_BRANCH=<its base branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> MODELS=<dir with C762's qwen4exp-moe-mtp{,-draft}.gguf>
#   [DAY=<the day merge's branch>] [BRANCHES='<branch> ...'] [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), logs in $WORK/logs and $WORK/checks; done when $WORK/C770_DONE
#   exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
BASE_SRV=$(realpath "${BASE_SRV:?BASE_SRV=<dir with the base llama-server>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BASE_BRANCH:?BASE_BRANCH=<its base branch>}"
RUN=$(dirname "$(realpath "$0")")
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
DAY=${DAY:-}
BRANCHES=${BRANCHES:-}
STEPS=${STEPS:-cpu names server off ctest rescan build}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C770_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$SRC:$SRC:ro" -v "$REPO:$REPO:ro" \
                -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
for b in "$BRANCH" "$BASE_BRANCH" $DAY $BRANCHES; do
    git -C "$SRC" fetch -q "$REMOTE" "$b:refs/remotes/$REMOTE/$b"
done
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
T=$C/tip/tree
echo "=== c770 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step cpu; then
    R=$C/tip
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
    echo "=== cpu $(date -Is)"
    box bash -c "export CCACHE_DIR=$C/ccache CCACHE_BASEDIR=$T
        cmake -S $T/src -B $T/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
            -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            > $R/cmake.log 2>&1 &&
        nice -n 19 cmake --build $T/build -j $JOBS -- -k > $R/build.log 2>&1"
    rc=$?
    cp "$T/build/bin/test-llama-archs" "$T/build/bin/llama-server" "$R/bin/" 2> /dev/null
    n=$(grep -cE '^    "[A-Z0-9_]+",$' "$T/build/common/switch-names.cpp" 2> /dev/null)
    verdict cpu-tip "$([ $rc = 0 ] && [ -x "$R/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); every target; the generated list: ${n:-no file} names; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

if step names; then
    echo "=== names $(date -Is)"
    N=$K/names
    rm -rf "$N" && mkdir -p "$N"
    box bash -c "SRC=$SRC SCAN=$T/src/common/switch-names.cmake DOCS=$REPO/docs/switches.md OUT=$N BASE=$BASE_BRANCH \
        TIP=$BRANCH DAY='$DAY' BRANCHES='$BRANCHES' REMOTE=$REMOTE nice -n 19 bash $RUN/names.sh" > "$N/console.txt" 2>&1
    rc=$?
    # the scan in the script against the list the build wrote
    same=$(cmp -s <(grep -oE '^    "[A-Z0-9_]+",$' "$T/build/common/switch-names.cpp" | tr -d ' ",') \
                  "$N/names/${BRANCH#pascal-p106-v4-b12246-}.txt" && echo "the build's list = the scan's" || echo "the build's list differs from the scan's")
    verdict names "$([ $rc = 0 ] && grep -q '^tip = base' "$N/names_verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $same; $(tr '\n' ';' < "$N/names_verdict.txt")"
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    rm -rf "$S" && mkdir -p "$S/models"
    M=$S/models/qwen4exp-moe-mtp.gguf
    DR=$S/models/qwen4exp-moe-mtp-draft.gguf
    # the model loads only with its MTP block as a draft (-md)
    cp "${MODELS:?MODELS=<dir with the two qwen4exp-moe-mtp files of C762>}"/qwen4exp-moe-mtp{,-draft}.gguf "$S/models/"
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin MODEL=$M DRAFT=$DR OUT=$S/out THREADS=$THREADS \
        SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    verdict server "$([ "$rc" = 0 ] && grep -q '^PASS' "$S/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; models $(md5sum "$M" "$DR" | cut -c1-32 | tr '\n' ' '); $(grep -v '^PASS\|^FAIL' "$S/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-3000)"
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
    box bash -c "cd $T/build && nice -n 19 ctest -j $JOBS --timeout 1800 --output-on-failure -E '$CTEST_EXCLUDE' --output-junit $K/ctest.xml" \
        > "$K/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$K/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$K/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}; not run: $CTEST_EXCLUDE"
fi

if step rescan; then
    echo "=== rescan $(date -Is)"
    Q=$K/rescan
    rm -rf "$Q" && mkdir -p "$Q"
    F=$T/src/src/llama-context.cpp
    line() { # <tag> [VAR=value]: the line of the tree's llama-server with no model to load
        box bash -c "env -i PATH=\$PATH $2 $T/build/bin/llama-server -m /nonexistent.gguf > $Q/$1.log 2>&1; true"
        grep -a 'common_params_print_info: switches:' "$Q/$1.log" | sed 's/.*switches: /switches: /'
    }
    build() { # <tag>
        box bash -c "export CCACHE_DIR=$C/ccache CCACHE_BASEDIR=$T; nice -n 19 cmake --build $T/build -j $JOBS --target llama-server > $Q/build-$1.log 2>&1"
    }
    l0=$(line before LLAMA_C770_LATER_SWITCH=1)
    cp "$F" "$Q/llama-context.cpp.orig"
    echo '// C770 check: "LLAMA_C770_LATER_SWITCH"' >> "$F"
    build added; rc1=$?
    l1=$(line added LLAMA_C770_LATER_SWITCH=1)
    cp "$Q/llama-context.cpp.orig" "$F"
    build removed; rc2=$?
    l2=$(line removed LLAMA_C770_LATER_SWITCH=1)
    cmp -s "$Q/llama-context.cpp.orig" "$F" || rc2=9
    printf 'state\tline\n%s\t%s\n%s\t%s\n%s\t%s\n' before "$l0" added "$l1" removed "$l2" > "$Q/rescan.tsv"
    ok=FAIL
    [[ $rc1 = 0 && $rc2 = 0 && $l0 == *"not read by this build: LLAMA_C770_LATER_SWITCH=1"* && \
       $l1 == "switches: 1 set: LLAMA_C770_LATER_SWITCH=1 "* && $l2 == "$l0" ]] && ok=OK
    verdict rescan "$ok" "rc=$rc1,$rc2; before: $l0; with the comment: $l1; after: $l2; $(grep -ac "Listing the LLAMA_" "$Q/build-added.log") scan and $(grep -ac "switch-names.cpp.o" "$Q/build-added.log") compile of the list in the rebuild"
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
    tree=$(ls -d "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | tail -n 1)
    hits=$(grep -rla 'this build reads ' "$tree" 2> /dev/null | sed "s|$tree/||" | tr '\n' ' ')
    names=$(grep -rla 'LLAMA_SAMPLING_TOP_K_ROWS' "$tree/lib" "$tree/bin" 2> /dev/null | sed "s|$tree/||" | tr '\n' ' ')
    verdict build-line "$([ -n "$hits" ] && echo OK || echo FAIL)" "the line's text in: ${hits:-none}; a name of the list in: ${names:-none}"
fi

echo "=== done $(date -Is)"
touch "$WORK/C770_DONE"
