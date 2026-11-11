#!/usr/bin/env bash
# C615 (Q1100): a drafted token's n-gram rows read ahead as it is sampled (LLAMA_DRAFT_PREFETCH), checked on a build
# machine, CPU only: every build and test in a container at low CPU priority, one at a time under LOCK. Made from C738's
# checks.sh (docs/data/2026-11-11_c738_verify_prepare).
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of TIP, every target, at $WORK/cpu/tip/tree/{src,build}
#   server   server_check.sh on a random qwen4exp model with an MTP block (written by TIP's test-llama-archs) with its
#            n-gram table mapped: BASE_SRV's llama-server against TIP's, the switch unset, on, on with its counts, with
#            a zero cap, the counts alone, and with the backend sampler
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with the
#            switch unset, =1 and =1 with its counts (its random models keep their tables resident: nothing calls the
#            prefetch, the variables alone must change nothing)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   merge    a test merge of TIP and MERGE_BRANCH (git merge-tree, never committed): a CPU build of it, server_check's
#            m arms (nothing set; the switch on; both it and LLAMA_NGRAM_PREFETCH on) against BASE_SRV, and
#            MERGE_BRANCH's own test (MERGE_TEST, e.g. test-llama-archs --ngram-prefetch) at seeds 1-3 with this switch on
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   rsrv     server_check's base and ons arms with the rig tarball's llama-server on the CPU (its CUDA backend loads
#            against the stub driver and finds no card; its CPU code has no AVX)
#   trace    server_check's troff and trc arms (C701's LLAMA_UBATCH_TRACE=1, the switch off and on), then the rig row's
#            readout (c615_read.py) on both logs: the readout's check on real lines
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [MERGE_BRANCH=pascal-p106-v4-b12246-c761]
#   [MERGE_TEST='--ngram-prefetch'] [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>] [DOCKER_ARGS=...]
#   [IMAGE=pq-c735-srv:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), logs in $WORK/logs and $WORK/checks; done when $WORK/C615_DONE
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
MERGE_BRANCH=${MERGE_BRANCH:-pascal-p106-v4-b12246-c761}
MERGE_TEST=${MERGE_TEST:---ngram-prefetch}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-cpu server off ctest merge build rsrv trace}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
SARGS='--override-kv qwen4exp.context_length=int:16384'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
M=$K/models/qwen4exp-moe-mtp.gguf
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C615_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c615 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

# cpu_build <tag> <tree-ish> <description>: a static native CPU build of every target at $C/<tag>
cpu_build() {
    local tag=$1 treeish=$2 desc=$3 R T rc
    R=$C/$tag T=$C/$tag/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$treeish" | tar -x -C "$T/src"
    echo "$desc" > "$R/tree.txt"
    echo "=== cpu $tag $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
    rc=$?
    for b in test-llama-archs llama-server; do
        cp "$T/build/bin/$b" "$R/bin/" 2> /dev/null
    done
    verdict "cpu-$tag" "$([ $rc = 0 ] && [ -x "$R/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $desc; every target; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
}

# srv <tag> <BIN> <MBIN> <arms>: server_check.sh into $K/<tag>
srv() {
    local tag=$1 bin=$2 mbin=$3 arms=$4 rc
    box bash -c "BASE=$BASE_SRV BIN=$bin MBIN='$mbin' MODEL=$M OUT=$K/$tag/out THREADS=$THREADS ARMS='$arms' \
        SARGS='$SARGS' nice -n 19 bash $RUN/server_check.sh" > "$K/$tag.console.txt" 2>&1
    rc=$?
    verdict "$tag" "$([ "$rc" = 0 ] && grep -q '^PASS' "$K/$tag/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -v '^PASS\|^FAIL' "$K/$tag/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-1800)"
}

if step cpu; then
    cpu_build tip "$TIP" "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")"
fi

if step server; then
    echo "=== server $(date -Is)"
    rm -rf "$K/server" "$K/models" && mkdir -p "$K/server" "$K/models"
    # the random qwen4exp MoE model with one MTP block, written by TIP's test-llama-archs
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $K/models --mtp > $K/models/save.log 2>&1"
    srv server "$C/tip/bin" '' 'base off on ons cap0 offs bbase bons'
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
    cmp_arm dp1 'LLAMA_DRAFT_PREFETCH=1'
    cmp_arm dps 'LLAMA_DRAFT_PREFETCH=1 LLAMA_DRAFT_PREFETCH_STATS=1'
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

if step merge; then
    echo "=== merge $(date -Is)"
    git -C "$SRC" fetch -q "$REMOTE" "$MERGE_BRANCH:refs/remotes/$REMOTE/$MERGE_BRANCH"
    MT=$(git -C "$SRC" rev-parse "$REMOTE/$MERGE_BRANCH")
    # the merged tree, without a commit or a work tree; a conflict leaves no tree
    MTREE=$(git -C "$SRC" merge-tree --write-tree "$TIP" "$MT" 2> "$WORK/logs/merge-tree.err" | head -n 1)
    if git -C "$SRC" cat-file -e "$MTREE^{tree}" 2> /dev/null && [ "$(git -C "$SRC" merge-tree --write-tree --name-only "$TIP" "$MT" | wc -l)" = 1 ]; then
        verdict merge-tree OK "$BRANCH ${TIP:0:9} + $MERGE_BRANCH ${MT:0:9}: tree ${MTREE:0:9}, no conflict"
        cpu_build merge "$MTREE" "tree $MTREE = $BRANCH $TIP + $MERGE_BRANCH $MT (git merge-tree)"
        [ -s "$M" ] || { mkdir -p "$K/models"; box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $K/models --mtp > $K/models/save.log 2>&1"; }
        srv merge-server "$C/tip/bin" "$C/merge/bin" 'base moff mons mboth'
        for s in 1 2 3; do
            log=$K/merge-test-s$s.log
            box bash -c 'env LLAMA_DRAFT_PREFETCH=1 LLAMA_DRAFT_PREFETCH_STATS=1 nice -n 19 '"$C"'/merge/bin/test-llama-archs '"$MERGE_TEST"' -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "merge-test-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" \
                "rc=$rc; test-llama-archs $MERGE_TEST -s $s with LLAMA_DRAFT_PREFETCH=1 LLAMA_DRAFT_PREFETCH_STATS=1; $(grep -a -c 'draft-prefetch:' "$log") draft-prefetch lines; $sum"
        done
    else
        verdict merge-tree FAIL "$BRANCH ${TIP:0:9} + $MERGE_BRANCH ${MT:0:9}: $(git -C "$SRC" merge-tree --write-tree --name-only "$TIP" "$MT" | tail -n +2 | tr '\n' ' ')"
    fi
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

if step rsrv; then
    echo "=== rsrv $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && [ -s "$M" ]; then
        rm -rf "$K/rsrv" && mkdir -p "$K/rsrv"
        # curl and jq from the distribution (the container needs the network for apt only)
        # shellcheck disable=SC2086
        locked docker run $D -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c '
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq curl jq > /dev/null &&
            mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
            export LD_LIBRARY_PATH='"$T"'/lib:'"$T"'/bin:/usr/local/cuda/lib64:/tmp/stub &&
            BASE='"$T"'/bin BIN='"$T"'/bin MODEL='"$M"' OUT='"$K"'/rsrv/out THREADS='"$THREADS"' ARMS="base ons" \
                SARGS="'"$SARGS"'" nice -n 19 bash '"$RUN"'/server_check.sh' > "$K/rsrv.console.txt" 2>&1
        rc=$?
        verdict rsrv "$([ "$rc" = 0 ] && grep -q '^PASS' "$K/rsrv/out/verdict.txt" && echo OK || echo FAIL)" \
            "rc=$rc; the rig build's llama-server on the CPU, base (unset) and ons; $(grep -v '^PASS\|^FAIL' "$K/rsrv/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-800)"
    else
        verdict rsrv FAIL "no llama-server in the tarball tree, or no model"
    fi
fi

if step trace; then
    echo "=== trace $(date -Is)"
    srv trace "$C/tip/bin" '' 'base troff trc'
    for a in troff trc; do
        box bash -c "nice -n 19 python3 -I $RUN/c615_read.py $K/trace/out/$a/server.log $K/trace/read/$a --load $a" \
            >> "$K/trace.read.txt" 2>&1
    done
    verdict trace-read "$([ -s "$K/trace/read/trc/summary.tsv" ] && [ -s "$K/trace/read/troff/summary.tsv" ] && echo OK || echo FAIL)" \
        "$(tr '\n' ';' < "$K/trace.read.txt" | cut -c1-600) $(awk -F'\t' '$2 == "verify" { printf "%s: %s verify evaluations, rows %s, ahead %s, pages %s, resident %s; ", $1, $3, $9, $10, $11, $12 }' "$K"/trace/read/*/summary.tsv 2>/dev/null)"
fi

echo "=== done $(date -Is)"
touch "$WORK/C615_DONE"
