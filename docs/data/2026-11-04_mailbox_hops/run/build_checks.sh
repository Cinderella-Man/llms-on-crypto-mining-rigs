#!/usr/bin/env bash
# C143 (Q682, Q270) on a build machine: the rig build of pascal-p106-v3-b12076-mailbox (card hops through a mailbox in
# mapped host memory, GGML_CUDA_HOP_MAILBOX, off by default, on -kshape), the test binaries from its tree, the kernels'
# machine code, a CPU-only build and its checks, and the hop microbenchmark with the streamed mailbox, in containers
# without a GPU:
#   1. build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1); test-hop-mailbox, test-hop-cast,
#                test-llama-archs and test-llama-context from its build tree (a second tarball, for the GPU checks)
#   2. sass      the mailbox kernels in the rig build: sm_61 machine code and compute_61 PTX of k_hop_mb_send and
#                k_hop_mb_recv, their registers and spills
#   3. cpubuild  the same tree without CUDA: test-hop-mailbox, test-hop-cast
#   4. cpu       test-hop-mailbox on the CPU build (no CUDA card: nothing to check, exit 0) and test-hop-cast --unit-only
#   5. bench     bench/megakernel_p106/build.sh (the hop microbenchmark with the streamed mailbox), and the rig bundle:
#                the binary, rig_mailbox.sh, fn_line.sh and fn_requests.py
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-mailbox] [REMOTE=fork] [JOBS=8] \
#   [LOCK=<file>] [DOCKER_ARGS=...] [STEPS='build sass cpubuild cpu bench'] bash build_checks.sh
#   WORK    the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs REMOTE as a
#           remote with a fetch URL that resolves inside the container)
#   LOCK    a file to flock around each build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra-<sha>.tar.gz, $WORK/cpu/bin/,
#   $WORK/bundle/, $WORK/checks/<check>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when
#   $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-mailbox}
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass cpubuild cpu bench}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks" "$WORK/cpu"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
echo "=== chain started $(date -Is), steps: $STEPS"

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "$REMOTE/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-hop-mailbox test-hop-cast test-llama-archs test-llama-context \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-hop-mailbox build/bin/test-hop-cast build/bin/test-llama-archs build/bin/test-llama-context /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
cpubuild() { # <source dir under /work> <output dir under /work> <targets>
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e S="$1" -e O="$2" -e T="$3" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S "/work/$S" -B "/work/$O/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build "/work/$O/build" -j "$JOBS" --target $T > "/work/logs/$O-build.log" 2>&1 \
            || { tail -40 "/work/logs/$O-build.log"; exit 1; }
        mkdir -p "/work/$O/bin"
        for t in $T; do cp "/work/$O/build/bin/$t" "/work/$O/bin/"; done
        chown -R "$HOST_UID:$HOST_GID" "/work/$O" /work/logs'
}
export -f build cpubuild
export WORK BRANCH REMOTE JOBS IMAGE D REPO

T=""
if step build; then
    echo "=== 1. build $(date -Is)"
    locked bash -c build
    rc=$?
    echo "=== build ended $(date -Is) rc=$rc"
    if [ $rc != 0 ]; then verdict build FAIL "see logs/build.log, logs/extra.log"; date -Is > "$WORK/DONE"; exit 1; fi
    t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
    T=$WORK/$(basename "$t" .tar.gz)
    sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
    (cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
    (cd "$WORK" && tar czf "extra-$sha.tar.gz" extra && md5sum "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
    cp "$WORK"/extra/test-* "$T/bin/"
    git -C "$WORK/src" rev-parse HEAD > "$WORK/COMMIT"
    verdict build OK "$(basename "$t"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"
fi
[ -n "$T" ] || T=$(ls -td "$WORK"/llama.cpp-*-"$BRANCH" 2>/dev/null | head -1)

if step sass && [ -n "$T" ]; then
    echo "=== 2. sass $(date -Is)"
    mkdir -p "$WORK/checks/sass"
    docker run --rm $D -v "$T:/t:ro" -v "$WORK/checks/sass:/k" "$IMAGE" bash -c '
        b=/t/bin/test-hop-mailbox
        cuobjdump -sass -arch sm_61 $b | grep -E "Function : .*k_hop_mb_(send|recv)" > /k/sm61_functions.txt
        cuobjdump -ptx $b | grep -E "\.entry .*k_hop_mb_(send|recv)" > /k/ptx_entries.txt
        cuobjdump -res-usage $b 2> /dev/null | grep -A1 -E "Function .*k_hop_mb_(send|recv)" > /k/res_usage.txt
        cuobjdump -sass -arch sm_61 -fun "$(grep -o "_Z[A-Za-z0-9_]*k_hop_mb_send[A-Za-z0-9_]*" /k/sm61_functions.txt | head -1)" $b \
            > /k/send_sm61.sass 2> /dev/null
        cuobjdump -sass -arch sm_61 -fun "$(grep -o "_Z[A-Za-z0-9_]*k_hop_mb_recv[A-Za-z0-9_]*" /k/sm61_functions.txt | head -1)" $b \
            > /k/recv_sm61.sass 2> /dev/null
        chown -R '"$(id -u):$(id -g)"' /k'
    ns=$(grep -c k_hop_mb "$WORK/checks/sass/sm61_functions.txt"); np=$(grep -c k_hop_mb "$WORK/checks/sass/ptx_entries.txt")
    spill=$(grep -o 'STACK:[0-9]*' "$WORK/checks/sass/res_usage.txt" | sort -u | tr '\n' ' ')
    membar=$(grep -c 'MEMBAR.SYS' "$WORK/checks/sass/send_sm61.sass" "$WORK/checks/sass/recv_sm61.sass" | tr '\n' ' ')
    verdict sass "$([ "$ns" = 2 ] && [ "$np" = 2 ] && echo OK || echo FAIL)" \
        "$ns sm_61 functions, $np PTX entries; $spill; MEMBAR.SYS per kernel: $membar; $(grep -o 'REG:[0-9]*' "$WORK/checks/sass/res_usage.txt" | tr '\n' ' ')"
fi

if step cpubuild; then
    echo "=== 3. cpubuild $(date -Is)"
    locked bash -c 'cpubuild src cpu "test-hop-mailbox test-hop-cast"' > "$WORK/logs/cpubuild.console" 2>&1
    rc=$?
    git -C "$WORK/src" rev-parse HEAD > "$WORK/cpu/bin/COMMIT" 2>/dev/null
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(cat "$WORK/cpu/bin/COMMIT" 2>/dev/null)"
fi

if step cpu; then
    echo "=== 4. cpu $(date -Is)"
    mkdir -p "$WORK/checks/cpu"
    docker run --rm $D -v "$WORK/cpu/bin:/x:ro" -v "$WORK/checks/cpu:/k" "$IMAGE" bash -c '
        /x/test-hop-mailbox > /k/mailbox.log 2>&1; echo "mailbox rc=$?" > /k/rc.txt
        nice -n 19 /x/test-hop-cast --unit-only > /k/unit.log 2>&1; echo "unit rc=$?" >> /k/rc.txt
        chown -R '"$(id -u):$(id -g)"' /k'
    verdict cpu "$(grep -q 'rc=[1-9]' "$WORK/checks/cpu/rc.txt" && echo FAIL || echo OK)" \
        "$(tr '\n' ' ' < "$WORK/checks/cpu/rc.txt"); $(tail -n 1 "$WORK/checks/cpu/mailbox.log"); $(tail -n 1 "$WORK/checks/cpu/unit.log")"
fi

if step bench; then
    echo "=== 5. bench $(date -Is)"
    locked env DOCKER_ARGS="$D" nice -n 19 bash "$REPO/bench/megakernel_p106/build.sh" > "$WORK/logs/bench.log" 2>&1
    rc=$?
    B=$REPO/bench/megakernel_p106
    if [ $rc = 0 ]; then
        rm -rf "$WORK/bundle" && mkdir -p "$WORK/bundle"
        F=$REPO/docs/data/2026-11-03_flash_next_baseline/run
        cp "$B/build/megakernel_p106" "$B/build/BUILDINFO.txt" "$HERE/rig_mailbox.sh" "$F/fn_line.sh" "$F/fn_requests.py" \
            "$WORK/bundle/"
        (cd "$WORK/bundle" && md5sum megakernel_p106 rig_mailbox.sh fn_line.sh fn_requests.py > MD5SUMS)
        cp "$B/build/ptxas.log" "$WORK/logs/bench-ptxas.log"
    fi
    verdict bench "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^(md5|avx|highest)' "$B/build/BUILDINFO.txt" 2>/dev/null | tr '\n' ';')"
fi

date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
