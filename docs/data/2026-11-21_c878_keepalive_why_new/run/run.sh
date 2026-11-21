#!/usr/bin/env bash
# C878 (2026-11-21): Miner Llama's GGML_CUDA_KEEPALIVE_TRACE (fork branch pascal-p106-v5-keepalive-why-new) built and
# checked on a build machine's CPU against its base, pascal-p106-v5. No GPU work: the CUDA build's test binary runs
# against the driver stub, and the GPU code is read with cuobjdump.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs and
#           test-cuda-keepalive added (as the base's tarball), the web UI archive pinned (HF_UI_VERSION), under LOCK;
#           warnings against BASE_RIG_LOG (the base tarball's build log)
#   ka      test-cuda-keepalive of the head's rig build and of the base's (BASE_RIG), in the CUDA image with the driver
#           stub, no card: every check of both, the head's with the trace's own
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the head: a CPU build of every target and ctest
#           without network; its result test by test against BASE_CTEST (the base's, written by the same script)
#   sass    cuobjdump -sass of both rig llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/), every differing kernel extracted and classified
#           (sass_classify.py, from docs/data/2026-11-17_c854b_upstream_newest/run/)
# Usage: W=<work dir> SEED=<llama.cpp clone> BASE_RIG=<base rig build folder> BASE_RIG_LOG=<its build log>
#   BASE_CTEST=<the base's ctest table> REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py
#   sass_classify.py> [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS=...] [JOBS=8] bash run.sh
# Every container: labels pascal-quants=1 and c878=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
BASE_CTEST=${BASE_CTEST:?BASE_CTEST=<the base ctest table>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-keepalive-why-new}
HEAD_SHA=${HEAD_SHA:-1d177d31e2cb0114ed9805c0462bf75696d79a63}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig ka cpu sass}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c878=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
UIDGID="$(id -u):$(id -g)"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c878=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS"

if step src; then
    S=$W/rig/src
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA" > "$W/commits.txt"
    git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA" >> "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum llama-server test-backend-ops test-llama-archs test-cuda-keepalive llama-bench llama-completion) | tee "$W/rig/bin-md5.txt"
    grep -a "AVX\|GLIBC" "$T/BUILDINFO.txt" 2> /dev/null | head -n 4
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt"), head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step ka; then
    T=$(head_rig)
    K=$W/checks/ka
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        for t in base head; do
            timeout 600 nice -n 19 /$t/bin/test-cuda-keepalive > /k/$t.log 2>&1; echo "$t rc=$?" >> /k/rc.txt
        done
        chown -R '"$UIDGID"' /k'
    cat "$K/rc.txt"
    for t in base head; do echo "$t: $(tail -n 1 "$K/$t.log")"; done
    grep -a "failed:" "$K/head.log" | head -n 20
    echo "=== ka done $(date -Is)"
fi

if step cpu; then
    rm -f "$W/cpu/summary.tsv"
    WORK="$W/cpu" SRC="$W/rig/src" TREES="c878=$HEAD_SHA" STEPS="tree build ctest" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    warnings "$W/cpu/c878/build.log" > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings, head: $(wc -l < "$W/checks/cpu-warnings-head.txt")"
    grep -a "keepalive" "$W/checks/cpu-warnings-head.txt"
    sed -n '/The following tests FAILED:/,$p' "$W/cpu/c878/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
        | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-c878.txt"
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/c878/ctest.log" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-c878.tsv"
    cp "$BASE_CTEST" "$W/checks/ctest-base.tsv"
    echo "ctest, test by test against the base: $(diff -q "$W/checks/ctest-base.tsv" "$W/checks/ctest-c878.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$W/checks/ctest-base.tsv" "$W/checks/ctest-c878.tsv"
    cat "$W/checks/ctest-failed-c878.txt"
    grep -a "test-cuda-keepalive" "$W/cpu/c878/ctest.log" | head -n 3
    echo "=== cpu done $(date -Is)"
fi

if step sass; then
    T=$(head_rig)
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S"
    B=$BASE_RIG/bin/llama-server
    N=$T/bin/llama-server
    dump() { local d; d=$(dirname "$1"); docker run --rm $LBL --cpu-shares=256 --network none -v "$d:$d:ro" "$IMAGE" \
        cuobjdump -sass "$1" | nice -n 19 c++filt; }
    md5sum "$B" "$N" > "$S/md5.txt"
    dump "$B" > "$S/dump-B.txt"
    dump "$N" > "$S/dump-N.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-B.txt" > "$S/h-B.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-N.txt" > "$S/h-N.tsv"
    echo "every kernel's row, base against head: $(cmp -s "$S/h-B.tsv" "$S/h-N.tsv" && echo identical || echo DIFFERENT)" > "$S/tables.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp.identity.tsv" > "$S/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1) b[$1 "\t" $2] = 1; next} FNR > 1 && !(($1 "\t" $2) in b) {print $1}' \
        "$S/h-B.tsv" "$S/h-N.tsv" | sort -u > "$S/only_new.list"
    if [ -s "$S/differ.list" ]; then
        nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-B.txt" > "$S/ext-B.txt"
        nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-N.txt" > "$S/ext-N.txt"
        nice -n 19 python3 -I "$REPO/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
        nice -n 19 python3 -I "$REPO/sass_classify.py" "$S/h-B.tsv" "$S/h-N.tsv" "$S/ext-B.txt" "$S/ext-N.txt" \
            > "$S/classes.tsv" 2> "$S/classes_summary.txt"
    fi
    rm -f "$S/dump-B.txt" "$S/dump-N.txt"
    cat "$S/md5.txt" "$S/tables.txt" "$S/summary.txt"
    [ -f "$S/linediff.txt" ] && cat "$S/linediff.txt" "$S/classes_summary.txt"
    echo "kernels: base $(($(wc -l < "$S/h-B.tsv") - 1)), head $(($(wc -l < "$S/h-N.tsv") - 1)); differing: $(wc -l < "$S/differ.list");"\
        "only in head: $(wc -l < "$S/only_new.list"); missing in head: $(wc -l < "$S/missing.list")"
    cat "$S/differ.list" "$S/only_new.list"
    echo "=== sass done $(date -Is)"
fi

finish 0
