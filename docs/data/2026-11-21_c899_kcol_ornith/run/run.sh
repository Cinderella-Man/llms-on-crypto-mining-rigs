#!/usr/bin/env bash
# C899 (2026-11-21): GGML_CUDA_MMVQ_PASCAL_MT=2 (the read-once kernel only for the products a P106 ran faster on it),
# built and checked on a build machine's CPU against the cumulative branch it was cut from (3dfad876f, C908's build).
# No GPU work: the CUDA build's binaries run against the driver stub, and the GPU code is read with cuobjdump.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   syntax  g++ -fsyntax-only of the new CPU test (fails in seconds rather than after a build)
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs, test-cuda-keepalive
#           and test-cuda-mmvq-pascal-pick added, the web UI archive pinned (HF_UI_VERSION), under LOCK; warnings
#           against BASE_RIG_LOG (the base tarball's build log)
#   unit    the rig build's test-cuda-mmvq-pascal-pick (the dispatch tables' checks and Ornith's kernel per product)
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the head: a CPU build of every target and
#           ctest without network; ctest test by test against BASE_CTEST (the base's, from C908), warnings against
#           BASE_CPU_WARN
#   sass    cuobjdump -sass of both llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/), every differing kernel extracted, compared line by line and
#           classified (sass_classify.py, from docs/data/2026-11-17_c854b_upstream_newest/run/)
# Usage: W=<work dir> SEED=<llama.cpp clone> BASE_RIG=<base rig build folder> BASE_RIG_LOG=<its build log>
#   BASE_CTEST=<base ctest-*.tsv> BASE_CPU_WARN=<base cpu-warnings-*.txt> REPO=<folder with build_llamacpp_docker.sh
#   cpu_tests.sh sass_compare.py sass_classify.py> [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>]
#   [STEPS=...] [JOBS=8] bash run.sh
# Every container: labels pascal-quants=1 and c899=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
export LC_ALL=C  # one sort order for every warning list, the base's included
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
BASE_CTEST=${BASE_CTEST:?BASE_CTEST=<base ctest tsv>}
BASE_CPU_WARN=${BASE_CPU_WARN:?BASE_CPU_WARN=<base cpu warnings>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-kcol-ornith}
HEAD_SHA=${HEAD_SHA:-37c05b76a2664e755c04001957ae527f343f4c0a}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src syntax rig unit cpu sass}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c899=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c899=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
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
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step syntax; then
    K=$W/checks/syntax
    rm -rf "$K" && mkdir -p "$K/src"
    git -C "$W/rig/src" archive "$HEAD_SHA" ggml/include ggml/src/ggml-cuda/mmvq-pascal-pick.h tests/test-cuda-mmvq-pascal-pick.cpp \
        | tar -x -C "$K/src"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$K:/k" "$CPU_IMAGE" bash -c '
        cd /k/src && g++ -std=c++17 -fsyntax-only -Wall -Wextra -Wpedantic -I ggml/include tests/test-cuda-mmvq-pascal-pick.cpp \
            > /k/syntax.log 2>&1; echo "rc=$?" >> /k/syntax.log; chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/syntax.log"
    grep -q '^rc=0$' "$K/syntax.log" || finish 1
    echo "=== syntax done $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive test-cuda-mmvq-pascal-pick" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    ls "$T/bin" | tr '\n' ' '; echo
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt"), head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step unit; then
    T=$(head_rig)
    K=$W/checks/unit
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        /head/bin/test-cuda-mmvq-pascal-pick > /k/unit.txt 2>&1; echo "rc=$?" >> /k/unit.txt
        chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/unit.txt"
    echo "=== unit done $(date -Is)"
fi

if step cpu; then
    rm -f "$W/cpu/summary.tsv"
    WORK="$W/cpu" SRC="$W/rig/src" TREES="c899=$HEAD_SHA" STEPS="tree build ctest" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    # the CPU tree's own folder appears in CMake's warnings: written as <tree> on both sides
    sed -E 's#[^ ]*/cpu/tree/src/#<tree>/#' "$BASE_CPU_WARN" | sort -u > "$W/checks/cpu-warnings-base.txt"
    warnings "$W/cpu/c899/build.log" | sed -E 's#[^ ]*/cpu/tree/src/#<tree>/#' | sort -u > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$W/checks/cpu-warnings-base.txt"), head $(wc -l < "$W/checks/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt"
    cp "$BASE_CTEST" "$W/checks/ctest-base.tsv"
    sed -n '/The following tests FAILED:/,$p' "$W/cpu/c899/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
        | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-c899.txt"
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/c899/ctest.log" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-c899.tsv"
    diff "$W/checks/ctest-base.tsv" "$W/checks/ctest-c899.tsv" > "$W/checks/ctest-diff.txt"
    echo "ctest, test by test against the base (< base, > head):"
    cat "$W/checks/ctest-diff.txt"
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
    # compare matches each base kernel against the new build's kq_fp32-off kernel of that name, so the base side takes
    # its kq_fp32-off kernels only and the kq_fp32=true ones are compared apart (kq.txt), as C854 step two did
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$S/h-B-off.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp.identity.tsv" > "$S/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
        "$S/h-B.tsv" "$S/h-N.tsv" | sort -u > "$S/only_new.list"
    nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-B.txt" > "$S/ext-B.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-N.txt" > "$S/ext-N.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
    echo "kq_fp32=true kernels' SHA-1, base against head: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-B.tsv" | sort) \
        <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)" > "$S/kq.txt"
    nice -n 19 python3 -I "$REPO/sass_classify.py" "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/ext-B.txt" "$S/ext-N.txt" \
        > "$S/classes.tsv" 2> "$S/classes_summary.txt"
    rm -f "$S/dump-B.txt" "$S/dump-N.txt"
    cat "$S/md5.txt" "$S/summary.txt" "$S/kq.txt" "$S/linediff.txt" "$S/classes_summary.txt"
    echo "differing kernels: $(wc -l < "$S/differ.list"); only in head: $(wc -l < "$S/only_new.list"); missing in head: $(wc -l < "$S/missing.list")"
    cat "$S/differ.list" "$S/only_new.list"
    echo "=== sass done $(date -Is)"
fi

finish 0
