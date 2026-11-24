#!/usr/bin/env bash
# C880 landing (2026-11-24): the batch-API branch (pascal-p106-v5-batch-api, the fork's own tools and tests on upstream's
# new batch calls) merged into Miner Llama's cumulative branch pascal-p106-v5, built and checked on a build machine
# against the cumulative branch's head before the merge. Processor only, no GPU work: the merge changes no engine code.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the merge
#           branch fetched and checked against HEAD_SHA, its diff against the base listed; compiler caches hard-linked
#           from RIG_CCACHE and CPU_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs and
#           test-cuda-keepalive added and the web page pinned (HF_UI_VERSION), as BASE_RIG's build, under LOCK; its
#           md5 and its binaries' md5s; warnings against BASE_RIG_LOG (the base tarball's build log)
#   archs   test-llama-archs -b CPU -s 1 of the head's rig build against the logits the base's rig build (BASE_RIG)
#           writes in the same run: every file bit for bit
#   sass    the card code: cuobjdump -sass and -ptx of both rig builds' llama-server; the two dumps compared whole,
#           and one SHA-1 per kernel (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/) compared
#   base    cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the base, its build kept: a CPU build of every
#           target, ctest without network, test-llama-archs (plain, write, compare), the hand-run tests
#   models  with the base's test-llama-archs, seed 1: a random qwen4exp MoE model (4,096-token vocabulary) and the
#           same with 32 experts, 8 used; add_screen_head.py adds a screened head to the first; prompts.txt, three
#           prompts for llama-screen-check (all as docs/data/2026-11-21_c880_batch_api/run/run.sh)
#   tbase   tools.sh (docs/data/2026-11-21_c880_batch_api/run/) with the base's binaries into $W/tools/base
#   head    cpu_tests.sh on the head, ident against the base's logits files, its build kept
#   thead   tools.sh with the head's binaries into $W/tools/head
#   compare the two CPU builds' warnings, their ctest results test by test (and against BASE_CTEST, a ctest table of
#           the base's engine from an earlier run, if given), the hand-run tests' verdicts (their random models are new
#           on every run), and compare.py (docs/data/2026-11-21_c880_batch_api/run/) on the two tools.sh outputs
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> BASE_RIG=<base rig build folder>
#   BASE_RIG_LOG=<its build log> REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py tools.sh
#   add_screen_head.py compare.py and pipespec/ (bench/pipespec)> [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>]
#   [BASE_CTEST=<ctest table>] [LOCK=<flock file>] [STEPS=...] [JOBS=8] [THREADS=6] bash run.sh
# Every container: labels pascal-quants=1 and c880land=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-land-batch-api}
HEAD_SHA=${HEAD_SHA:-8a9629942a4635ea0b2711b72e59556b96a63f86}
BASE_SHA=${BASE_SHA:-14d67960410bd3d6ff1f9ac819ce69c9c6f9396f}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig archs sass base models tbase head thead compare}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c880land=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
mkdir -p "$W/cpu" "$W/rig" "$W/logs" "$W/checks" "$W/tools"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c880land=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
export LC_ALL=C # sort and comm in one collation
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
deprecated() { grep -c "is deprecated: use llama_" "$1"; }
cbox() { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:$REPO:ro" "$CPU_IMAGE" "$@"; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
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
    git -C "$S" remote set-url fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    echo "engine files changed (src ggml include common tools): $(git -C "$S" diff --name-only "$BASE_SHA" "$HEAD_SHA" -- src ggml include common tools | wc -l)"
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
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    ls "$T/bin" | tr '\n' ' '; echo
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$BASE_RIG/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "binaries, head against base (the build-info commit string differs in every binary that links common):"
    join -j 2 <(sort -k2 "$W/rig/bin-md5-base.txt") <(sort -k2 "$W/rig/bin-md5-head.txt") \
        | awk '{print $1 "\t" ($2 == $3 ? "same" : "differs")}' | tee "$W/checks/bin-md5.tsv"
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines ($(deprecated "$W/checks/rig-warnings-base.txt") deprecated batch API),"\
        "head $(wc -l < "$W/checks/rig-warnings-head.txt") ($(deprecated "$W/checks/rig-warnings-head.txt")),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "--- only in base:"; comm -23 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step archs; then
    T=$(head_rig)
    K=$W/checks/archs
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        run() { env $3 nice -n 19 timeout 3600 $2/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}" >> /k/rc.txt; }
        run head-plain /head/bin ""
        run base-write /base/bin LLAMA_ARCHS_LOGITS=/k/logits-
        ls /k/logits-*.f32 | wc -l > /k/n_files.txt
        run head-compare /head/bin LLAMA_ARCHS_LOGITS=/k/logits-
        rm -f /k/logits-*.f32
        chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/rc.txt"
    echo "files written by base: $(cat "$K/n_files.txt"); compared: $(grep -ac 'logits against' "$K/head-compare.log");"\
        "differ: $(grep -a 'logits against' "$K/head-compare.log" | grep -vc ', 0 of ')"
    for f in head-plain base-write head-compare; do echo "$f: $(grep -a -m 1 '^Summary' "$K/$f.log")"; done
    echo "=== archs done $(date -Is)"
fi

if step sass; then
    T=$(head_rig)
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S"
    cuo() { local d; d=$(dirname "$2"); docker run --rm $LBL --cpu-shares=256 --network none -v "$d:$d:ro" "$IMAGE" cuobjdump "$1" "$2"; }
    for x in B N; do b=$BASE_RIG/bin/llama-server; [ $x = N ] && b=$T/bin/llama-server
        md5sum "$b" >> "$S/md5.txt"
        cuo -sass "$b" | nice -n 19 c++filt > "$S/dump-$x.txt"
        cuo -ptx "$b" > "$S/ptx-$x.txt"
        nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-$x.txt" > "$S/h-$x.tsv"
    done
    {
        echo "SASS dump: base $(wc -l < "$S/dump-B.txt") lines, head $(wc -l < "$S/dump-N.txt");"\
            "whole dumps $(cmp -s "$S/dump-B.txt" "$S/dump-N.txt" && echo identical || echo DIFFERENT)"
        echo "PTX dump: base $(wc -l < "$S/ptx-B.txt") lines, head $(wc -l < "$S/ptx-N.txt");"\
            "whole dumps $(cmp -s "$S/ptx-B.txt" "$S/ptx-N.txt" && echo identical || echo DIFFERENT)"
        echo "kernel copies hashed: base $(($(wc -l < "$S/h-B.tsv") - 1)), head $(($(wc -l < "$S/h-N.tsv") - 1));"\
            "distinct names: base $(tail -n +2 "$S/h-B.tsv" | cut -f1,2 | sort -u | wc -l), head $(tail -n +2 "$S/h-N.tsv" | cut -f1,2 | sort -u | wc -l)"
        echo "per-kernel SHA-1 tables, sorted: $(diff -q <(sort "$S/h-B.tsv") <(sort "$S/h-N.tsv") > /dev/null && echo identical || echo DIFFERENT)"
    } | tee "$S/summary.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B.tsv" "$S/h-N.tsv" "$S/cmp" | tee -a "$S/summary.txt"
    diff <(sort "$S/h-B.tsv") <(sort "$S/h-N.tsv") | head -n 40
    rm -f "$S/dump-B.txt" "$S/dump-N.txt" "$S/ptx-B.txt" "$S/ptx-N.txt"
    echo "=== sass done $(date -Is)"
fi

cpu_tests() { # <trees> <steps> <log>
    WORK="$W/cpu" SRC="$W/rig/src" TREES="$1" STEPS="$2" JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" \
        DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$3" 2>&1
}

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
    tr '\n' ' ' < "$W/tools/$1/rc.tsv"; echo
}

if step tbase; then
    grep -q "commit $BASE_SHA" "$W/cpu/base/tree.txt" || { echo "the build tree is not the base's"; finish 1; }
    tools base
    echo "=== tbase done $(date -Is)"
fi

if step head; then
    cpu_tests "land=$HEAD_SHA@base" "tree build ctest archs ident hand" "$W/logs/cpu-head.log"
    echo "head cpu tests rc=$?"
    grep -P '^land\t' "$W/cpu/summary.tsv"
    echo "=== head done $(date -Is)"
fi

if step thead; then
    grep -q "commit $HEAD_SHA" "$W/cpu/land/tree.txt" || { echo "the build tree is not the head's"; finish 1; }
    tools head
    echo "=== thead done $(date -Is)"
fi

if step compare; then
    K=$W/checks
    warnings "$W/cpu/base/build.log" > "$K/cpu-warnings-base.txt"
    warnings "$W/cpu/land/build.log" > "$K/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$K/cpu-warnings-base.txt") lines ($(deprecated "$K/cpu-warnings-base.txt") deprecated batch API),"\
        "head $(wc -l < "$K/cpu-warnings-head.txt") ($(deprecated "$K/cpu-warnings-head.txt")), only in head: $(comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt"
    for t in base land; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$K/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$K/ctest-$t.tsv"
    done
    echo "ctest: base $(wc -l < "$K/ctest-base.tsv") tests ($(grep -c 'Passed$' "$K/ctest-base.tsv") passed), head $(wc -l < "$K/ctest-land.tsv") ($(grep -c 'Passed$' "$K/ctest-land.tsv") passed)"
    echo "ctest, test by test, head against base: $(diff -q "$K/ctest-base.tsv" "$K/ctest-land.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$K/ctest-base.tsv" "$K/ctest-land.tsv"
    if [ -n "${BASE_CTEST:-}" ]; then
        echo "ctest, test by test, head against $BASE_CTEST: $(diff -q "$BASE_CTEST" "$K/ctest-land.tsv" > /dev/null && echo same || echo DIFFERENT)"
        diff "$BASE_CTEST" "$K/ctest-land.tsv"
    fi
    echo "failed on head:"; cat "$K/ctest-failed-land.txt"
    # the hand-run tests' random models differ per run, so their verdicts are compared, not their logs
    grep -P '^(base|land)\thand ' "$W/cpu/summary.tsv" | cut -f1-3 | sort -t$'\t' -k2,2 -k1,1 > "$K/hand-verdicts.tsv"
    echo "hand-run tests, verdicts: $(diff -q <(grep -P '^base\t' "$K/hand-verdicts.tsv" | cut -f2-3) \
        <(grep -P '^land\t' "$K/hand-verdicts.tsv" | cut -f2-3) > /dev/null && echo same || echo DIFFERENT)"
    cbox python3 -I "$REPO/compare.py" "$W/tools/base" "$W/tools/head" "$K/tools.tsv"
    echo "compare rc=$?"
    echo "=== compare done $(date -Is)"
fi
finish 0
