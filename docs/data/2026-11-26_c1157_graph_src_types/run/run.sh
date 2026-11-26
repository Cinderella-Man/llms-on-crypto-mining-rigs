#!/usr/bin/env bash
# C1157 (Q1890, 2026-11-26): the CUDA graph cache tells a node's sources of different types apart. ggml-cuda replays a
# captured CUDA graph while every node, and each node's sources' data addresses, shapes and strides, are as they were
# (ggml_cuda_graph_update_required); the sources' types were not compared, so a BF16 matrix at the address of the F16
# matrix just checked replayed the F16 kernels (C1111). Fork branch pascal-p106-v5-graph-src-types: TEST_SHA adds
# tests/test-cuda-graph-src-types.cpp on pascal-p106-v5 (BASE_SHA), HEAD_SHA adds the fix (each source's type in
# node_properties). Built and checked on a build machine with newer NVIDIA cards (RTX 3090, compute 8.6) that run the
# rig build's compute 6.1 PTX through the driver's JIT. Correctness only: nothing here is timed.
# CPU steps (STEPS, in this order):
#   src       a local clone of SEED (a llama.cpp clone that holds the base), the fork as remote "fork", the branch
#             fetched and checked against HEAD_SHA (and TEST_SHA its parent), its commits and diff listed; the compiler
#             cache hard-linked from RIG_CCACHE
#   rig       the branch's rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 (test-backend-ops and every
#             check tool, test-cuda-graph-src-types among them), the web page pinned (HF_UI_VERSION), under LOCK; its md5
#             and its binaries' md5s against the base build's; warnings against BASE_LOG (the base's build log)
#   testonly  the test without the fix: TEST_SHA built the same way with CHECK_TOOLS=0 and EXTRA_TARGETS=
#             test-cuda-graph-src-types; only that binary is kept ($W/testonly/bin), the rest of the build is deleted
#   c1111fix  the fix on C1111's failing build: C1111_SHA (pascal-p106-v5-cand-b12674) with the fix commit's diff as a
#             patch, TESTS=1 CHECK_TOOLS=0, in its own clone ($W/rigc) with its own compiler cache (C1111_CCACHE, one
#             that holds that tree's objects); only test-backend-ops is kept ($W/c1111fix/bin)
#   sass      cuobjdump -sass of the base's llama-server (BASE) and the branch's: one SHA-1 per kernel copy
#             (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/), the names whose copies differ, and for
#             those whether only constants differ (sass_classify.py, from docs/data/2026-11-24_c912_upstream_b12674/run/)
# GPU steps (GPU_STEPS; lane 0 on the first card of GPUS, lane 1 on the second, side by side, then the two-card steps):
#   test      lane 0: test-cuda-graph-src-types, N_TEST runs per build (testonly: the base with the test; branch) and
#             setting (graphs: the default; nographs: GGML_CUDA_DISABLE_GRAPHS=1; forced: the fork's compute-6.1-only
#             paths forced at a P106's shapes, scripts/pascal_forced.sh's PASCAL_FORCED), with GGML_CUDA_GRAPH_DEBUG=1
#             (each warm-up, capture and reset logged)
#   again     lane 0: C1111's reproduction, docs/data/2026-11-26_c1111_bf16_intermittent/run/c1111.sh again: its
#             12-case F16/BF16 sibling set (sib), every case checked live (--cpu-check 1), N_AGAIN runs per build, with
#             CUDA graphs and with GGML_CUDA_DISABLE_GRAPHS=1, on the builds AGAIN_BUILDS names: c1111base (C1111's
#             base, C1111_BASE, where it fails), base, branch, c1111fix (C1111's base with the fix, from the c1111fix
#             step); default "c1111base base branch"
#   mm        lane 0: MUL_MAT_ID and MUL_MAT replayed from the kept record REC_MM (docs/data/2026-11-24_c1023_suite_record/
#             run/run.sh, STEPS=replay, ALLOW_MISSING=1) on the base and the branch, the fork's matrix-vector kernels
#             forced, then the base again with --cpu-check 25
#   fa        lane 1: FLASH_ATTN_EXT (no switch, the Pascal tile kernel, every Pascal attention switch) and the fused
#             cases (PASCAL_FORCED, PASCAL_FORCED_MMV) replayed from the kept records REC_FA
#             (docs/data/2026-11-26_c1042_suite_records/run/run.sh, STEPS=replay, ALLOW_MISSING=1) on both builds, then
#             the cpu-check
#   pipe      both cards: bench/pipespec/gpu_pipe_check.sh on the base's and the branch's bin/ with its default arms (the
#             random models pipelined against plain on one card and split over the two)
#   flash     both cards: Qwen3.8-Flash-Next (FN, with the MTP draft DRAFT) with its v5 preset
#             (release/flash-next-v5/settings.env), the layer split changed to the two cards (-ts 1,1), the fork's
#             compute-6.1-only paths forced as on a P106 (pascal_force_preset): one llama-server per build (branch,
#             then base), two greedy requests (/completion, temperature 0, top-k 1, 256 tokens, no prompt cache, token
#             ids returned: C854b's coding request and prose paragraph); token ids and draft counts compared
#   tables    run/tables.py over the outputs, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the base> BASE=<the base's rig build folder> BASE_LOG=<its build log>
#   C1111_BASE=<C1111's base build folder> REPO=<a copy of this repository> REC_MM=<C1023's record folder>
#   REC_FA=<C1042's record folder> FN=<Flash-Next shard 1> DRAFT=<its MTP draft> [RIG_CCACHE=<dir>]
#   [C1111_SHA=<its commit>] [C1111_CCACHE=<dir>] [AGAIN_BUILDS=...] [LOCK=<flock file>] [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <ids>">] [MOUNTS='<folders the containers mount
#   read-only>'] [GW=<work dir of the GPU steps, default W>] [JIT_SEED=<a PTX JIT cache of the base's kernels to copy
#   into GW/jitcache and GW/mm/jitcache>] [JIT_SEED_AGAIN=<one for GW/again/jit>] [JIT_SEED_FA=<one for
#   GW/fa/jitcache>] [STEPS=...] [GPU_STEPS=...] [GPUS=0,1] [JOBS=8] [N_TEST=3] [N_AGAIN=20] [PORT=18957]
#   [NAME=c1157] [ROW=c1157] bash run.sh
# Every container: labels pascal-quants=1 and $ROW=1, --cpu-shares=256; the named ones start with $NAME-, and at the
# end this invocation stops only those (give a second invocation beside a running one its own NAME). Done when
# $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
GW=${GW:-$W}
SEED=${SEED:?SEED=<llama.cpp clone with the base>}
BASE=${BASE:?BASE=<the base rig build folder>}
BASE_LOG=${BASE_LOG:?BASE_LOG=<its build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-graph-src-types}
HEAD_SHA=${HEAD_SHA:-9862e27344a4e5f91f1fc4294db42edf452fb9c7}
TEST_SHA=${TEST_SHA:-582cfbfe3204ab35f16ef22fb3f40e3bb53879f3}
BASE_SHA=${BASE_SHA:-2d839da1d8b7ae1dc0098acb8d4e0897d06757e7}
C1111_SHA=${C1111_SHA:-eee615cdb9d26c917de01e3c49f794c41de31684}
AGAIN_BUILDS=${AGAIN_BUILDS:-c1111base base branch}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12674}
STEPS=${STEPS:-src rig testonly sass}
GPU_STEPS=${GPU_STEPS:-test again mm fa pipe flash tables}
GPUS=${GPUS:-0,1}
JOBS=${JOBS:-8}
N_TEST=${N_TEST:-3}
N_AGAIN=${N_AGAIN:-20}
PORT=${PORT:-18957}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1200}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c1157}
ROW=${ROW:-c1157}
D=$REPO/docs/data/2026-11-26_c1157_graph_src_types/run
SASS_PY=$REPO/docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py
CLASSIFY_PY=$REPO/docs/data/2026-11-24_c912_upstream_b12674/run/sass_classify.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $ROW=1"
MEM="--memory=16g --memory-swap=16g"
IFS=, read -ra G <<< "$GPUS"
G0=${G[0]}
G1=${G[1]:-${G[0]}}
VOL=(-v "$W:$W" -v "$GW:$GW" -v "$BASE:$BASE:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
mkdir -p "$W/rig" "$W/logs" "$W/checks" "$W/tables" "$GW/jitcache"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
gstep()  { [[ " $GPU_STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$ROW=1" --filter "name=^$NAME-" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
export LC_ALL=C
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
build_dir() { ls -d "$1"/llama.cpp-*-"${2:0:9}"* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# a clone of SEED at $1 with upstream as origin and the fork as "fork" (build_llamacpp_docker.sh fetches both)
clone_seed() {
    local S=$1
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || return 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" fetch -q "$SEED" 'refs/tags/*:refs/tags/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
}
echo "=== start $(date -Is): $REF ($HEAD_SHA, test $TEST_SHA) on $BASE_SHA; steps: $STEPS; GPU steps: $GPU_STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    clone_seed "$S" || finish 1
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; parent $(git -C "$S" rev-parse "$got^"); grandparent $(git -C "$S" rev-parse "$got^^")"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    [ "$(git -C "$S" rev-parse "$got^")" = "$TEST_SHA" ] || { echo "the test commit is not the branch's parent: stop"; finish 1; }
    [ "$(git -C "$S" rev-parse "$got^^")" = "$BASE_SHA" ] || { echo "the base is not the test's parent: stop"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    git -C "$S" diff "$TEST_SHA" "$HEAD_SHA" > "$W/fix.diff"
    cat "$W/commits.txt"
    # hard links on the same disk, else a copy
    if [ ! -d "$W/rig/ccache" ] && [ -n "${RIG_CCACHE:-}" ]; then
        cp -al "$RIG_CCACHE" "$W/rig/ccache" 2> /dev/null || { rm -rf "$W/rig/ccache"; cp -a "$RIG_CCACHE" "$W/rig/ccache"; }
    fi
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$BASE/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), base $(wc -l < "$W/rig/bin-md5-base.txt");"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-base.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep '^[<>]' | tr '\n' ' ')"
    grep -aE '^(name|commit|web page|avx|highest glibc|check tools):' "$T/BUILDINFO.txt"
    warnings "$BASE_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step testonly; then
    # the same clone, the test commit alone; build_llamacpp_docker.sh cleans the tree (git clean -fdx) first
    locked env WORK="$W/rig" TESTS=1 CHECK_TOOLS=0 EXTRA_TARGETS=test-cuda-graph-src-types JOBS="$JOBS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$TEST_SHA" > "$W/logs/testonly.log" 2>&1
    echo "testonly rc=$?"
    X=$(build_dir "$W/rig" "$TEST_SHA")
    [ -n "$X" ] && [ -x "$X/bin/test-cuda-graph-src-types" ] || { echo "no test-only build"; finish 1; }
    mkdir -p "$W/testonly/bin"
    cp "$X/bin/test-cuda-graph-src-types" "$X/BUILDINFO.txt" "$W/testonly/bin/"
    (cd "$W/testonly/bin" && md5sum test-cuda-graph-src-types) | tee "$W/testonly/md5.txt"
    grep -aE '^(name|commit)' "$X/BUILDINFO.txt"
    rm -rf "$X" "$X.tar.gz" "$W/rig/src/build"
    echo "=== testonly done $(date -Is)"
fi

if step c1111fix; then
    S=$W/rigc/src
    mkdir -p "$W/rigc"
    clone_seed "$S" || finish 1
    git -C "$S" cat-file -e "$C1111_SHA^{commit}" || { echo "no $C1111_SHA in the clone: stop"; finish 1; }
    [ -s "$W/fix.diff" ] || { echo "no fix.diff (the src step writes it)"; finish 1; }
    cp "$W/fix.diff" "$W/rigc/fix.diff"
    if [ ! -d "$W/rigc/ccache" ] && [ -n "${C1111_CCACHE:-}" ]; then
        cp -al "$C1111_CCACHE" "$W/rigc/ccache" 2> /dev/null || { rm -rf "$W/rigc/ccache"; cp -a "$C1111_CCACHE" "$W/rigc/ccache"; }
    fi
    locked env WORK="$W/rigc" TESTS=1 CHECK_TOOLS=0 PATCHES=fix.diff JOBS="$JOBS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$C1111_SHA" > "$W/logs/c1111fix.log" 2>&1
    echo "c1111fix rc=$?"
    X=$(build_dir "$W/rigc" "$C1111_SHA")
    [ -n "$X" ] && [ -x "$X/bin/test-backend-ops" ] || { echo "no c1111fix build"; finish 1; }
    mkdir -p "$W/c1111fix/bin"
    cp "$X/bin/test-backend-ops" "$X/BUILDINFO.txt" "$W/c1111fix/bin/"
    (cd "$W/c1111fix/bin" && md5sum test-backend-ops) | tee "$W/c1111fix/md5.txt"
    grep -aE '^(name|commit|patch)' "$X/BUILDINFO.txt"
    rm -rf "$X" "$X.tar.gz" "$S/build"
    echo "=== c1111fix done $(date -Is)"
fi

if step sass; then
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S/bin"
    H=$(build_dir "$W/rig" "$HEAD_SHA")
    for x in B H; do
        case $x in B) b=$BASE/bin/llama-server ;; H) b=$H/bin/llama-server ;; esac
        ln -f "$b" "$S/bin/$x-llama-server" 2> /dev/null || cp "$b" "$S/bin/$x-llama-server"
        echo "$x $(md5sum < "$S/bin/$x-llama-server" | cut -c1-32) $b" >> "$S/md5.txt"
        docker run --rm $LBL --cpu-shares=256 --network none -v "$S/bin:$S/bin:ro" "$IMAGE" cuobjdump -sass "$S/bin/$x-llama-server" \
            | nice -n 19 c++filt > "$S/dump-$x.txt"
        nice -n 19 python3 -I "$SASS_PY" hash < "$S/dump-$x.txt" > "$S/h-$x.tsv"
        say "sass $x: $(($(wc -l < "$S/h-$x.tsv") - 1)) kernel copies"
    done
    O=$S/BH; mkdir -p "$O"
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$O/h-B-off.tsv"
    nice -n 19 python3 -I "$SASS_PY" compare "$O/h-B-off.tsv" "$S/h-H.tsv" "$O/cmp" > "$O/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$O/cmp.identity.tsv" > "$O/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$O/cmp.identity.tsv" > "$O/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
        "$S/h-B.tsv" "$S/h-H.tsv" | sort -u > "$O/only_new.list"
    if [ -s "$O/differ.list" ]; then
        nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-B.txt" > "$O/ext-B.txt"
        nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-H.txt" > "$O/ext-N.txt"
        nice -n 19 python3 -I "$CLASSIFY_PY" "$O/h-B-off.tsv" "$S/h-H.tsv" "$O/ext-B.txt" "$O/ext-N.txt" \
            > "$O/classes.tsv" 2> "$O/classes_summary.txt"
    fi
    {
        echo "--- base against branch: $(head -n 1 "$O/summary.txt")"
        echo "    names whose copies differ $(wc -l < "$O/differ.list"), missing $(wc -l < "$O/missing.list"), only in branch $(wc -l < "$O/only_new.list")"
        [ -f "$O/classes_summary.txt" ] && sed 's/^/    /' "$O/classes_summary.txt" | head -n 12
    } | tee "$S/pairs.txt"
    rm -f "$O/ext-B.txt" "$O/ext-N.txt" "$S"/dump-?.txt "$S"/bin/?-llama-server
    echo "=== sass done $(date -Is)"
fi

# ---- GPU steps -------------------------------------------------------------------------------------------------------
[ "$GPU_STEPS" = none ] && finish 0
T=$(build_dir "$W/rig" "$HEAD_SHA")
[ -n "$T" ] || { echo "no branch build"; finish 1; }
# shellcheck source=../../../../scripts/pascal_forced.sh
. "$TOOLS/pascal_forced.sh"
DR=(docker run --rm $LBL --cpu-shares=256 "${VOL[@]}" -e "CUDA_CACHE_PATH=$GW/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# the PTX JIT's caches: each sub-driver keeps its own; a copy of one made with the base's kernels saves compiling them
# again, since a change of host code leaves the branch's kernels the base's
seed_jit() {  # <seed> <cache folder>
    mkdir -p "$2"
    if [ -n "$1" ] && [ -d "$1" ] && [ -z "$(ls -A "$2")" ]; then cp -a "$1/." "$2/"; say "JIT cache $2 seeded from $1"; fi
}
if [ -n "$PREFLIGHT" ] && [ "$GPU_STEPS" != tables ]; then
    $PREFLIGHT -g "$GPUS" > "$W/checks/preflight.txt" 2>&1; rc=$?; say "preflight on cards $GPUS rc=$rc"
    [ $rc = 0 ] || finish 1
fi

lane0() {
    seed_jit "${JIT_SEED:-}" "$GW/jitcache"
    if gstep test; then
        local o=$GW/test arm bin env i f
        mkdir -p "$o"
        printf 'build\tsetting\trun\trc\tright\tevaluations\tverdict\twarmups_complete\twarmups_reset\n' > "$o/test.tsv"
        for b in testonly branch; do
            bin=$W/testonly/bin; [ "$b" = branch ] && bin=$T/bin
            for arm in graphs nographs forced; do
                env=""
                [ "$arm" = nographs ] && env=GGML_CUDA_DISABLE_GRAPHS=1
                [ "$arm" = forced ] && env=$PASCAL_FORCED
                for i in $(seq 1 "$N_TEST"); do
                    f=$o/$b-$arm-$i.log
                    "${DR[@]}" --name "$NAME-test" --gpus "\"device=$G0\"" "$IMAGE" \
                        bash -c "env $env GGML_CUDA_GRAPH_DEBUG=1 timeout 300 $bin/test-cuda-graph-src-types > $f 2>&1; echo rc=\$? >> $f" > /dev/null 2>&1
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$b" "$arm" "$i" "$(grep -a -o 'rc=[0-9]*' "$f" | tail -n 1 | cut -d= -f2)" \
                        "$(grep -a -oE '^[0-9]+ of [0-9]+ evaluations right' "$f" | cut -d' ' -f1)" \
                        "$(grep -a -oE '^[0-9]+ of [0-9]+ evaluations right' "$f" | cut -d' ' -f3)" \
                        "$(grep -a -oE '(PASS|FAIL)$' "$f" | tail -n 1)" \
                        "$(grep -a -c 'warmup complete' "$f")" "$(grep -a -c 'warmup reset' "$f")" >> "$o/test.tsv"
                done
                say "test $b $arm: $(awk -F'\t' -v b="$b" -v a="$arm" '$1==b && $2==a {n++; if ($7=="PASS") p++} END {print p+0 " of " n+0 " runs pass"}' "$o/test.tsv")"
            done
        done
    fi
    if gstep again; then
        local B3="" a
        for a in $AGAIN_BUILDS; do
            case $a in
                c1111base) B3+="c1111base=${C1111_BASE:?}/bin " ;;
                base)      B3+="base=$BASE/bin " ;;
                branch)    B3+="branch=$T/bin " ;;
                c1111fix)  B3+="c1111fix=$W/c1111fix/bin " ;;
            esac
        done
        local A=(env W="$GW/again" REC="${REC_MM:?}/mul_mat" MOUNTS="${MOUNTS:-}" NAME="$NAME-again" GPU="$G0"
            DOCKER_ARGS="$LBL --cpu-shares=256" TBO_TIMEOUT=600 AGAIN_SET=sib AGAIN_CC=1 N_AGAIN="$N_AGAIN" BUILDS="$B3")
        seed_jit "${JIT_SEED_AGAIN:-}" "$GW/again/jit"
        "${A[@]}" AGAIN_TAG=graphs bash "$REPO/docs/data/2026-11-26_c1111_bf16_intermittent/run/c1111.sh" again
        "${A[@]}" AGAIN_TAG=nographs AGAIN_ENV=GGML_CUDA_DISABLE_GRAPHS=1 \
            bash "$REPO/docs/data/2026-11-26_c1111_bf16_intermittent/run/c1111.sh" again
    fi
    if gstep mm; then
        seed_jit "${JIT_SEED:-}" "$GW/mm/jitcache"
        env WORK="$GW/mm" NEW="$T/bin" REC="${REC_MM:?}" BUILDS="branch=$T/bin base=$BASE/bin" STEPS=replay ALLOW_MISSING=1 \
            CPU_CHECK=25 GPU="$G0" NAME="$NAME-mm" MOUNTS="${MOUNTS:-}" DOCKER_ARGS="$LBL --cpu-shares=256" \
            bash "$REPO/docs/data/2026-11-24_c1023_suite_record/run/run.sh" > "$W/logs/mm.log" 2>&1
        say "mm rc=$?"
        grep -v '^    ' "$GW/mm/bops/summary.txt" 2>/dev/null
    fi
}

lane1() {
    if gstep fa; then
        seed_jit "${JIT_SEED_FA:-}" "$GW/fa/jitcache"
        env WORK="$GW/fa" NEW="$T/bin" REC="${REC_FA:?}" BUILDS="branch=$T/bin base=$BASE/bin" STEPS=replay ALLOW_MISSING=1 \
            CPU_CHECK=25 GPUS="$G1" NAME="$NAME-fa" MOUNTS="${MOUNTS:-}" DOCKER_ARGS="$LBL --cpu-shares=256" \
            PASCAL_FORCED_SH="$TOOLS/pascal_forced.sh" \
            bash "$REPO/docs/data/2026-11-26_c1042_suite_records/run/run.sh" > "$W/logs/fa.log" 2>&1
        say "fa rc=$?"
        grep -v '^    ' "$GW/fa/bops/summary.txt" 2>/dev/null
    fi
}

if gstep test || gstep again || gstep mm || gstep fa; then
    lane0 > "$W/logs/lane0.log" 2>&1 &
    p0=$!
    lane1 > "$W/logs/lane1.log" 2>&1 &
    p1=$!
    wait "$p0" "$p1"
    cat "$W/logs/lane0.log" "$W/logs/lane1.log"
    echo "=== lanes done $(date -Is)"
fi

if gstep pipe; then
    env WORK="$GW/pipe" BUILDS="base=$BASE/bin branch=$T/bin" GPUS="$G0,$G1" MOUNTS="${MOUNTS:-}" \
        DOCKER_ARGS="$LBL --cpu-shares=256" PREFLIGHT="$PREFLIGHT" JITCACHE="$GW/jitcache" NAME="$NAME-pipe" \
        bash "$REPO/bench/pipespec/gpu_pipe_check.sh" > "$W/logs/pipe.log" 2>&1
    say "pipe rc=$?"
    cut -f1-6 "$GW/pipe/summary.tsv" 2>/dev/null
    echo "=== pipe done $(date -Is)"
fi

if gstep flash; then
    : "${FN:?}" "${DRAFT:?}"
    P_CODE='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
    P_PROSE='Explain in one paragraph how a hash map handles collisions, then give one example in Python.'
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1091
    . "$REPO/release/flash-next-v5/settings.env"
    M_ENV="$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")"
    M_ARGS=$(sed -E "s/-ts [0-9,]+/-ts 1,1/" <<< "$PRESET_ARGS")
    [[ "$M_ARGS" == *"-ts 1,1 "* ]] || { say "flash: no -ts in the preset to change"; finish 1; }
    load() {  # <label> <bin folder>
        local o=$GW/flash/$1 c=$NAME-flash-$1 up=0 t0 p body
        mkdir -p "$o"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'env: %s\nargs: -m %s -md %s %s\nbin: %s\n' "$M_ENV" "$FN" "$DRAFT" "$M_ARGS" "$2" > "$o/command.txt"
        md5sum "$2/llama-server" >> "$o/command.txt"
        # shellcheck disable=SC2086
        "${DR[@]}" -d --name "$c" --gpus "\"device=$G0,$G1\"" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" \
            bash -c "env $M_ENV $2/llama-server -m $FN -md $DRAFT $M_ARGS --host 0.0.0.0 --port $PORT --metrics --no-webui \
                > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
        if [ $up = 1 ]; then
            sleep 20
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$G0,$G1" > "$o/mem.csv"
            for p in code prose; do
                if [ $p = code ]; then body=$P_CODE; else body=$P_PROSE; fi
                printf '{"prompt": "%s", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false, "return_tokens": true}' "$body" |
                    curl -s -m 900 -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/completion" \
                    > "$o/reply-$p.json" 2> "$o/reply-$p.err"
            done
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$G0,$G1" > "$o/mem-after.csv"
        fi
        docker stop -t 60 "$c" > /dev/null 2>&1
        docker rm -f "$c" > /dev/null 2>&1
        say "flash $1: $(cat "$o/load.txt")"
    }
    [ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$G0,$G1" > "$W/checks/flash-preflight.txt" 2>&1; say "flash preflight rc=$?"; }
    md5sum "$FN" "$DRAFT" > "$W/checks/flash-md5.txt"
    load branch "$T/bin"
    load base "$BASE/bin"
    echo "=== flash done $(date -Is)"
fi

if gstep tables; then
    nice -n 19 python3 -I "$D/tables.py" "$W" "$GW" "$W/tables" | tee "$W/tables/summary.txt"
    echo "=== tables done $(date -Is)"
fi
finish 0
