#!/usr/bin/env bash
# C993 (Q1726, 2026-11-24): a size for the Pascal prompt kernel's fault switch. Fork branch pascal-p106-v5-fault-scale
# (two commits on pascal-p106-v5, ggml/src/ggml-cuda/fattn-tile-pascal.cu and .cuh) adds
# GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=<factor>: the factor GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 multiplies the
# compute-6.1 prompt and decode attention kernels' output by (default 1.05, the fixed factor before). Built and checked
# on a build machine with newer NVIDIA cards (compute 8.6), which run the rig build's compute 6.1 PTX through the
# driver's JIT. Correctness only: nothing here is timed.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base), the fork as remote "fork", the branch fetched
#           and checked against HEAD_SHA, its diff against the base listed; the compiler cache hard-linked from
#           RIG_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and EXTRA_TARGETS, the base's web page
#           (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s against the base's; warnings against BASE_LOG
#   sass    cuobjdump -sass of the base's llama-server (BASE/bin) and the head's: one SHA-1 per kernel copy
#           (sass_compare.py hash, from docs/data/2026-11-17_c861a_fp32_scale/run/), each kernel (name and kq_fp32 flag)
#           against the base's (c993_tables.py hashcmp), and for those that differ every instruction that differs
#           (c993_tables.py sassdiff)
#   check   on one card (GPU, after PREFLIGHT), test-backend-ops -o FLASH_ATTN_EXT on the head-256 cases (FILTER) with the
#           fork's prompt kernel forced on any card (GGML_CUDA_FATTN_TILE_PASCAL=2):
#             record  the cases' inputs and CPU references, once, with the base build and no fault (the record step's
#                     own live check)
#             arms    the record replayed with each arm's binary and switches (ARMS below), --replay-log: per case the
#                     verdict and its error (NMSE); two arms add --cpu-check 1 (a live CPU check of every case first)
#   tables  run/c993_tables.py over the replay logs and the SASS, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the base> BASE=<the base's rig build folder> BASE_LOG=<its build log>
#   REPO=<a copy of this repository> [RIG_CCACHE=<dir>] [LOCK=<flock file>] [PREFLIGHT=<cuda check, run as
#   "$PREFLIGHT -g <id>">] [STEPS=...] [JOBS=8] [GPU=2] [JIT_SEED=<a PTX JIT cache to copy>] [NAME=c993] bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the cuobjdump containers without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base>}
BASE=${BASE:?BASE=<the base rig build folder>}
BASE_LOG=${BASE_LOG:?BASE_LOG=<its build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-fault-scale}
HEAD_SHA=${HEAD_SHA:-ceaf710a13a13259ad09a5156d331cdf9f7b8989}
BASE_REF=${BASE_REF:-fork/pascal-p106-v5}
BASE_SHA=${BASE_SHA:-af00bb13b0fd59612bcab545964b61978f04b8d6}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-top-k-sort}
STEPS=${STEPS:-src rig sass check tables}
JOBS=${JOBS:-8}
GPU=${GPU:-2}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c993}
FILTER=${FILTER:-^hsk=256,hsv=256,}
D=$REPO/docs/data/2026-11-24_c993_fault_scale/run
TOOLS=$REPO/scripts
SASS_PY=$REPO/docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
TILE=GGML_CUDA_FATTN_TILE_PASCAL=2
F=GGML_CUDA_FATTN_TILE_PASCAL_FAULT
S=GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE
DEC="GGML_CUDA_FATTN_DEC_PASCAL=4 GGML_CUDA_FATTN_TILE_PASCAL_MIN=1"
# <arm> <binary: base|head> <switches> (all with the prompt kernel forced); the decode arms send the Q8_0 cases of up
# to 4 query tokens to the decode kernel, the other site that reads the factor
ARMS=${ARMS:-"base-unset|base|
head-unset|head|
base-f1|base|$F=1
head-f1|head|$F=1
head-s1.05|head|$F=1 $S=1.05
head-s1.00|head|$F=1 $S=1.0
head-s1.01|head|$F=1 $S=1.01
head-s1.02|head|$F=1 $S=1.02
head-s1.03|head|$F=1 $S=1.03
head-sbad|head|$F=1 $S=x1.03
head-snof|head|$S=1.03
base-dec-unset|base|$DEC
head-dec-unset|head|$DEC
base-dec-f1|base|$DEC $F=1
head-dec-s1.03|head|$DEC $F=1 $S=1.03"}
# arms that also run a live CPU check of every case before replaying it
LIVE_ARMS=${LIVE_ARMS:-head-unset head-s1.03}
mkdir -p "$W/rig" "$W/logs" "$W/checks" "$W/jitcache" "$W/tables"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
strip()  { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
export LC_ALL=C
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
build_dir() { ls -d "$1"/llama.cpp-*-"${2:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
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
echo "=== start $(date -Is): $REF ($HEAD_SHA) on $BASE_REF ($BASE_SHA); steps: $STEPS"

if step src; then
    SRC=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    clone_seed "$SRC" || finish 1
    git -C "$SRC" fetch -q fork "${REF#fork/}:refs/remotes/$REF" || finish 1
    got=$(git -C "$SRC" rev-parse "refs/remotes/$REF")
    echo "$REF = $got"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$SRC" cat-file -e "$BASE_SHA^{commit}" || { echo "no $BASE_SHA in the clone: stop"; finish 1; }
    { git -C "$SRC" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$SRC" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -n "${JIT_SEED:-}" ] && [ -z "$(ls -A "$W/jitcache")" ] && cp -a "$JIT_SEED/." "$W/jitcache/"
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="$EXTRA_TARGETS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$BASE/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), base $(wc -l < "$W/rig/bin-md5-base.txt");"\
        "the same bytes: $(comm -12 <(sort "$W/rig/bin-md5-base.txt") <(sort "$W/rig/bin-md5-head.txt") | wc -l);"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-base.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep -c '^[<>]')"
    grep -aE '^(name|commit|web page|avx|highest glibc|check tools):' "$T/BUILDINFO.txt"
    warnings "$BASE_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step sass; then
    SS=$W/checks/sass
    rm -rf "$SS" && mkdir -p "$SS/bin"
    H=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$H" ] || { echo "no head build"; finish 1; }
    for x in B H; do
        case $x in B) b=$BASE/bin/llama-server ;; H) b=$H/bin/llama-server ;; esac
        ln -f "$b" "$SS/bin/$x-llama-server" 2> /dev/null || cp "$b" "$SS/bin/$x-llama-server"
        echo "$x $(md5sum < "$SS/bin/$x-llama-server" | cut -c1-32) $b" >> "$SS/md5.txt"
        docker run --rm $LBL --cpu-shares=256 --network none -v "$SS/bin:$SS/bin:ro" "$IMAGE" cuobjdump -sass "$SS/bin/$x-llama-server" \
            | nice -n 19 c++filt > "$SS/dump-$x.txt"
        nice -n 19 python3 -I "$SASS_PY" hash < "$SS/dump-$x.txt" > "$SS/h-$x.tsv"
        say "sass $x: $(($(wc -l < "$SS/h-$x.tsv") - 1)) kernel copies, $(grep -c 'code for sm_' "$SS/dump-$x.txt") code sections"
    done
    nice -n 19 python3 -I "$D/c993_tables.py" hashcmp "$SS/h-B.tsv" "$SS/h-H.tsv" "$SS/differ.list" > "$SS/summary.txt" 2>&1
    nice -n 19 python3 -I "$SASS_PY" extract "$SS/differ.list" < "$SS/dump-B.txt" > "$SS/ext-B.txt"
    nice -n 19 python3 -I "$SASS_PY" extract "$SS/differ.list" < "$SS/dump-H.txt" > "$SS/ext-H.txt"
    nice -n 19 python3 -I "$D/c993_tables.py" sassdiff "$SS/ext-B.txt" "$SS/ext-H.txt" "$SS/lines.tsv" > "$SS/diff_summary.txt"
    # the base's constant 1.05f (0x3f866666, printed 1.0499999523...) and the head's constant-bank operand, for the record
    for x in B H; do
        echo "$(grep -cE '0x3f866666|1\.04999995' "$SS/ext-$x.txt") $(grep -c 'c\[0x3\]' "$SS/ext-$x.txt")" > "$SS/imm-$x.txt"
    done
    {
        echo "--- base against head: $(head -n 1 "$SS/summary.txt")"
        echo "    lines with 1.05f / with c[0x3][...] in the differing kernels: base $(cat "$SS/imm-B.txt"), head $(cat "$SS/imm-H.txt")"
        sed 's/^/    /' "$SS/diff_summary.txt" | head -n 40
    } | tee "$SS/pairs.txt"
    gzip -f "$SS/ext-B.txt" "$SS/ext-H.txt"
    rm -f "$SS"/dump-?.txt "$SS"/bin/?-llama-server
    echo "=== sass done $(date -Is)"
fi

C=$W/check
H=
VOL=(-v "$W:$W" -v "$BASE:$BASE:ro")
bin() { case $1 in base) echo "$BASE/bin/test-backend-ops" ;; head) echo "$H/bin/test-backend-ops" ;; esac; }
# <tag> <binary: base|head> <switches> <arguments after "test-backend-ops test -b CUDA0">: one container, its output in
# $C/<tag>.log, its command in $C/<tag>.cmd, its exit code in $C/rc.txt
tbo() {
    local tag=$1 b=$2 envs=$3 args=$4
    printf 'binary: %s\nenv: %s %s\nargs: test -b CUDA0 %s\n' "$(bin "$b")" "$TILE" "$envs" "$args" > "$C/$tag.cmd"
    timeout 3600 docker run --rm $LBL --cpu-shares=256 $MEM --name "$NAME-$tag" --gpus "\"device=$GPU\"" "${VOL[@]}" \
        -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" \
        bash -c "env $TILE $envs timeout 3000 $(bin "$b") test -b CUDA0 $args > $C/$tag.log 2>&1; echo $tag rc=\$? >> $C/rc.txt" \
        > /dev/null 2>&1
    say "  $tag: $(grep "^$tag " "$C/rc.txt" | tail -n 1); $(strip "$C/$tag.log" | grep -a 'tests passed' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$C/$tag.log" | grep -a -c ': FAIL')"
}

if step check; then
    H=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$H" ] || { echo "no head build"; finish 1; }
    rm -rf "$C" && mkdir -p "$C"
    if [ -n "$PREFLIGHT" ]; then
        $PREFLIGHT -g "$GPU" > "$C/preflight.txt" 2>&1 || { echo "CUDA preflight failed on GPU $GPU"; cat "$C/preflight.txt"; finish 1; }
        echo "preflight GPU $GPU: ok"
    fi
    md5sum "$BASE/bin/test-backend-ops" "$H/bin/test-backend-ops" | tee "$C/md5.txt"
    say "record: the head-256 FLASH_ATTN_EXT cases ($FILTER), base build, no fault"
    rm -rf "$W/rec"
    tbo record base "" "-o FLASH_ATTN_EXT -p '$FILTER' --record $W/rec --record-budget-gb 40"
    du -sh "$W/rec" | tee "$C/rec-size.txt"
    say "arms: the record replayed"
    while IFS='|' read -r tag b envs; do
        [ -n "$tag" ] || continue
        extra=""
        [[ " $LIVE_ARMS " == *" $tag "* ]] && extra="--cpu-check 1"
        tbo "$tag" "$b" "$envs" "-o FLASH_ATTN_EXT -p '$FILTER' --replay $W/rec --replay-log $C/$tag.tsv $extra"
    done <<< "$ARMS"
    echo "=== check done $(date -Is)"
fi

if step tables; then
    printf '%s\n' "$ARMS" > "$C/arms.txt"
    python3 -I "$D/c993_tables.py" tables "$W" "$W/tables" || finish 1
    echo "=== tables done $(date -Is)"
fi
finish 0
