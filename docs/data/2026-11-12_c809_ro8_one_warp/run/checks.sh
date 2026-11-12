#!/usr/bin/env bash
# C809 (2026-11-12): with GGML_CUDA_MMVQ_PASCAL_RO8 the read-once kernels leave a launch of one warp a block to the
# k-column kernel, on pascal-p106-v4 d21820340: built once and checked on a build machine, CPU only. The method is C805's
# (docs/data/2026-11-12_c805_kcol_exact/run/checks.sh). Every build and test runs in a container without GPUs, waits for
# LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on a shared machine.
# Steps (STEPS; by default pick and pickcmp first, then the rest in this order):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in bin/)
#           in $M/main/rig, whose src/ and ccache/ it uses; its CUDA objects stay in src/build for the steps below
#   names   C770's switch list in the build (src/build/common/switch-names.cpp) against BASE_NAMES: no name added or lost
#   sass    the change is host code only: cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE,
#           anonymous-namespace hashes masked, cut into functions (the day merge's sass_attrib.py dump): every function
#           the base's, none missing or added
#   sassobj the device code (-sass, -ptx) of the two changed objects (mmvq-pascal-dmt.cu, mmvq-pascal-q5k.cu) and of
#           mmvq.cu and mmvq-pascal-kcol.cu against BASE_OBJ's, function by function
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; its test-llama-archs and
#           llama-server in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch and with the switch at 1 and 2 (CUDA switches: the CPU results must be the base's)
#   pick    run/c809_pick.cu built against the base's and the new mmvq-pascal-dmt.cu and, with -DPICK_Q5K,
#           mmvq-pascal-q5k.cu (nvcc, no GPU) and run under each arm of PICK_ARMS on both trees
#   pickcmp the comparisons: without _RO8 every line the base's; with it, the only lines that change are products of 5-8
#           tokens that the base launches at one warp a block, now "declined (one warp)", and none where _DMT_NW=1
#           (_Q5K_MT_NW=1) asks for one warp
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the new build tree: the modes dmt8 and q5k8 in it
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip,
#           test-llama-archs against the base's logits with no switch and with _RO8=1, and the pick of the merged tree's two
#           files against MERGED_PICK_BASE's (the fork's pascal-p106-v4 before the merge): only the one-warp _RO8 launches
#           change (MERGED_STEPS="build off pick")
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (d21820340 in full), BASE_TREE (an unpacked rig
#   tarball whose ggml/ is the base's), BASE_OBJ (that build's ggml-cuda.dir), BASE_NAMES (that build's
#   common/switch-names.cpp), BASE_CPU (bin/ of a static CPU build whose logits are the base's: test-llama-archs),
#   SEED_SRC (a llama.cpp clone with the fork as remote "fork", https), SEED_CCACHE, SEED_CPU_CCACHE, SEED_C565_STATIC and
#   SEED_C565_SHARED (optional ccache folders), BRANCH (default pascal-p106-v4-b12246-c809), FINAL_BRANCH (default
#   pascal-p106-v4), MERGED_PICK_BASE (a commit), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/pick, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c809}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-d218203406}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-pick pickcmp rig names sass sassobj cpu off bitid c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-12_c809_ro8_one_warp/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
P=GGML_CUDA_MMVQ_PASCAL
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c809.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
# shellcheck disable=SC2086
box()    { locked docker run --rm $DA --network none -v "$M:$M" "$@"; }
room() {
    local i
    for i in $(seq 1 24); do
        [ "$(free_gb)" -ge "$ROOM_GB" ] && return 0
        [ "$i" = 1 ] && log "$1 waits for $ROOM_GB GB free ($(free_gb) GB)"
        sleep 300
    done
    add "$1" disk FAIL "under $ROOM_GB GB free for 2 hours ($(free_gb) GB): stopped"
    exit 1
}
# <label> <dir> <out dir> <files...>: each object's device code (sass, ptx) cut into functions (cuobjdump in the CUDA
# image, the cutting on the host: the image has no python3)
objdump() {
    local l=$1 d=$2 o=$3 f w
    shift 3
    mkdir -p "$o"
    for f in "$@"; do
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$d/$f" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$o/$l.$f.$w.tsv"
        done
    done
}

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
BASE_COMMIT=$(git -C "$SRC" rev-parse "$BASE_COMMIT")
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$M/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
log "C809 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1)"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -d "$O" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step names; then
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    nb=$(grep -c '^    "' "${BASE_NAMES:?BASE_NAMES=<the base build switch-names.cpp>}" 2> /dev/null)
    add names "switch list" "$([ -n "$n" ] && [ "$n" -gt 0 ] && diff -q <(grep '^    "' "$L") <(grep '^    "' "$BASE_NAMES") > /dev/null && echo SAME || echo DIFFERS)" \
        "$n names in the build's list, $nb in the base's; the same names: $(diff -q <(grep '^    "' "$L") <(grep '^    "' "$BASE_NAMES") > /dev/null && echo yes || echo no)"
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    # <label> <llama-server>: its SASS and PTX cut into functions
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin c809 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c809.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c809.$w.tsv" "$S/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c809.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    log "sass done"
fi

# the changed objects and their callers: the new build's against the base build's, function by function
if step sassobj; then
    S=$M/sass
    rm -rf "$S/obj" && mkdir -p "$S/obj"
    OBJS="mmvq-pascal-dmt.cu.o mmvq-pascal-q5k.cu.o mmvq.cu.o mmvq-pascal-kcol.cu.o"
    objdump c809 "$O" "$S/obj" $OBJS
    objdump base "${BASE_OBJ:?BASE_OBJ=<the base build ggml-cuda.dir>}" "$S/obj" $OBJS
    nf=0 nd=0 nfn=0 dl=
    for f in $OBJS; do
        for w in sass ptx; do
            nf=$((nf + 1))
            nfn=$((nfn + $(wc -l < "$S/obj/c809.$f.$w.tsv")))
            cmp -s "$S/obj/c809.$f.$w.tsv" "$S/obj/base.$f.$w.tsv" || { nd=$((nd + 1)); dl="$dl$f.$w "; }
        done
    done
    add sass "the four objects against the base's" "$([ "$nd" = 0 ] && [ "$nfn" -gt 0 ] && echo SAME || echo DIFFERS)" \
        "$OBJS, sass and ptx: $nf dumps, $nfn functions; dumps that differ: $nd ${dl}"
    log "sassobj done"
fi

C=$M/cpu
if step cpu; then
    rm -rf "$C/tree" "$C/bin" && mkdir -p "$C/tree/src" "$C/bin"
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    git -C "$SRC" archive "$TIP" | tar -x -C "$C/tree/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$C/tree.txt"
    box -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$C/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$C/tree/src"' -B '"$C/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$C/tree/build"' -j '"$JOBS"' -- -k > '"$C/build.log"' 2>&1'
    rc=$?
    cp "$C/tree/build/bin/test-llama-archs" "$C/tree/build/bin/llama-server" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && [ -x "$C/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$C/tree.txt" | cut -c1-140); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

# <dir for logs> <bin> <arms...>: test-llama-archs of <bin> against the base's logits in $A/base, one line per arm
archs_arms() {
    local A=$1 B=$2 arm tag envs rc n bad wrote
    shift 2
    for arm in "$@"; do
        tag=${arm%%:*} envs=${arm#*:}
        box "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- '"$envs"' nice -n 19 '"$B"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        wrote=$(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log")
        add "$(basename "$A")" "$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && [ "$wrote" = 0 ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; against the base's $(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l) files: $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
}
# <dir>: the base's test-llama-archs writes every architecture's logits into <dir>/base
archs_base() {
    local A=$1 BC=${BASE_CPU:?BASE_CPU=<bin of the base static CPU build>} rc nb
    rm -rf "$A" && mkdir -p "$A/base"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$BC:$BC:ro" "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- nice -n 19 '"$BC"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A"'/base.log; exit ${PIPESTATUS[0]}'
    rc=$?
    nb=$(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l)
    add "$(basename "$A")" "base writes" "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" "rc=$rc; ${BC%/bin}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log")"
}

if step off; then
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "ro8-1:${P}_RO8=1" "ro8-2:${P}_RO8=2"
    rm -f "$M"/off/base/logits-*.f32
fi

PK=$M/pick
# name:switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out); every arm runs on both trees and both programs (dmt: IQ4_XS,
# Q4_K and Q6_K; q5k: Q5_K), each reading its own. The v4 kernel merge's 37 arms, then Q5_K's.
PICK_ARMS=${PICK_ARMS:-$(cat "$D/2026-11-11_v4_kernel_merge/pick_arms.txt")'
q5k:Q5K_MT=1
q5k-pf:Q5K_MT=1 Q5K_MT_PF=1
q5k-nopf:Q5K_MT=2
q5k-r2:Q5K_MT=1 Q5K_MT_R=2
q5k-nw1:Q5K_MT=1 Q5K_MT_NW=1
q5k-nw2:Q5K_MT=1 Q5K_MT_NW=2
q5k-1r:Q5K_1R=1
ro8-q5kpf:RO8=1 Q5K_MT_PF=1
ro8-q5knw1:RO8=1 Q5K_MT_NW=1
ro8-q5knw2:RO8=1 Q5K_MT_NW=2
ro8-q5k:RO8=1 Q5K_MT=1
ro8-nopf-q5kpf:RO8=2 Q5K_MT_PF=1
kern:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 RO8=1
kern-m2:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 DMT_KT_M=2 RO8=1
kern-nw1:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 RO8=1 DMT_NW=1 Q5K_MT_NW=1
ro8-0:RO8=0 DMT=1 Q5K_MT=1
ro8-3:RO8=3 DMT=1 Q5K_MT=1'}
# <tag 1> <tag 2>: the counts of the eight expressions c809_pick.cu repeats in the two trees' files under $PK/<tag>
pick_exprs() {
    local T1=$1 T2=$2 D1 D2 D3 D4 D5 Q1 Q2 Q3
    fd() { grep -cF "$2" "$PK/$1/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu"; }
    fq() { grep -cF "$2" "$PK/$1/ggml/src/ggml-cuda/mmvq-pascal-q5k.cu"; }
    cd_() { echo "$(fd "$T1" "$1")$(fd "$T2" "$1")"; }
    cq_() { echo "$(fq "$T1" "$1")$(fq "$T2" "$1")"; }
    D1='if (ncols < 2 || ncols > DMT_RO8_COLS || dmt_nt_mode(type, (int) ncols) == 0 ||'
    D2='return type == GGML_TYPE_Q6_K && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
    D3='const bool one_warp_declined = type == GGML_TYPE_Q6_K || ncols > DMT_MAX_COLS;'
    D4='return one_warp_declined && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
    D5='if (r == 1 && dmt_kt(type, vx, vy, dst, nt, nsb, nrows, row_bytes, col_bytes_y, stride_col_dst, nw, stream)) {'
    Q1='(ncols == 1 ? !q5kmt_one_row(nrows) : (ncols <= Q5KMT_MAX_COLS ? q5kmt_mode() : q5kmt_ro8()) == 0)) {'
    Q2='return q5kmt_nw((int) ncols, (int) (K / QK_K)) > 0;'
    Q3='return ncols > Q5KMT_MAX_COLS && q5kmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
    echo "$(cd_ "$D1") $(cd_ "$D2") $(cd_ "$D3") $(cd_ "$D4") $(cd_ "$D5") $(cq_ "$Q1") $(cq_ "$Q2") $(cq_ "$Q3")"
}
EXPRS_OK="11 10 01 01 11 11 10 01"
if step pick; then
    rm -rf "$PK" && mkdir -p "$PK/arms"
    printf '%s\n' "$PICK_ARMS" > "$PK/arms.txt"
    cp "$RUN/c809_pick.cu" "$PK/pick.cu"
    for t in base:$BASE_COMMIT c809:$TIP; do
        mkdir -p "$PK/${t%%:*}"
        git -C "$SRC" archive "${t#*:}" ggml | tar -x -C "$PK/${t%%:*}"
    done
    found=$(pick_exprs base c809)
    add pick "the decision's lines in each tree" "$([ "$found" = "$EXPRS_OK" ] && echo OK || echo FAIL)" \
        "base and c809 counts of the eight expressions c809_pick.cu repeats: $found (expected $EXPRS_OK)"
    FM=$O/flags.make; [ -f "$FM" ] || FM=$BASE_OBJ/flags.make # the same cmake configuration
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$FM")
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
        set -e
        for t in base c809; do
            W='"$PK"'/$t
            for p in dmt q5k; do
                nice -n 19 nvcc -std=c++17 -O1 -w -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                    -DTREE_$(echo $t | tr a-z A-Z) $([ $p = q5k ] && echo -DPICK_Q5K) -I$W/ggml/include -I$W/ggml/src \
                    -I$W/ggml/src/ggml-cuda -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections \
                    -o $W/pick-$p '"$PK"'/pick.cu
            done
        done
        while IFS=: read -r name sw; do
            [ -n "$name" ] || continue
            envs=""
            for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
            for t in base c809; do
                for p in dmt q5k; do
                    env $envs '"$PK"'/$t/pick-$p > '"$PK"'/arms/$name-$p-$t.tsv
                done
            done
        done < '"$PK"'/arms.txt
        chown -R '"$(id -u):$(id -g)"' '"$PK"'' > "$M/logs/pick.log" 2>&1
    echo $? > "$PK/rc.txt"
fi

# <base tree tag> <new tree tag> <out prefix>: the comparisons of two trees' arms
pick_compare() {
    local bt=$1 nt=$2 out=$3 rc nl name sw p ro8 nw1 bad= nsame=0 narms=0
    rc=$(cat "$PK/rc.txt" 2> /dev/null || echo 1)
    nl=$(( $(wc -l < "$PK/arms/none-dmt-$bt.tsv" 2> /dev/null || echo 1) + $(wc -l < "$PK/arms/none-q5k-$bt.tsv" 2> /dev/null || echo 1) - 2 ))
    printf 'arm\tprogram\ttype\tNT\tK\tnsb\tbase\tnew\n' > "$PK/$out-changes.tsv"
    printf 'arm\tprogram\tswitch\tone_warp_asked\tlines\tchanged\tchanged_not_one_warp_ro8\tone_warp_ro8_left\n' > "$PK/$out-counts.tsv"
    while IFS=: read -r name sw; do
        [ -n "$name" ] || continue
        narms=$((narms + 1))
        # the arm's _RO8 (any value but 0 sends 5-8 tokens), and whether it asks for one warp (_DMT_NW=1 for dmt, _Q5K_MT_NW=1 for q5k)
        ro8=$(echo " $sw " | sed -n 's/.* RO8=\([0-9]*\) .*/\1/p')
        for p in dmt q5k; do
            nw1=0
            [ $p = dmt ] && [[ " $sw " == *" DMT_NW=1 "* ]] && nw1=1
            [ $p = q5k ] && [[ " $sw " == *" Q5K_MT_NW=1 "* ]] && nw1=1
            paste "$PK/arms/$name-$p-$bt.tsv" "$PK/arms/$name-$p-$nt.tsv" | awk -F'\t' -v arm="$name" -v prog="$p" \
                -v ro8="${ro8:-0}" -v nw1="$nw1" -v cf="$PK/$out-changes.tsv" 'BEGIN { OFS = "\t" }
                NR == 1 { next }
                {
                    # the two lines side by side: the base line has 5 fields (off, declined) or 11 (a launch)
                    n1 = ($5 == "off" || $5 ~ /^declined/) ? 5 : 11
                    b = ""; for (i = 5; i <= n1; i++) b = b (i > 5 ? " " : "") $i
                    n = ""; for (i = n1 + 5; i <= NF; i++) n = n (i > n1 + 5 ? " " : "") $i
                    lines++
                    # a launch of one warp a block, 5-8 tokens: the base line has 11 fields and warps (field 7) = 1
                    onew = (n1 == 11 && $7 == 1 && $2 >= 5)
                    if (onew && ro8 != 0 && !nw1) left_expected++
                    if (b != n) {
                        ch++
                        print arm, prog, $1, $2, $3, $4, b, n >> cf
                        if (!(onew && ro8 != 0 && !nw1 && n == "declined (one warp)")) bad++
                    } else if (onew && ro8 != 0 && !nw1) {
                        left++
                    }
                }
                END { printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\n", arm, prog, ro8, nw1, lines, ch, bad, left }' >> "$PK/$out-counts.tsv"
        done
    done < "$PK/arms.txt"
    echo "$rc $nl $narms"
}

# <counts file> <row> <label>: the two verdicts of a comparison (pick_compare's counts)
pick_verdicts() {
    local cf=$1 row=$2 label=$3 off offbad on onbad
    # 1. without _RO8 (0 or unset): every line the base's
    off=$(awk -F'\t' 'NR > 1 && $3 == 0' "$cf")
    offbad=$(echo "$off" | awk -F'\t' '$6 != 0 { printf "%s/%s ", $1, $2 }')
    add "$row" "${label:+$label }switch off: the base's choices" "$([ "$rc" = 0 ] && [ -n "$off" ] && [ -z "$offbad" ] && [ "$nl" -gt 1 ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; $(echo "$off" | wc -l) arm-programs without _RO8 (of $narms arms x 2 programs), $nl products each: every line the base's; differ: ${offbad:-none}"
    # 2. with it (any value but 0): only the base's one-warp 5-8-token launches change, each to "declined (one warp)", all
    # of them, and none where the arm asks for one warp
    on=$(awk -F'\t' 'NR > 1 && $3 != 0' "$cf")
    onbad=$(echo "$on" | awk -F'\t' '$7 != 0 || $8 != 0 || ($4 == 1 && $6 != 0) { printf "%s/%s ", $1, $2 }')
    add "$row" "${label:+$label }switch on: only the one-warp launches change, all declined" "$([ "$rc" = 0 ] && [ -n "$on" ] && [ -z "$onbad" ] && echo OK || echo FAIL)" \
        "$(echo "$on" | wc -l) arm-programs with _RO8; lines changed per arm-program: $(echo "$on" | awk -F'\t' '{ printf "%s/%s %d; ", $1, $2, $6 }'); failing: ${onbad:-none}"
}

if step pickcmp; then
    read -r rc nl narms <<< "$(pick_compare base c809 pick)"
    pick_verdicts "$PK/pick-counts.tsv" pick ""
    # Flash-Next's two from R120b, in the arm "kern" (R120b's): the base's launch and the new tree's
    fn=$(awk -F'\t' '$1 == "kern" && (($3 == "iq4_xs" && $4 == 6) || ($3 == "q5_K" && $4 == 5)) && $5 == 6144 { printf "%s %s tokens K %s: %s -> %s; ", $3, $4, $5, $7, $8 }' "$PK/pick-changes.tsv")
    add pick "R120b's two losses" "$([ "$(echo "$fn" | grep -o 'declined (one warp)' | wc -l)" = 2 ] && echo OK || echo FAIL)" "${fn:-none found}"
    # the one-warp launches the base makes at 5-8 tokens under _RO8=1 (arm ro8 for IQ4_XS and Q4_K, ro8-q5k for Q5_K): by
    # type and token count
    {
        printf 'type\tNT\trow widths K declined\tcount\n'
        awk -F'\t' 'BEGIN { OFS = "\t" }
            NR > 1 && (($1 == "ro8" && $2 == "dmt") || ($1 == "ro8-q5k" && $2 == "q5k")) {
                k = $3 "\t" $4; w[k] = w[k] (w[k] ? " " : "") $5; c[k]++ }
            END { for (k in w) print k, w[k], c[k] }' "$PK/pick-changes.tsv" | sort -t$'\t' -k1,1 -k2,2n
    } > "$PK/one_warp_ro8.tsv"
    add pick "the products declined under _RO8=1" OK \
        "$(tail -n +2 "$PK/one_warp_ro8.tsv" | awk -F'\t' '{ printf "%s %s tokens: %d widths; ", $1, $2, $4 }')"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/bitid.log" 2>&1
    rc=$?
    modes=
    for m in dmt8 q5k8 dmt q5k; do grep -aqw "$m" "$X" 2> /dev/null && modes="$modes$m "; done
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$modes" = "dmt8 q5k8 dmt q5k " ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; modes in it: ${modes:-none}; linked against ${TIP:0:9}"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        seed=$([ $sh = OFF ] && echo "${SEED_C565_STATIC:-}" || echo "${SEED_C565_SHARED:-}")
        [ -d "$M/c565-$v/ccache" ] || [ -z "$seed" ] || { mkdir -p "$M/c565-$v" && cp -al "$seed" "$M/c565-$v/ccache"; }
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c809=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="--label pascal-quants=1 ${DOCKER_ARGS:-}" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        tail -n +2 "$M/c565-$v/summary.tsv" 2> /dev/null | awk -F'\t' -v r="c565 $v" 'BEGIN { OFS = "\t" }
            NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next } { print r, $1, $2, $3 }' \
            | tee -a "$M/summary.tsv"
        rm -rf "$M/c565-$v/tree"
    done
fi

if step trim; then
    rm -rf "$C/tree"
    add trim "CPU tree" OK "removed (the binaries in cpu/bin stay); $(du -sh "$M" | cut -f1) in $M; $(free_gb) GB free"
fi

if step merged; then
    MS=${MERGED_STEPS:-build off pick}
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); commits since the base: $(git -C "$SRC" rev-list --count "$BASE_COMMIT..$F")"
    MC=$M/merged
    if [[ " $MS " == *" build "* ]]; then
        rm -rf "$MC/tree" "$MC/bin" && mkdir -p "$MC/tree/src" "$MC/bin"
        [ -d "$MC/ccache" ] || cp -al "$C/ccache" "$MC/ccache" 2> /dev/null || mkdir -p "$MC/ccache"
        git -C "$SRC" archive "$F" | tar -x -C "$MC/tree/src"
        box -e CCACHE_DIR="$MC/ccache" -e CCACHE_BASEDIR="$MC/tree" "$CPU_IMAGE" bash -c '
            cmake -S '"$MC/tree/src"' -B '"$MC/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$MC/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$MC/tree/build"' -j '"$JOBS"' -- -k > '"$MC/build.log"' 2>&1'
        rc=$?
        cp "$MC/tree/build/bin/test-llama-archs" "$MC/bin/" 2> /dev/null
        add merged "cpu build" "$([ $rc = 0 ] && [ -x "$MC/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; ${F:0:9}; static, every target; $(grep -ac 'warning:' "$MC/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$MC/build.log") error lines"
        rm -rf "$MC/tree"
    fi
    if [[ " $MS " == *" off "* ]]; then
        archs_base "$M/moff"
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-ro8-1:${P}_RO8=1"
        rm -f "$M"/moff/base/logits-*.f32
    fi
    if [[ " $MS " == *" pick "* ]]; then
        # the merged tree's two files against those of the fork's tip before the merge (MERGED_PICK_BASE), with the
        # tree macros of TREE_C809 and TREE_BASE; then the same comparisons as pickcmp
        PB=$(git -C "$SRC" rev-parse "${MERGED_PICK_BASE:?MERGED_PICK_BASE=<pascal-p106-v4 before the merge>}")
        for t in mbase:$PB merged:$F; do
            rm -rf "${PK:?}/${t%%:*}" && mkdir -p "$PK/${t%%:*}"
            git -C "$SRC" archive "${t#*:}" ggml | tar -x -C "$PK/${t%%:*}"
        done
        found=$(pick_exprs mbase merged)
        add merged "pick: the decision's lines in each tree" "$([ "$found" = "$EXPRS_OK" ] && echo OK || echo FAIL)" \
            "${PB:0:9} and ${F:0:9} counts of the eight expressions c809_pick.cu repeats: $found (expected $EXPRS_OK)"
        FM=$O/flags.make; [ -f "$FM" ] || FM=$BASE_OBJ/flags.make # the same cmake configuration
        DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$FM")
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
            set -e
            for t in mbase merged; do
                W='"$PK"'/$t
                D=$([ $t = merged ] && echo -DTREE_C809 || echo -DTREE_BASE)
                for p in dmt q5k; do
                    nice -n 19 nvcc -std=c++17 -O1 -w -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                        $D $([ $p = q5k ] && echo -DPICK_Q5K) -I$W/ggml/include -I$W/ggml/src -I$W/ggml/src/ggml-cuda \
                        -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections -o $W/pick-$p '"$PK"'/pick.cu
                done
            done
            while IFS=: read -r name sw; do
                [ -n "$name" ] || continue
                envs=""
                for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
                for t in mbase merged; do
                    for p in dmt q5k; do
                        env $envs '"$PK"'/$t/pick-$p > '"$PK"'/arms/$name-$p-$t.tsv
                    done
                done
            done < '"$PK"'/arms.txt
            chown -R '"$(id -u):$(id -g)"' '"$PK"'' > "$M/logs/mpick.log" 2>&1
        echo $? > "$PK/rc.txt"
        read -r rc nl narms <<< "$(pick_compare mbase merged mpick)"
        pick_verdicts "$PK/mpick-counts.tsv" merged "pick ${PB:0:9} -> ${F:0:9}:"
        add merged "pick: the same changes as c809 against its base" \
            "$(cmp -s <(tail -n +2 "$PK/mpick-changes.tsv") <(tail -n +2 "$PK/pick-changes.tsv") && echo SAME || echo DIFFERS)" \
            "$(($(wc -l < "$PK/mpick-changes.tsv") - 1)) changed lines on the merged tree, $(($(wc -l < "$PK/pick-changes.tsv") - 1)) on c809 against d21820340"
    fi
fi

log "done"
date -Is > "$M/DONE"
