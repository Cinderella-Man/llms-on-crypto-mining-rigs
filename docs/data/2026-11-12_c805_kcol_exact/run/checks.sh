#!/usr/bin/env bash
# C805 (2026-11-12): the k-column kernel's declined products keep the one-token sums (GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT,
# off by default), on pascal-p106-v4 d097d6d88: built once and checked on a build machine, CPU only. The method is the v4
# kernel merge's (docs/data/2026-11-11_v4_kernel_merge/run/checks.sh). Every build and test runs in a container without
# GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on a shared
# machine. Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in bin/)
#           in $M/main/rig, whose src/ and ccache/ it uses; its CUDA objects stay in src/build for the steps below
#   names   C770's switch list in the build (src/build/common/switch-names.cpp): its count, and the two new names in it and
#           in bin/llama-server
#   sass    the change is host code only: cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE
#           (d097d6d88's checked rig tarball), anonymous-namespace hashes masked, cut into functions (the day merge's
#           sass_attrib.py dump): every function the base's, none missing or added (sass_attrib.py attrib, no branch)
#   sassobj the device code (-sass, -ptx) of the 17 k-column objects (mmvq-pascal-kcol.cu and the 16 per-type files)
#           against BASE_OBJ's (the base build's objects), function by function
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; its test-llama-archs and
#           llama-server in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch and with each of the new switch's settings and its fault (CUDA switches: the CPU results must be
#           the base's)
#   pick    run/kcol_pick.cu built against the base's and the new mmvq-pascal-kcol.cu (nvcc, no GPU) and run under each
#           arm of PICK_ARMS on both: with KCOL_EXACT unset (or not 1 or 2) every line the base's; with it on, every line
#           the base's but those of Q3_K, Q2_0, IQ4_NL and Q8_0 in rows of chunks not a multiple of 4 (1-8 tokens) that
#           the base leaves to other kernels, which the new tree takes, each launch within 48 KB; at =1 a product splits
#           into launches only where the base declines it at a 48 KB budget
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the new build tree: the modes q3k and glu in it
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip and
#           test-llama-archs against the base's logits with no switch and with KCOL_EXACT=1 (the "every switch off"
#           check of the merged tree; MERGED_STEPS="build off", or "same" alone when the merge's tree is BRANCH's)
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (d097d6d88 in full), BASE_TREE (the base's
#   unpacked rig tarball), BASE_OBJ (the base build's ggml-cuda.dir), BASE_CPU (bin/ of the base's static CPU build:
#   test-llama-archs), SEED_SRC (a llama.cpp clone with the fork as remote "fork", https), SEED_CCACHE and
#   SEED_CPU_CCACHE (optional), BRANCH (default pascal-p106-v4-b12246-c805), FINAL_BRANCH (default pascal-p106-v4),
#   JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/pick, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c805}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-d097d6d88a3ee3fb2e3dbdb90819d320557d9809}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-rig names sass sassobj cpu off pick pickcmp bitid c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-12_c805_kcol_exact/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
P=GGML_CUDA_MMVQ_PASCAL
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c805.log"; }
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
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$M/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
log "C805 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
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
    NEW="${P}_KCOL_EXACT ${P}_KCOL_EXACT_FAULT"
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    miss= mbin=
    for s in $NEW; do
        grep -q "^    \"$s\",\$" "$L" 2> /dev/null || miss="$miss$s "
        grep -aq "$s" "$T/bin/llama-server" || mbin="$mbin$s "
    done
    add names "switch list" "$([ -n "$n" ] && [ -z "$miss" ] && [ -z "$mbin" ] && echo OK || echo FAIL)" \
        "$n names in the build's list (the base's: 858); the two new names missing from it: ${miss:-none}, from bin/llama-server: ${mbin:-none}"
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
    dumpbin c805 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c805.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c805.$w.tsv" "$S/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c805.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    log "sass done"
fi

# the k-column objects: the new build's against the base build's, function by function
if step sassobj; then
    S=$M/sass
    rm -rf "$S/obj" && mkdir -p "$S/obj"
    OBJS="mmvq-pascal-kcol.cu.o"
    for t in q8_0 q2_0 q3_K q4_K q5_K q6_K iq4_xs q4_0 q5_0 iq4_nl iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_m; do
        OBJS="$OBJS mmvq-pascal-kcol-$t.cu.o"
    done
    objdump c805 "$O" "$S/obj" $OBJS
    objdump base "${BASE_OBJ:?BASE_OBJ=<the base build ggml-cuda.dir>}" "$S/obj" $OBJS
    nf=0 nd=0 nfn=0 dl=
    for f in $OBJS; do
        for w in sass ptx; do
            nf=$((nf + 1))
            nfn=$((nfn + $(wc -l < "$S/obj/c805.$f.$w.tsv")))
            cmp -s "$S/obj/c805.$f.$w.tsv" "$S/obj/base.$f.$w.tsv" || { nd=$((nd + 1)); dl="$dl$f.$w "; }
        done
    done
    add sass "k-column objects against the base's" "$([ "$nd" = 0 ] && [ "$nfn" -gt 0 ] && echo SAME || echo DIFFERS)" \
        "17 objects, sass and ptx: $nf dumps, $nfn functions; dumps that differ: $nd ${dl}"
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
    archs_arms "$M/off" "$C/bin" "tip:" "exact1:${P}_KCOL_EXACT=1" "exact2:${P}_KCOL_EXACT=2" \
        "exact1-fault:${P}_KCOL_EXACT=1 ${P}_KCOL_EXACT_FAULT=1" \
        "preset-exact1:${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0 ${P}_KCOL_EXACT=1"
    rm -f "$M"/off/base/logits-*.f32
fi

PK=$M/pick
# name:switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out); every arm runs on both trees, each tree reading its own
PICK_ARMS=${PICK_ARMS:-'none:
preset:KCOL_SMEM=32768 KCOL_TABLE=0
table0:KCOL_TABLE=0
smem48:KCOL_SMEM=49152 KCOL_TABLE=0
r1:KCOL_R=1 KCOL_SMEM=32768 KCOL_TABLE=0
r2:KCOL_R=2 KCOL_SMEM=32768 KCOL_TABLE=0
itt1:KCOL_ITT=1 KCOL_TABLE=0
mode2:KCOL=2
smem24:KCOL_SMEM=24576
none-x1:KCOL_EXACT=1
preset-x1:KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=1
table0-x1:KCOL_TABLE=0 KCOL_EXACT=1
smem48-x1:KCOL_SMEM=49152 KCOL_TABLE=0 KCOL_EXACT=1
r1-x1:KCOL_R=1 KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=1
r2-x1:KCOL_R=2 KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=1
itt1-x1:KCOL_ITT=1 KCOL_TABLE=0 KCOL_EXACT=1
mode2-x1:KCOL=2 KCOL_EXACT=1
smem24-x1:KCOL_SMEM=24576 KCOL_EXACT=1
none-x2:KCOL_EXACT=2
preset-x2:KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=2
table0-x2:KCOL_TABLE=0 KCOL_EXACT=2
r2-x2:KCOL_R=2 KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=2
preset-x3:KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=3
preset-x1f:KCOL_SMEM=32768 KCOL_TABLE=0 KCOL_EXACT=1 KCOL_EXACT_FAULT=1'}
if step pick; then
    rm -rf "$PK" && mkdir -p "$PK/arms"
    printf '%s\n' "$PICK_ARMS" > "$PK/arms.txt"
    cp "$RUN/kcol_pick.cu" "$PK/pick.cu"
    for t in base:$BASE_COMMIT c805:$TIP; do
        mkdir -p "$PK/${t%%:*}"
        git -C "$SRC" archive "${t#*:}" ggml | tar -x -C "$PK/${t%%:*}"
    done
    # the lines kcol_pick.cu repeats, in each tree's mmvq-pascal-kcol.cu
    f() { grep -cF "$2" "$PK/$1/ggml/src/ggml-cuda/mmvq-pascal-kcol.cu"; }
    c() { echo "$(f base "$1")$(f c805 "$1")"; }
    L1='if (t != kcol::KQ3_K && t != kcol::KIQ4_NL && !q8_short) {'
    L2='if (table && kcol_table_declines(src0->type, K, ncols)) {'
    L3='if (table && s.fallback) {'
    L4='const bool table = mode == 1 && ncols >= 2 && kcol_table_on();'
    L5='const bool table = mode == 1 && ncols >= 2 && kcol_table_on() && !exact;'
    L6='const bool exact = kcol_exact() != 0 && kcol_one_token_sums(t, K);'
    L7='o.r, o.itt, o.smem, kcol_exact(), s, cols, smsm, kcol_blocks)) {'
    found="$(c "$L1") $(c "$L2") $(c "$L3") $(c "$L4") $(c "$L5") $(c "$L6") $(c "$L7")"
    add pick "the decision's lines in each tree" "$([ "$found" = "11 11 11 10 01 01 01" ] && echo OK || echo FAIL)" \
        "base and c805 counts of the seven lines kcol_pick.cu repeats: $found (expected 11 11 11 10 01 01 01)"
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$O/flags.make")
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
        set -e
        for t in base c805; do
            W='"$PK"'/$t
            nice -n 19 nvcc -std=c++17 -O1 -w -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                -DTREE_$(echo $t | tr a-z A-Z) -I$W/ggml/include -I$W/ggml/src -I$W/ggml/src/ggml-cuda \
                -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections -o $W/pick '"$PK"'/pick.cu
        done
        while IFS=: read -r name sw; do
            envs=""
            for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
            for t in base c805; do
                env $envs '"$PK"'/$t/pick > '"$PK"'/arms/$name-$t.tsv
            done
        done < '"$PK"'/arms.txt
        chown -R '"$(id -u):$(id -g)"' '"$PK"'' > "$M/logs/pick.log" 2>&1
    echo $? > "$PK/rc.txt"
fi

# the comparisons, from the pick step's arms
if step pickcmp; then
    rc=$(cat "$PK/rc.txt" 2> /dev/null || echo 1)
    nl=$(wc -l < "$PK/arms/none-base.tsv" 2> /dev/null || echo 0)
    # 1. without the switch (and with a value other than 1 or 2) every line the base's
    bad= nsame=0
    while IFS=: read -r name sw; do
        case $name in *-x1|*-x2|*-x1f) continue ;; esac
        if cmp -s "$PK/arms/$name-base.tsv" "$PK/arms/$name-c805.tsv"; then nsame=$((nsame + 1)); else bad="$bad$name "; fi
    done < "$PK/arms.txt"
    add pick "switch off: the base's choices" "$([ $rc = 0 ] && [ -z "$bad" ] && [ "$nl" -gt 1 ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; $nsame arms (9 settings without KCOL_EXACT, preset with KCOL_EXACT=3), $((nl - 1)) products each (16 types, 15 widths, 14 row counts, 1-8 tokens): every line the base's; differ: ${bad:-none}"
    # 2. with the switch: the changed lines are the four types' the base leaves to another kernel, now
    # taken; the four types' products all taken (one token: those the kernel runs) (but rows the type's units do not divide); each launch
    # within 48 KB; launches x columns cover the tokens
    bad=
    : > "$PK/changes.tsv"
    : > "$PK/changes_count.tsv"
    while IFS=: read -r name sw; do
        case $name in *-x1|*-x2|*-x1f) ;; *) continue ;; esac
        paste "$PK/arms/$name-base.tsv" "$PK/arms/$name-c805.tsv" | awk -F'\t' -v arm="$name" 'BEGIN { OFS = "\t" }
            NR == 1 { next }
            {
                set = ($1 == "q3_K" || $1 == "q2_0" || $1 == "iq4_nl" || ($1 == "q8_0" && $2 % 64 == 0 && ($2 / 64) % 4 != 0))
                same = 1; for (i = 1; i <= 12; i++) if ($i != $(i + 12)) same = 0
                if (!same) {
                    nch++
                    if (!set || $5 == "kcol" || $17 != "kcol") bad_change++
                    print arm, $1, $2, $3, $4, $5, $17 ":" $18 "x" $19 " R" $20 " itt" $21 " " $22 " B " $23 " blk"
                }
                if (set && $17 != "kcol" && $17 != "width" && $17 != "one-token") not_taken++
                if ($17 == "kcol" && ($22 > 49152 || $18 * $19 < $4 || ($18 - 1) * $19 >= $4)) bad_launch++
                if ($17 == "kcol" && $18 > 1) nsplit++
            }
            END { printf "%s\t%d\t%d\t%d\t%d\t%d\n", arm, nch, bad_change, not_taken, bad_launch, nsplit > "/dev/stderr" }' \
            >> "$PK/changes.tsv" 2>> "$PK/changes_count.tsv"
    done < "$PK/arms.txt"
    bad=$(awk -F'\t' '$3 != 0 || $4 != 0 || $5 != 0 { printf "%s ", $1 }' "$PK/changes_count.tsv")
    add pick "switch on: only the four types' declined products change, all taken, within 48 KB" \
        "$([ $rc = 0 ] && [ -z "$bad" ] && [ -s "$PK/changes_count.tsv" ] && echo OK || echo FAIL)" \
        "arm, lines changed, changes not of the kind, products of the four types not taken, launches past 48 KB or not covering the tokens, split products: $(tr '\t' ' ' < "$PK/changes_count.tsv" | tr '\n' ';'); arms failing: ${bad:-none}"
    # 3. at =1 a product splits only where the base declines it at a 48 KB budget (preset-x1 against smem48-base)
    sp=$(paste "$PK/arms/preset-x1-c805.tsv" "$PK/arms/smem48-base.tsv" | awk -F'\t' 'NR > 1 && $5 == "kcol" {
            s = $6 > 1; d = $17 == "declined"; if (s != d) bad++; if (s) n++ } END { printf "%d %d", n, bad }')
    add pick "=1 splits only past 48 KB" "$([ "${sp#* }" = 0 ] && echo OK || echo FAIL)" \
        "preset-x1's split products and the base's declines at KCOL_SMEM=49152 KCOL_TABLE=0 (of the products preset-x1 takes): split ${sp% *}, mismatches ${sp#* }"
    # R120a's products that differed (q3k: 64 x 8192, 77 x 4096, 96 x 12288, 129 x 2048, Flash-Next's 2560 x 6144; glu's
    # 33 x 8192 in Q2_0 and Q3_K) under the preset: the base's choice, =1's and =2's, 1-8 tokens
    paste "$PK/arms/preset-base.tsv" "$PK/arms/preset-x1-c805.tsv" "$PK/arms/preset-x2-c805.tsv" | awk -F'\t' 'BEGIN { OFS = "\t"
            print "type", "rows", "K", "tokens", "preset (base)", "KCOL_EXACT=1", "KCOL_EXACT=2" }
        function plan(o) { return $(o + 5) == "kcol" ? $(o + 6) " x " $(o + 7) ", R " $(o + 8) ", itt " $(o + 9) ", " $(o + 10) " B" : $(o + 5) }
        NR > 1 && (($1 == "q3_K" && (($3 == 64 && $2 == 8192) || ($3 == 77 && $2 == 4096) || ($3 == 96 && $2 == 12288) ||
                    ($3 == 129 && $2 == 2048) || ($3 == 2560 && $2 == 6144) || ($3 == 33 && $2 == 8192))) ||
                   ($1 == "q2_0" && $3 == 33 && $2 == 8192)) { print $1, $3, $2, $4, plan(0), plan(12), plan(24) }' \
        > "$PK/r120a_products.tsv"
    add pick "R120a's products under the preset" "$(awk -F'\t' 'NR > 1 && $4 >= 2 && ($6 !~ / x / || $7 !~ / x /) { bad++ } END { exit bad > 0 }' "$PK/r120a_products.tsv" && echo OK || echo FAIL)" \
        "$(($(wc -l < "$PK/r120a_products.tsv") - 1)) lines (pick/r120a_products.tsv); at 2-8 tokens declined by the base: $(awk -F'\t' 'NR > 1 && $4 >= 2 && $5 == "declined"' "$PK/r120a_products.tsv" | wc -l), taken at =1 and =2: $(awk -F'\t' 'NR > 1 && $4 >= 2 && $6 ~ / x / && $7 ~ / x /' "$PK/r120a_products.tsv" | wc -l) of $(awk -F'\t' 'NR > 1 && $4 >= 2' "$PK/r120a_products.tsv" | wc -l)"
    # the faults change no choice
    add pick "the fault changes no choice" "$(cmp -s "$PK/arms/preset-x1-c805.tsv" "$PK/arms/preset-x1f-c805.tsv" && echo SAME || echo DIFFERS)" \
        "preset-x1f (KCOL_EXACT_FAULT=1) against preset-x1 in the new tree"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/bitid.log" 2>&1
    rc=$?
    modes=
    for m in q3k glu; do grep -aqw "$m" "$X" 2> /dev/null && modes="$modes$m "; done
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$modes" = "q3k glu " ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; modes in it: ${modes:-none}; linked against ${TIP:0:9}"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c805=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
    MS=${MERGED_STEPS:-build off}
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    ft=$(git -C "$SRC" rev-parse "$F^{tree}") tt=$(git -C "$SRC" rev-parse "$TIP^{tree}")
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); its tree $([ "$ft" = "$tt" ] && echo "is" || echo "is not") ${TIP:0:9}'s; commits since the base: $(git -C "$SRC" rev-list --count "$BASE_COMMIT..$F")"
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
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-exact1:${P}_KCOL_EXACT=1"
        rm -f "$M"/moff/base/logits-*.f32
    fi
fi

log "done"
date -Is > "$M/DONE"
