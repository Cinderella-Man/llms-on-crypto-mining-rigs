#!/usr/bin/env bash
# C808 (2026-11-12): C728's K tiles only where they gain on a P106 (GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 takes Q4_K's 4-token
# products where C722's kernel holds at most 3 warps a block; _DMT_KT_M defaults to 2), on pascal-p106-v4 d21820340:
# built once and checked on a build machine, CPU only. The method is C805's (2026-11-12_c805_kcol_exact/run/checks.sh)
# with the v4 kernel merge's kern_pick.cu (2026-11-11_v4_kernel_merge/run/kern_pick.cu). Every build and test runs in a
# container without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at
# a time on a shared machine. Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in bin/)
#           in $M/main/rig, whose src/ and ccache/ it uses; its CUDA objects stay in src/build for the steps below
#   sass    the change is host code only: cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE
#           (a checked rig tarball whose ggml is the base's), anonymous-namespace hashes masked, cut into functions (the
#           day merge's sass_attrib.py): every function the base's, none missing or added
#   sassobj the device code (-sass, -ptx) of mmvq-pascal-dmt.cu's object against BASE_OBJ's, function by function
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; test-llama-archs in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch and with the tiles' settings (CUDA switches: the CPU results must be the base's)
#   pick    kern_pick.cu (-DTREE_KERN) built against the base's and the new mmvq-pascal-dmt.cu (nvcc, no GPU) and run
#           under each arm of PICK_ARMS on both; for every arm with _DMT_KT=1 also on the base under the same switches
#           without _DMT_KT and _DMT_KT_M (the kernel the product keeps when the tiles leave it)
#   pickcmp the comparisons: (1) with _DMT_KT unset every line the base's; (2) with _DMT_KT=2 every line the base's but the
#           tiles' M (2 unless _DMT_KT_M=1; the base's 1 unless _DMT_KT_M=2); (3) with _DMT_KT=1 the tiles only for Q4_K
#           at 4 tokens where the base's kernel without the tiles holds at most 3 warps, with the new M, and every other
#           line the base's (a tile the base took: the kernel without the tiles); (4) R120b's question: _DMT_KT_M=2
#           changes no line but a tile's M, in either tree (kern against kernm2, the arms R120b timed)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the new build tree (for the rig: mode dmtkt in it), copied to $M/bitid
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip and
#           test-llama-archs against the base's logits with no switch and with the tiles on (the "every switch off" check
#           of the merged tree), and kern_pick.cu on its mmvq-pascal-dmt.cu under every arm against BRANCH's lines (with
#           C809's one-warp rule modelled where the merged tree holds it: the only change allowed)
#           (MERGED_STEPS="build off pick", or "same" alone when the merge's tree is BRANCH's)
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (d21820340 in full), BASE_TREE (an unpacked rig
#   tarball whose ggml is the base's), BASE_OBJ (its build's ggml-cuda.dir), BASE_CPU (bin/ of a static CPU build of the
#   base's: test-llama-archs), SEED_SRC (a llama.cpp clone with the fork as remote "fork", https), SEED_CCACHE and
#   SEED_CPU_CCACHE (optional), BRANCH (default pascal-p106-v4-b12246-c808), FINAL_BRANCH (default pascal-p106-v4),
#   FINAL_REF (merged: a merge commit already in the clone, checked before it is pushed), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/pick, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c808}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-d21820340}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-rig sass sassobj cpu off pick pickcmp bitid c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-12_c808_ktiles_gain/run
KERN_PICK=$D/2026-11-11_v4_kernel_merge/run/kern_pick.cu
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
P=GGML_CUDA_MMVQ_PASCAL
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c808.log"; }
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
BASE_COMMIT=$(git -C "$SRC" rev-parse "$BASE_COMMIT") || { echo "no $BASE_COMMIT"; exit 1; }
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$M/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
log "C808 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
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
    dumpbin c808 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c808.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c808.$w.tsv" "$S/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c808.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    log "sass done"
fi

# the read-once kernels' object: the new build's against the base build's, function by function
if step sassobj; then
    S=$M/sass
    rm -rf "$S/obj" && mkdir -p "$S/obj"
    OBJS="mmvq-pascal-dmt.cu.o"
    objdump c808 "$O" "$S/obj" $OBJS
    objdump base "${BASE_OBJ:?BASE_OBJ=<the base build ggml-cuda.dir>}" "$S/obj" $OBJS
    nf=0 nd=0 nfn=0 nkt=0 dl=
    for f in $OBJS; do
        for w in sass ptx; do
            nf=$((nf + 1))
            nfn=$((nfn + $(wc -l < "$S/obj/c808.$f.$w.tsv")))
            nkt=$((nkt + $(grep -c 'mul_mat_vec_q_pascal_dmt_kt' "$S/obj/c808.$f.$w.tsv")))
            cmp -s "$S/obj/c808.$f.$w.tsv" "$S/obj/base.$f.$w.tsv" || { nd=$((nd + 1)); dl="$dl$f.$w "; }
        done
    done
    add sass "mmvq-pascal-dmt.cu.o against the base's" "$([ "$nd" = 0 ] && [ "$nfn" -gt 0 ] && echo SAME || echo DIFFERS)" \
        "sass and ptx: $nf dumps, $nfn functions ($nkt of them the tiles' kernels); dumps that differ: $nd ${dl}"
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
    cp "$C/tree/build/bin/test-llama-archs" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cut -c1-140 "$C/tree.txt"); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
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
KT_ARMS=("kt1:${P}_DMT=1 ${P}_DMT_KT=1" "kt1-m1:${P}_DMT=1 ${P}_DMT_KT=1 ${P}_DMT_KT_M=1")

if step off; then
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "${KT_ARMS[@]}"
    rm -f "$M"/off/base/logits-*.f32
fi

PK=$M/pick
# name:switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out); every arm runs on both trees, each tree reading its own.
# The v4 kernel merge's 37 arms, then R120b's perf arms (kern, kernm2) and the settings of M.
PICK_ARMS=${PICK_ARMS:-"$(cat "$D/2026-11-11_v4_kernel_merge/pick_arms.txt")
kern:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 RO8=1
kernm2:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 RO8=1 DMT_KT_M=2
kernm1:DMT=1 Q5K_MT=1 Q6K_MT=1 DMT_KT=1 RO8=1 DMT_KT_M=1
kt1-m1:DMT=1 DMT_KT=1 DMT_KT_M=1
kt2-m1:DMT=1 DMT_KT=2 DMT_KT_M=1
kt2-m2:DMT=1 DMT_KT=2 DMT_KT_M=2
kt1-nw4:DMT=1 DMT_KT=1 DMT_KT_NW=4
kt1-fault:DMT=1 DMT_KT=1 DMT_KT_FAULT=1
kt0:DMT=1 DMT_KT=0 DMT_KT_M=2
kt3:DMT=1 DMT_KT=3
all-m1:DMT=1 Q6K_MT=1 DMT_KT=1 DMT_KT_M=1 RO8=1"}
# <switches> -> the value of switch <name> in them (empty when unset)
swval() { local s; for s in $1; do [ "${s%%=*}" = "$2" ] && echo "${s#*=}"; done; }
# <switches> without _DMT_KT, _DMT_KT_M, _DMT_KT_NW and _DMT_KT_FAULT
nokt() { local s o=; for s in $1; do case ${s%%=*} in DMT_KT|DMT_KT_M|DMT_KT_NW|DMT_KT_FAULT) ;; *) o="$o $s" ;; esac; done; echo "${o# }"; }
if step pick; then
    rm -rf "$PK" && mkdir -p "$PK/arms"
    printf '%s\n' "$PICK_ARMS" | grep -v '^$' > "$PK/arms.txt"
    # the base under each _DMT_KT=1 arm's switches without the tiles
    while IFS=: read -r name sw; do
        [ "$(swval "$sw" DMT_KT)" = 1 ] && printf '%s\n' "$name:$(nokt "$sw")"
    done < "$PK/arms.txt" > "$PK/arms0.txt"
    cp "$KERN_PICK" "$PK/pick.cu"
    for t in base:$BASE_COMMIT c808:$TIP; do
        mkdir -p "$PK/${t%%:*}"
        git -C "$SRC" archive "${t#*:}" ggml | tar -x -C "$PK/${t%%:*}"
    done
    # the expressions kern_pick.cu repeats, in each tree (once each): the switch test of ggml_cuda_mmvq_pascal_dmt_takes,
    # Q6_K's one-warp rule, the tiles' call, the shared memory
    f() { grep -cF "$2" "$PK/$1/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu"; }
    E1='if (ncols < 2 || ncols > DMT_RO8_COLS || dmt_nt_mode(type, (int) ncols) == 0 ||'
    E2='return type == GGML_TYPE_Q6_K && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
    E3='if (r == 1 && dmt_kt(type, vx, vy, dst, nt, nsb, nrows, row_bytes, col_bytes_y, stride_col_dst, nw, stream)) {'
    E4='const int smem   = act + nw * p.wstage + dmt_tail(type);'
    found="base $(f base "$E1")$(f base "$E2")$(f base "$E3")$(f base "$E4") c808 $(f c808 "$E1")$(f c808 "$E2")$(f c808 "$E3")$(f c808 "$E4")"
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$O/flags.make")
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
        set -e
        for t in base c808; do
            W='"$PK"'/$t
            nice -n 19 nvcc -std=c++17 -O1 -w -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                -DTREE_KERN -I$W/ggml/include -I$W/ggml/src -I$W/ggml/src/ggml-cuda \
                -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections -o $W/pick '"$PK"'/pick.cu
        done
        run() { # <arms file> <suffix> <trees...>
            local f=$1 x=$2 name sw envs s t
            shift 2
            while IFS=: read -r name sw; do
                envs=""
                for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
                for t in "$@"; do
                    env $envs '"$PK"'/$t/pick > '"$PK"'/arms/$name-$t$x.tsv
                done
            done < $f
        }
        run '"$PK"'/arms.txt "" base c808
        run '"$PK"'/arms0.txt 0 base
        chown -R '"$(id -u):$(id -g)"' '"$PK"'' > "$M/logs/pick.log" 2>&1
    rc=$?
    echo $rc > "$PK/rc.txt"
    add pick build "$([ $rc = 0 ] && [ "$found" = "base 1111 c808 1111" ] && [ -s "$PK/arms/kern-c808.tsv" ] && echo OK || echo FAIL)" \
        "rc=$rc; base ${BASE_COMMIT:0:9}, c808 ${TIP:0:9}; the expressions kern_pick.cu repeats: $found (expected base 1111 c808 1111); $(wc -l < "$PK/arms.txt") arms ($(wc -l < "$PK/arms0.txt") with _DMT_KT=1, each also without the tiles on the base), $(tail -n +2 "$PK/arms/kern-c808.tsv" 2> /dev/null | wc -l) products each"
fi

if step pickcmp; then
    Ar=$PK/arms
    rc=$(cat "$PK/rc.txt" 2> /dev/null || echo 1)
    : > "$PK/changes.tsv"
    printf 'arm\tDMT_KT\tDMT_KT_M\tlines\tchanged\tunexpected\ttiles base\ttiles c808\n' > "$PK/counts.tsv"
    bad0= bad2= bad1=
    while IFS=: read -r name sw; do
        kt=$(swval "$sw" DMT_KT) km=$(swval "$sw" DMT_KT_M)
        mb=$([ "$km" = 2 ] && echo 2 || echo 1)
        mc=$([ "$km" = 1 ] && echo 1 || echo 2)
        ref=
        [ "$kt" = 1 ] && ref=$Ar/$name-base0.tsv
        # each c808 line against the line expected from the base's (and, at _KT=1, the base's line without the tiles);
        # a tile line's M is the last but one template argument of its kernel
        res=$(awk -F'\t' -v OFS='\t' -v arm="$name" -v kt="${kt:-0}" -v mb="$mb" -v mc="$mc" -v ref="$ref" -v ch="$PK/changes.tsv" \
                -v cf="$Ar/$name-c808.tsv" '
            BEGIN { while ((getline l < cf) > 0) c[++ncl] = l; if (ref != "") while ((getline l < ref) > 0) r[++nr] = l }
            FNR == 1 { next }
            {
                b = $0; cl = c[FNR]; n++
                tb = b ~ /pascal_dmt_kt</; tc = cl ~ /pascal_dmt_kt</
                ntb += tb; ntc += tc
                if (tb && b !~ ", " mb ", (true|false)>$") badm++
                want = b
                if (tb && (kt == 1 || kt == 2)) {
                    tile = b
                    sub(/, [12], (true|false)>$/, ", " mc ", " (b ~ /true>$/ ? "true" : "false") ">", tile)
                    want = tile
                    if (kt == 1) {
                        split(b, f, "\t"); split(r[FNR], g, "\t")
                        if (!(f[1] == "q4_K" && f[2] == 4 && r[FNR] !~ /pascal_dmt_kt</ && g[7] + 0 >= 1 && g[7] + 0 <= 3)) want = r[FNR]
                    }
                }
                if (cl != b) { nch++; print arm, b, "=>", cl >> ch }
                if (cl != want) { bad++; print arm, "UNEXPECTED", b, "=>", cl, "expected", want >> ch }
            }
            END { if (ncl != FNR) bad++; printf "%d\t%d\t%d\t%d\t%d", n, nch, bad + badm, ntb, ntc }' "$Ar/$name-base.tsv")
        printf '%s\t%s\t%s\t%s\n' "$name" "${kt:--}" "${km:--}" "$res" >> "$PK/counts.tsv"
        u=$(echo "$res" | cut -f3)
        case ${kt:-0} in 1) [ "$u" = 0 ] || bad1="$bad1$name($u) " ;; 2) [ "$u" = 0 ] || bad2="$bad2$name($u) " ;; *) [ "$u" = 0 ] || bad0="$bad0$name($u) " ;; esac
    done < "$PK/arms.txt"
    sum() { awk -F'\t' -v w="$1" 'NR > 1 && (w == "0" ? ($2 != "1" && $2 != "2") : $2 == w) { a++; l += $4; c += $5; tb += $7; tc += $8 } END { printf "%d arms, %d lines, %d changed, tiles base %d, c808 %d", a, l, c, tb, tc }' "$PK/counts.tsv"; }
    add pickcmp "_DMT_KT unset (or 0, 3): the base's lines" "$([ "$rc" = 0 ] && [ -z "$bad0" ] && echo SAME || echo DIFFERS)" "$(sum 0); unexpected: ${bad0:-none}"
    add pickcmp "_DMT_KT=2: the base's lines, the tiles' M the new default" "$([ "$rc" = 0 ] && [ -z "$bad2" ] && echo OK || echo FAIL)" "$(sum 2); unexpected: ${bad2:-none}"
    add pickcmp "_DMT_KT=1: tiles only for Q4_K at 4 tokens from at most 3 warps, M the new default" "$([ "$rc" = 0 ] && [ -z "$bad1" ] && echo OK || echo FAIL)" "$(sum 1); unexpected: ${bad1:-none}"
    # R120b's question: in each tree, kernm2 against kern changes no line but a tile's M
    for t in base c808; do
        r=$(awk -v cf="$Ar/kernm2-$t.tsv" 'BEGIN { while ((getline l < cf) > 0) c[++n] = l }
                FNR > 1 && $0 != c[FNR] { a = $0; b = c[FNR]; sub(/, [12], (true|false)>$/, "", a); sub(/, [12], (true|false)>$/, "", b)
                    if (a == b && $0 ~ /pascal_dmt_kt</) mt++; else other++ }
                END { if (n != FNR) other++; printf "%d %d", mt, other }' "$Ar/kern-$t.tsv")
        add pickcmp "R120b: _DMT_KT_M=2 against unset ($t tree)" "$([ "${r#* }" = 0 ] && echo OK || echo FAIL)" \
            "kernm2 against kern: lines that differ in a tile's M ${r% *}, any other line ${r#* }"
    done
    # the products R120b timed at 2560 x 6144 (K = 6144), 2-5 tokens: each tree's kernel under kern, kernm2, kernm1
    {
        printf 'type\ttokens\tK'
        for a in kern kernm2 kernm1; do for t in base c808; do printf '\t%s %s' "$a" "$t"; done; done
        printf '\n'
        for key in "iq4_xs 2" "iq4_xs 3" "iq4_xs 4" "q4_K 2" "q4_K 3" "q4_K 4" "q4_K 5" "q6_K 2" "q6_K 3" "q6_K 4"; do
            set -- $key
            printf '%s\t%s\t6144' "$1" "$2"
            for a in kern kernm2 kernm1; do for t in base c808; do
                printf '\t%s' "$(awk -F'\t' -v ty="$1" -v nt="$2" '$1 == ty && $2 == nt && $3 == 6144 { if (NF >= 11) printf "%s, %s warps, %s B", $11, $7, $9; else print $5 }' "$Ar/$a-$t.tsv")"
            done; done
            printf '\n'
        done
    } > "$PK/r120b_cases.tsv"
    # what _DMT_KT=1 (kern) sends to the tiles in each tree, over every row width kern_pick.cu covers
    for t in base c808; do
        awk -F'\t' -v t="$t" 'NR > 1 && $11 ~ /pascal_dmt_kt</ { print t, $1, $2, $3, $7 }' OFS='\t' "$Ar/kern-$t.tsv"
    done > "$PK/kern_tiles.tsv"
    # Flash-Next's widths (K = 2560 and 6144) under kern, both trees
    paste "$Ar/kern-base.tsv" "$Ar/kern-c808.tsv" | awk -F'\t' 'NR == 1 || $3 == 2560 || $3 == 6144' > "$PK/flashnext_kern.tsv"
    add pickcmp "tables" OK "pick/r120b_cases.tsv, pick/kern_tiles.tsv ($(awk '$1 == "base"' "$PK/kern_tiles.tsv" | wc -l) tile products in the base under kern, $(awk '$1 == "c808"' "$PK/kern_tiles.tsv" | wc -l) in c808), pick/flashnext_kern.tsv, pick/counts.tsv, pick/changes.tsv"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/bitid.log" 2>&1
    rc=$?
    # a copy outside REPO, which is replaced whenever the scripts are copied in again
    mkdir -p "$M/bitid" && cp "$X" "$M/bitid/multitoken_bitid" 2> /dev/null
    modes=
    for m in dmtkt dmt q6k; do grep -aqw "$m" "$X" 2> /dev/null && modes="$modes$m "; done
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$modes" = "dmtkt dmt q6k " ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; modes in it: ${modes:-none}; linked against ${TIP:0:9} (needs a GPU: run on the rig)"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c808=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
    # FINAL_REF: the merge commit, already in $SRC, checked before it is pushed to the fork; else the fork's FINAL_BRANCH
    if [ -n "${FINAL_REF:-}" ]; then
        F=$(git -C "$SRC" rev-parse "$FINAL_REF^{commit}") || { add merged "$FINAL_REF" FAIL "not in $SRC"; date -Is > "$M/DONE"; exit 1; }
    else
        git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
        F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    fi
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "${FINAL_REF:-fork/$FINAL_BRANCH} ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    ft=$(git -C "$SRC" rev-parse "$F^{tree}") tt=$(git -C "$SRC" rev-parse "$TIP^{tree}")
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); its tree $([ "$ft" = "$tt" ] && echo "is" || echo "is not") ${TIP:0:9}'s; commits since the base: $(git -C "$SRC" rev-list --count "$BASE_COMMIT..$F"); mmvq-pascal-dmt.cu $(git -C "$SRC" diff --quiet "$TIP" "$F" -- ggml/src/ggml-cuda/mmvq-pascal-dmt.cu && echo "the same as" || echo "differs from") ${TIP:0:9}'s"
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
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-kt1:${P}_DMT=1 ${P}_DMT_KT=1"
        rm -f "$M"/moff/base/logits-*.f32
    fi
    # kern_pick.cu on the merged tree under every arm, against BRANCH's lines. Where the merged tree holds C809's one-warp
    # rule (pascal-p106-v4 from d7cdd032a: a block of one warp is left to the k-column kernel for Q6_K and for 5 to 8
    # tokens, unless _DMT_NW=1), kern_pick.cu's copy of that rule (PICK_Q6K1) is set to it, and the one change allowed
    # against BRANCH's lines is such a launch, which the merged tree declines
    if [[ " $MS " == *" pick "* ]]; then
        MP=$M/mpick
        rm -rf "$MP" && mkdir -p "$MP/arms" "$MP/merged"
        git -C "$SRC" archive "$F" ggml | tar -x -C "$MP/merged"
        fm() { grep -cF "$1" "$MP/merged/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu"; }
        E1='if (ncols < 2 || ncols > DMT_RO8_COLS || dmt_nt_mode(type, (int) ncols) == 0 ||'
        E2='return type == GGML_TYPE_Q6_K && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
        E2A='const bool one_warp_declined = type == GGML_TYPE_Q6_K || ncols > DMT_MAX_COLS;'
        E2B='return one_warp_declined && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
        E3='if (r == 1 && dmt_kt(type, vx, vy, dst, nt, nsb, nrows, row_bytes, col_bytes_y, stride_col_dst, nw, stream)) {'
        E4='const int smem   = act + nw * p.wstage + dmt_tail(type);'
        Q1='#define PICK_Q6K1(type, nw) ((type) == GGML_TYPE_Q6_K && dmt_nw_env() != 1 && (nw) < 2)'
        Q9='#define PICK_Q6K1(type, nw) (((type) == GGML_TYPE_Q6_K || nt > DMT_MAX_COLS) && dmt_nw_env() != 1 && (nw) < 2)'
        c809=0
        if [ "$(fm "$E2A")$(fm "$E2B")" = 11 ]; then
            c809=1
            awk -v o="$Q1" -v n="$Q9" '$0 == o { print n; c++; next } { print } END { exit c != 1 }' "$KERN_PICK" > "$MP/pick.cu" || c809=2
        else
            cp "$KERN_PICK" "$MP/pick.cu"
        fi
        DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$O/flags.make")
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
            set -e
            W='"$MP"'/merged
            nice -n 19 nvcc -std=c++17 -O1 -w -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                -DTREE_KERN -I$W/ggml/include -I$W/ggml/src -I$W/ggml/src/ggml-cuda \
                -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections -o $W/pick '"$MP"'/pick.cu
            while IFS=: read -r name sw; do
                envs=""
                for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
                env $envs $W/pick > '"$MP"'/arms/$name-merged.tsv
            done < '"$PK"'/arms.txt
            chown -R '"$(id -u):$(id -g)"' '"$MP"'' > "$M/logs/mpick.log" 2>&1
        rc=$?
        # each arm: BRANCH's line => the merged tree's; allowed only (with C809's rule, _DMT_NW not 1) a one-warp launch of
        # C722's kernel at 5 to 8 tokens becoming "declined (one warp)"
        nd=0 nok=0 bad=
        printf 'arm\tlines that differ\tone-warp launches of 5-8 tokens declined\tunexpected\n' > "$MP/by_arm.tsv"
        while IFS=: read -r name sw; do
            nw1=$([ "$(swval "$sw" DMT_NW)" = 1 ] && echo 1 || echo 0)
            r=$(awk -F'\t' -v mf="$MP/arms/$name-merged.tsv" -v c809="$c809" -v nw1="$nw1" '
                BEGIN { while ((getline l < mf) > 0) m[++n] = l }
                FNR > 1 && $0 != m[FNR] {
                    d++
                    if (c809 == 1 && nw1 == 0 && $2 >= 5 && $7 == 1 && $11 ~ /pascal_dmt</ &&
                        m[FNR] == $1 "\t" $2 "\t" $3 "\t" $4 "\tdeclined (one warp)") ok++
                    else bad++
                }
                END { if (n != FNR) bad++; printf "%d %d %d", d, ok, bad }' "$PK/arms/$name-c808.tsv")
            set -- $r
            printf '%s\t%s\t%s\t%s\n' "$name" "$1" "$2" "$3" >> "$MP/by_arm.tsv"
            nd=$((nd + $1)) nok=$((nok + $2))
            [ "$3" = 0 ] || bad="$bad$name($3) "
        done < "$PK/arms.txt"
        add merged "kern_pick.cu's expressions in the merged tree" \
            "$([ "$(fm "$E1")$(fm "$E3")$(fm "$E4")" = 111 ] && { [ "$(fm "$E2")" = 1 ] || [ "$c809" = 1 ]; } && echo OK || echo NOTE)" \
            "the switch test, the tiles' call, the shared memory found $(fm "$E1")$(fm "$E3")$(fm "$E4") (111 = modelled); the one-warp rule: $([ "$c809" = 1 ] && echo "C809's (Q6_K and 5-8 tokens), PICK_Q6K1 set to it" || { [ "$c809" = 2 ] && echo "C809's, PICK_Q6K1 not found in kern_pick.cu" || echo "Q6_K's (found $(fm "$E2"))"; })"
        add merged "kern_pick.cu: merged against ${TIP:0:9}" \
            "$([ "$rc" = 0 ] && [ -z "$bad" ] && { [ "$nd" = 0 ] && echo SAME || echo OK; } || echo FAIL)" \
            "rc=$rc; $(wc -l < "$PK/arms.txt") arms; lines that differ: $nd, of them one-warp launches of 5-8 tokens the merged tree declines (C809): $nok; unexpected: ${bad:-none} (mpick/by_arm.tsv)"
    fi
fi

log "done"
date -Is > "$M/DONE"
