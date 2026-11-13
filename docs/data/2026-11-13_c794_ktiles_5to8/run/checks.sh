#!/usr/bin/env bash
# C794 (2026-11-13): C728's K tiles at 5 to 8 tokens (GGML_CUDA_MMVQ_PASCAL_RO8_KT, fork branch
# pascal-p106-v4-b12246-c794 on pascal-p106-v4 2144cf760), built once and checked on a build machine, CPU only. The
# method is C808's (2026-11-12_c808_ktiles_gain/run/checks.sh) with C728's replay and register steps. Every build and
# test runs in a container without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19).
# Steps (STEPS, in this order by default):
#   obj     mmvq-pascal-dmt.cu of the tip and of the base compiled alone with the base build's flags (sm_61 and
#           compute_61), before the rig build: every function of the base's object the same code in the tip's, the new
#           ones only mul_mat_vec_q_pascal_dmt_kt at NT = 5 to 8 (2 types x 2 M x (2 PF at 5 and 6, 1 at 7 and 8) = 24); their registers,
#           stack and local memory (no spills) and instruction counts (C728's sass_check.sh)
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1) in $M/main/rig
#   sass    cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE (the base's rig tarball),
#           anonymous-namespace hashes masked, cut into functions (the day merge's sass_attrib.py): every function the
#           base's code, the new ones only the 24 tile kernels
#   sassobj mmvq-pascal-dmt.cu's object of the rig build against BASE_OBJ's, function by function, and sass_check.sh's
#           table of the rig build's kernels (registers, spills, counts)
#   record  the replay set ro8kt (scripts/record_checks.sh SUITE=ro8kt) with the tarball's test-backend-ops; the cases a
#           set in LINK_FROM holds are hardlinked in
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the rig build's tree (modes kt8, dmtkt, dmt8; it
#           needs a GPU: the rig runs it), copied to $M/bitid
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target); test-llama-archs in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch and with the tiles' settings (CUDA switches: the CPU results must be the base's)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  (not by default) FINAL_REF, the merge of BRANCH into FINAL_BRANCH, checked before it is pushed: its
#           mmvq-pascal-dmt.cu compiled alone against the tip's (the same functions), a static CPU build and
#           test-llama-archs against the base's logits, off and with the tiles on
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (2144cf760), BASE_TREE (an unpacked rig
#   tarball whose ggml is the base's), BASE_OBJ (its build's ggml-cuda.dir, with flags.make), BASE_CPU (bin/ of a static
#   CPU build whose CPU code is the base's: test-llama-archs), SEED_SRC (a llama.cpp clone with the fork as remote
#   "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), LINK_FROM (replay sets to link cases from), BRANCH
#   (default pascal-p106-v4-b12246-c794), FINAL_BRANCH (pascal-p106-v4), FINAL_REF, JOBS (8), THREADS (8), DOCKER_ARGS,
#   IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/obj, $M/sass, $M/sets, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c794}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-2144cf760}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-obj rig sass sassobj record bitid cpu off c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SASS_CHECK=$D/2026-11-11_c728_readonce_k6144/run/sass_check.sh
SRC=$M/main/rig/src
O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
P=GGML_CUDA_MMVQ_PASCAL
KTN=24 # new tile kernels: IQ4_XS and Q4_K, M = 1 and 2, at NT = 5 and 6 with and without the prefetch, at 7 and 8 without
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c794.log"; }
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
# <listing file> <sass|ptx> <out tsv>: the listing masked and cut into functions (name, md5s)
cut_fn() { sed -E "$MASK" "$1" | nice -n 19 python3 "$ATTRIB" dump "$2" > "$3"; }
# <base tsv> <new tsv> <out tsv>: per function same, new, gone or DIFFERS; prints "<n> <same> <new kt> <other>"
cmp_fn() {
    awk -F'\t' 'BEGIN { OFS = "\t" } NR == FNR { b[$1] = $2; next } { t[$1] = $2 }
        END { for (f in t) print f, (f in b ? (b[f] == t[f] ? "same" : "DIFFERS") : "new")
              for (f in b) if (!(f in t)) print f, "gone" }' "$1" "$2" | sort -k2,2 -k1,1 > "$3"
    awk -F'\t' '{ n++ } $2 == "same" { s++ } $2 == "new" && $1 ~ /mul_mat_vec_q_pascal_dmt_kt/ { k++ }
        $2 != "same" && !($2 == "new" && $1 ~ /mul_mat_vec_q_pascal_dmt_kt/) { o++ }
        END { printf "%d %d %d %d", n, s, k, o }' "$3"
}
# <dir with dmt_res.txt, dmt.sass, mmvqp.sass.gz> <label>: sass_check.sh's table and its verdict on the new kernels
kt_table() {
    local S=$1 k spills regs
    SASS=$S bash "$SASS_CHECK" > "$S/sass_counts.tsv" 2> "$S/sass_check.err"
    k=$(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" && $3 >= 5' "$S/sass_counts.tsv" | wc -l)
    spills=$(awk -F'\t' 'NR > 1 && $1 ~ /^mul_mat_vec_q_pascal_dmt/ && ($7 != 0 || $8 != 0)' "$S/sass_counts.tsv" | wc -l)
    regs=$(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" && $3 >= 5 { print $6 }' "$S/sass_counts.tsv" | sort -n | sed -n '1p;$p' | tr '\n' '-' | sed 's/-$//')
    add "$2" "tile kernels at 5-8 tokens" "$([ "$k" = "$KTN" ] && [ "$spills" = 0 ] && echo OK || echo CHECK)" \
        "$k instantiations; kernels of the file with stack or local memory: $spills; registers $regs (at most 168 for 2 blocks of 6 warps); BAR a kernel at 5-8 tokens: $(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" && $3 >= 5 { print $21 }' "$S/sass_counts.tsv" | sort -u | tr '\n' ' ')"
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
log "C794 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1)"

# <out dir> <commit> <label>: mmvq-pascal-dmt.cu of <commit> compiled alone with the base build's flags; its -res-usage,
# -sass and -ptx in <out dir>/<label>.{res.txt,sass,ptx}
obj_one() {
    local OB=$1 c=$2 l=$3 DEFS FLAGS
    rm -rf "$OB/$l" && mkdir -p "$OB/$l"
    git -C "$SRC" archive "$c" ggml | tar -x -C "$OB/$l"
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "${BASE_OBJ:?BASE_OBJ=<the base build ggml-cuda.dir>}/flags.make")
    FLAGS=$(sed -n 's/^CUDA_FLAGS = //p' "$BASE_OBJ/flags.make")
    box "$IMAGE" bash -c '
        set -e
        cd '"$OB/$l"'
        nice -n 19 nvcc '"$DEFS"' -I'"$OB/$l"'/ggml/src -I'"$OB/$l"'/ggml/include '"$FLAGS"' -x cu \
            -c ggml/src/ggml-cuda/mmvq-pascal-dmt.cu -o '"$OB/$l"'.o
        cuobjdump -res-usage '"$OB/$l"'.o > '"$OB/$l"'.res.txt
        cuobjdump -sass '"$OB/$l"'.o > '"$OB/$l"'.sass
        cuobjdump -ptx '"$OB/$l"'.o > '"$OB/$l"'.ptx
        chown -R '"$(id -u):$(id -g)"' '"$OB"'' > "$OB/$l.log" 2>&1
}
# <out dir> <label a> <label b> <row>: the two objects' functions compared, sass and ptx
obj_cmp() {
    local OB=$1 a=$2 b=$3 row=$4 w r
    for w in sass ptx; do
        cut_fn "$OB/$a.$w" $w "$OB/$a.$w.tsv"
        cut_fn "$OB/$b.$w" $w "$OB/$b.$w.tsv"
        r=$(cmp_fn "$OB/$a.$w.tsv" "$OB/$b.$w.tsv" "$OB/$b-vs-$a.$w.tsv")
        set -- $r
        add "$row" "mmvq-pascal-dmt.cu $w: $b against $a" "$([ "$1" -gt 0 ] && [ "$4" = 0 ] && echo SAME || echo DIFFERS)" \
            "$1 functions: $2 the same code, $3 new tile kernels, $4 other (new, gone or differing; $OB/$b-vs-$a.$w.tsv)"
    done
}

if step obj; then
    OB=$M/obj
    mkdir -p "$OB"
    obj_one "$OB" "$TIP" tip
    rt=$?
    obj_one "$OB" "$BASE_COMMIT" base
    rb=$?
    add obj compile "$([ $rt = 0 ] && [ $rb = 0 ] && echo OK || echo FAIL)" "tip rc=$rt, base rc=$rb; $(grep -ac 'warning' "$OB/tip.log") warning lines in the tip's"
    if [ $rt = 0 ] && [ $rb = 0 ]; then
        obj_cmp "$OB" base tip obj
        S=$OB/check && mkdir -p "$S"
        cp "$OB/tip.res.txt" "$S/dmt_res.txt" && cp "$OB/tip.sass" "$S/dmt.sass"
        # the one-token kernels (the last two FFMA of Q4_K's) from the base build's mmvq-pascal.cu.o
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$M:$M" -v "$BASE_OBJ:$BASE_OBJ:ro" "$IMAGE" bash -c \
            "cuobjdump -sass $BASE_OBJ/mmvq-pascal.cu.o | gzip > $S/mmvqp.sass.gz; chown $(id -u):$(id -g) $S/mmvqp.sass.gz"
        kt_table "$S" obj
    fi
    log "obj done"
fi

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -d "$O" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); llama-server md5 $(md5sum < "$T/bin/llama-server" | cut -d' ' -f1); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    for side in "c794:$T/bin/llama-server" "base:${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"; do
        l=${side%%:*} b=${side#*:} d=$(dirname "${side#*:}")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$b" > "$S/$l.$w"
            cut_fn "$S/$l.$w" $w "$S/$l.$w.tsv"
            rm -f "$S/$l.$w"
        done
    done
    for w in sass ptx; do
        r=$(cmp_fn "$S/base.$w.tsv" "$S/c794.$w.tsv" "$S/functions.$w.tsv")
        set -- $r
        add sass "llama-server $w against the base" "$([ "$1" -gt 0 ] && [ "$3" = "$KTN" ] && [ "$4" = 0 ] && echo SAME || echo DIFFERS)" \
            "$1 functions: $2 the base's code, $3 new tile kernels (expected $KTN), $4 other (new, gone or differing)"
    done
    log "sass done"
fi

if step sassobj; then
    S=$M/sass/obj
    rm -rf "$S" && mkdir -p "$S"
    for side in "c794:$O" "base:$BASE_OBJ"; do
        l=${side%%:*} d=${side#*:}
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$d:$d:ro" -v "$M:$M" "$IMAGE" bash -c "
            cuobjdump -res-usage $d/mmvq-pascal-dmt.cu.o > $S/$l.res.txt
            cuobjdump -sass $d/mmvq-pascal-dmt.cu.o > $S/$l.sass
            cuobjdump -ptx $d/mmvq-pascal-dmt.cu.o > $S/$l.ptx
            cuobjdump -sass $d/mmvq-pascal.cu.o | gzip > $S/$l.mmvqp.sass.gz
            chown -R $(id -u):$(id -g) $S" > "$M/logs/sassobj.log" 2>&1
    done
    obj_cmp "$S" base c794 sassobj
    mkdir -p "$S/check"
    cp "$S/c794.res.txt" "$S/check/dmt_res.txt" && cp "$S/c794.sass" "$S/check/dmt.sass" && cp "$S/c794.mmvqp.sass.gz" "$S/check/mmvqp.sass.gz"
    kt_table "$S/check" sassobj
    add sassobj "mmvq-pascal.cu.o (the one-token kernels)" "$(cmp -s <(zcat "$S/c794.mmvqp.sass.gz" | sed -E "$MASK") <(zcat "$S/base.mmvqp.sass.gz" | sed -E "$MASK") && echo SAME || echo DIFFERS)" "SASS, masked, against the base's"
    log "sassobj done"
fi

if step record; then
    mkdir -p "$M/sets"
    locked env STEPS="record tar" OUT="$M/sets" SUITE=ro8kt BIN="$T/bin" JOBS="$JOBS" LINK_FROM="${LINK_FROM:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$M/logs/record.log" 2>&1
    rc=$?
    add record "set ro8kt" "$([ $rc = 0 ] && [ -f "$M/sets/ro8kt.tar" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -a 'linked .* in all' "$M/sets/ro8kt.log" | tail -n 1 | cut -d' ' -f2-); $(grep -a 'manifest:' "$M/sets/ro8kt.log" | tail -n 1 | cut -d' ' -f2-); runs $(wc -l < "$M/sets/ro8kt/runs.tsv" 2> /dev/null); $(tail -n 1 "$M/sets/ro8kt.log" 2> /dev/null | cut -d' ' -f2-)"
    log "record done"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/bitid.log" 2>&1
    rc=$?
    mkdir -p "$M/bitid" && cp "$X" "$M/bitid/multitoken_bitid" 2> /dev/null
    modes=
    for m in kt8 dmtkt dmt8; do grep -aqw "$m" "$X" 2> /dev/null && modes="$modes$m "; done
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$modes" = "kt8 dmtkt dmt8 " ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; modes in it: ${modes:-none}; linked against ${TIP:0:9} (needs a GPU: run on the rig)"
fi

C=$M/cpu
cpu_build() { # <dir> <commit> <row>: a static native CPU build, test-llama-archs in <dir>/bin
    local B=$1 c=$2 row=$3 rc
    rm -rf "$B/tree" "$B/bin" && mkdir -p "$B/tree/src" "$B/bin"
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    git -C "$SRC" archive "$c" | tar -x -C "$B/tree/src"
    box -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$B/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$B/tree/src"' -B '"$B/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$B/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$B/tree/build"' -j '"$JOBS"' -- -k > '"$B/build.log"' 2>&1'
    rc=$?
    cp "$B/tree/build/bin/test-llama-archs" "$B/bin/" 2> /dev/null
    add "$row" "cpu build" "$([ $rc = 0 ] && [ -x "$B/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; ${c:0:9}; static, every target; $(grep -ac 'warning:' "$B/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$B/build.log") error lines"
}
if step cpu; then
    mkdir -p "$C"
    cpu_build "$C" "$TIP" cpu
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
KT_ARMS=("ro8kt1:${P}_RO8=1 ${P}_RO8_KT=1" "ro8kt2:${P}_RO8=1 ${P}_RO8_KT=2"
         "all:${P}_DMT=1 ${P}_DMT_KT=1 ${P}_RO8=1 ${P}_RO8_KT=1")

if step off; then
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "${KT_ARMS[@]}"
    rm -f "$M"/off/base/logits-*.f32
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c794=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
    F=$(git -C "$SRC" rev-parse "${FINAL_REF:?FINAL_REF=<the merge commit, in the clone>}^{commit}") || { add merged "$FINAL_REF" FAIL "not in $SRC"; date -Is > "$M/DONE"; exit 1; }
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); mmvq-pascal-dmt.cu $(git -C "$SRC" diff --quiet "$TIP" "$F" -- ggml/src/ggml-cuda/mmvq-pascal-dmt.cu && echo "the same as" || echo "differs from") ${TIP:0:9}'s; files against ${TIP:0:9}: $(git -C "$SRC" diff --stat "$TIP" "$F" | tail -n 1)"
    OB=$M/obj
    if git -C "$SRC" diff --quiet "$TIP" "$F" -- ggml/src/ggml-cuda/mmvq-pascal-dmt.cu ggml/src/ggml-cuda/mmvq-pascal-dmt.cuh; then
        add merged "mmvq-pascal-dmt.cu" SAME "the file and its header are the tip's: the obj step stands"
    else
        obj_one "$OB" "$F" merged
        obj_cmp "$OB" tip merged merged
    fi
    MC=$M/merged
    mkdir -p "$MC"
    cpu_build "$MC" "$F" merged
    rm -rf "$MC/tree"
    archs_base "$M/moff"
    archs_arms "$M/moff" "$MC/bin" "merged:" "${KT_ARMS[2]}"
    rm -f "$M"/moff/base/logits-*.f32
fi

log "done"
date -Is > "$M/DONE"
