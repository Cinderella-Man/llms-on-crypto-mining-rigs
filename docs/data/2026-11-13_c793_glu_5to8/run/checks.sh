#!/usr/bin/env bash
# checks.sh: C793 (Q1388) on a build machine: GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 and 4, C757's one launch of a gate, an up
# and their SWIGLU also for the pairs with an IQ4_XS, Q4_K or Q5_K product at 5-8 tokens (fork branch
# pascal-p106-v4-b12246-c793), built and checked. Every build and test runs in a container at a low CPU share
# (--cpu-shares=256, nice 19) and waits for LOCK (flock), so one runs at a time on a shared machine. Only the ptx step
# uses a GPU (GPU=<index>), and only when it is in STEPS. C757's checks.sh (docs/data/2026-11-11_c757_shexp_fused) with
# C813's rig, sass, c565 and merged steps (docs/data/2026-11-12_c813_hrow_half).
# Steps (STEPS, in this order by default; ptx and merged are run on their own):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1 EXTRA_TARGETS=test-llama-archs: sm_61, no
#           AVX, test-backend-ops and test-llama-archs in bin/) in $M/main/rig; its md5
#   sass    bin/llama-server's SASS and PTX against BASE_TREE's (the base's rig tree), cut into functions (the day merge's
#           sass_attrib.py), anonymous namespace hashes masked: every base function the same, the new ones
#           mul_mat_vec_q_pascal_glu_ro8 only; the GLU object's register use, stack and local memory and its SWIGLU
#           against the separate GLU kernel's (run/sass_check.sh)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the tip's rig build tree, copied next to the tarball
#   record  the replay set glumt (scripts/record_checks.sh SUITE=glumt, LINK_FROM=GLUMT_SET: C757's cases, the same
#           filter) with the tarball's test-backend-ops: the runs of C757 and C793
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree
#   off     test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, the tip's compares, with no
#           switch and with GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 GGML_CUDA_MMVQ_PASCAL_RO8=1 (CUDA switches: the same results)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed
#   ptx     multitoken_bitid glu through the compute-6.1 PTX on GPU (CUDA_FORCE_PTX_JIT=1, GGML_CUDA_MMVQ_PASCAL=2), with
#           the separate kernels it repeats (_DMT=1 _DMT_R=1 _Q5K_MT=1, the preset's k-column switches): at _GLU_MT=3
#           with _RO8=1, at 4 with _RO8=2, at 3 with _NP=1, and the fault (_GLU_MT_FAULT=1, which must differ)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip and
#           test-llama-archs against the base's logits with no switch and with the switch at 3
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT, BASE_TREE (the base's rig tarball tree),
#   BASE_CPU (bin/ of a static CPU build with the base's CPU code: test-llama-archs), SEED_SRC (a llama.cpp clone with the
#   fork as remote "fork", https), SEED_CCACHE and CPU_CCACHE_SEED (optional), GLUMT_SET (C757's complete set folder
#   sets/glumt), BRANCH
#   (default pascal-p106-v4-b12246-c793), FINAL_BRANCH (default pascal-p106-v4), GPU (ptx), JOBS (8), THREADS (8),
#   DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/ptx, $M/sets, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c793}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
: "${BASE_COMMIT:?BASE_COMMIT=<the base commit>}"
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-rig sass bitid record cpu off c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
HERE=$(cd "$(dirname "$0")" && pwd)
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
P=GGML_CUDA_MMVQ_PASCAL
SW=${P}_GLU_MT
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c793.log"; }
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
log "C793 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1); $(git -C "$SRC" diff --name-only "$BASE_COMMIT" "$TIP" | tr '\n' ' ')"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 EXTRA_TARGETS=test-llama-archs JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); llama-server md5 $(md5sum < "$T/bin/llama-server" | cut -d' ' -f1); test-llama-archs in bin/: $([ -x "$T/bin/test-llama-archs" ] && echo yes || echo no); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    # <label> <llama-server>: its SASS and PTX cut into functions (cuobjdump in the CUDA image, the cutting on the host)
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin c793 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        # per function: the same, differs, new (only in C793's build), missing (only in the base's)
        awk -F'\t' 'BEGIN { OFS = "\t" } NR == FNR { b[$1] = $2; next }
            { if ($1 in b) { print $1, ($2 == b[$1] ? "same" : "DIFFERS"); delete b[$1] } else print $1, "new" }
            END { for (f in b) print f, "MISSING" }' "$S/base.$w.tsv" "$S/c793.$w.tsv" | sort > "$S/funcs_$w.tsv"
        n=$(wc -l < "$S/funcs_$w.tsv")
        same=$(awk -F'\t' '$2 == "same"' "$S/funcs_$w.tsv" | wc -l)
        new=$(awk -F'\t' '$2 == "new"' "$S/funcs_$w.tsv" | wc -l)
        newo=$(awk -F'\t' '$2 == "new" && $1 !~ /mul_mat_vec_q_pascal_glu_ro8/' "$S/funcs_$w.tsv" | wc -l)
        bad=$(awk -F'\t' '$2 == "DIFFERS" || $2 == "MISSING"' "$S/funcs_$w.tsv" | wc -l)
        add sass "llama-server $w against the base" "$([ "$n" -gt 0 ] && [ "$bad" = 0 ] && [ "$newo" = 0 ] && [ "$new" -gt 0 ] && echo OK || echo CHECK)" \
            "$n functions ($(wc -l < "$S/base.$w.tsv") in the base's): $same the same, $new new ($newo not mul_mat_vec_q_pascal_glu_ro8), $bad differ or missing: $(awk -F'\t' '$2 == "DIFFERS" || $2 == "MISSING" { printf "%s ", substr($1, 1, 80) }' "$S/funcs_$w.tsv" | cut -c1-600)"
    done
    O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
        cuobjdump -res-usage '"$O"'/mmvq-pascal-glu.cu.o > '"$S"'/glu_res.txt 2>&1
        cuobjdump -sass '"$O"'/mmvq-pascal-glu.cu.o > '"$S"'/glu.sass 2>&1
        cuobjdump -sass '"$O"'/unary.cu.o > '"$S"'/unary.sass 2>&1
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$M/logs/sass.log" 2>&1
    SASS=$S bash "$HERE/sass_check.sh" > "$S/sass_counts.tsv" 2> "$S/sass_check.err"
    k=$(awk -F'\t' 'NR > 1' "$S/sass_counts.tsv" | wc -l)
    k8=$(awk -F'\t' 'NR > 1 && $1 ~ /_ro8$/' "$S/sass_counts.tsv" | wc -l)
    spills=$(awk -F'\t' 'NR > 1 && ($5 != 0 || $6 != 0)' "$S/sass_counts.tsv" | wc -l)
    add sass "GLU kernels" "$([ "$k" = 22 ] && [ "$k8" = 8 ] && [ "$spills" = 0 ] && echo OK || echo CHECK)" \
        "$k instantiations ($k8 mul_mat_vec_q_pascal_glu_ro8), $spills with stack or local memory; registers of the new: $(awk -F'\t' 'NR > 1 && $1 ~ /_ro8$/ { printf "NT %s PF %s %s; ", $2, $3, $4 }' "$S/sass_counts.tsv")the others $(awk -F'\t' 'NR > 1 && $1 !~ /_ro8$/ { print $4 }' "$S/sass_counts.tsv" | sort -n | sed -n '1p;$p' | tr '\n' '-' | sed 's/-$//')"
    bad=$(awk -F'\t' 'NR > 1 { split($NF, a, " "); if (a[1] != a[3] || a[3] == 0) print }' "$S/sass_counts.tsv" | wc -l)
    add sass swiglu "$([ "$k" -gt 0 ] && [ "$bad" = 0 ] && echo SAME || echo DIFFERS)" \
        "every instantiation's SWIGLU against unary_gated_op_kernel<op_silu, float>: $(awk -F'\t' 'NR > 1 { split($NF, a, " "); s += a[1]; t += a[3] } END { print s " of " t }' "$S/sass_counts.tsv") the same op sequence (swiglu.txt)"
    log "sass done"
fi

if step bitid; then
    locked env DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$SRC" "$SRC" \
        > "$M/logs/bitid_build.log" 2>&1
    rc=$?
    B=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    cp "$B" "$M/main/rig/multitoken_bitid" 2> /dev/null
    add bitid build "$([ $rc = 0 ] && [ -x "$M/main/rig/multitoken_bitid" ] && echo OK || echo FAIL)" \
        "rc=$rc; md5 $(md5sum < "$M/main/rig/multitoken_bitid" 2> /dev/null | cut -d' ' -f1); takes functions defined: $(nm "$M/main/rig/multitoken_bitid" 2> /dev/null | grep -cE ' T .*ggml_cuda_mmvq_pascal_(dmt|q5k_mt)_takes')"
fi

if step record; then
    mkdir -p "$M/sets"
    locked env OUT="$M/sets" SUITE=glumt BIN="$T/bin" JOBS="$JOBS" LINK_FROM="${GLUMT_SET:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$M/logs/record.log" 2>&1
    rc=$?
    add record glumt "$([ $rc = 0 ] && [ -f "$M/sets/glumt.tar" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(tail -n 1 "$M/sets/glumt.log" 2> /dev/null); cases $(wc -l < "$M/sets/glumt/manifest.tsv" 2> /dev/null), runs $(wc -l < "$M/sets/glumt/runs.tsv" 2> /dev/null); glumt.tar $(stat -c %s "$M/sets/glumt.tar" 2> /dev/null) B, md5 $(cut -d' ' -f1 "$M/sets/glumt.tar.md5" 2> /dev/null)"
fi

# test-llama-archs: <dir> <tag> <bin dir> <env>: the base writes, the others compare
archs() {
    local A=$1 tag=$2 bin=$3 envs=$4
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$bin:$bin:ro" "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
    echo $?
}
off_check() { # <dir> <row> <bin dir> <arm:env> ...
    local A=$1 row=$2 bin=$3 arm tag envs rc n bad
    shift 3
    rm -rf "$A" && mkdir -p "$A"
    rc=$(archs "$A" base "$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")" '')
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    add "$row" "archs base" "$([ "$rc" = 0 ] && [ "$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; $(ls "$A"/logits-*.f32 2> /dev/null | wc -l) logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    for arm in "$@"; do
        tag=${arm%%:*} envs=${arm#*:}
        rc=$(archs "$A" "$tag" "$bin" "$envs")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        add "$row" "archs $tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -f "$A"/*.f32
}
cpu_build() { # <dir> <commit>: a static native CPU build, test-llama-archs in <dir>/bin
    local C=$1 c=$2 rc
    rm -rf "$C/tree" "$C/bin" && mkdir -p "$C/tree/src" "$C/bin"
    [ -d "$C/ccache" ] || cp -al "${CPU_CCACHE_SEED:-/nonexistent}" "$C/ccache" 2> /dev/null || mkdir -p "$C/ccache"
    git -C "$SRC" archive "$c" | tar -x -C "$C/tree/src"
    box -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$C/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$C/tree/src"' -B '"$C/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$C/tree/build"' -j '"$JOBS"' -- -k > '"$C/build.log"' 2>&1'
    rc=$?
    cp "$C/tree/build/bin/test-llama-archs" "$C/bin/" 2> /dev/null
    echo "$rc"
}

C=$M/cpu
if step cpu; then
    rc=$(cpu_build "$C" "$TIP")
    add cpu build "$([ "$rc" = 0 ] && [ -x "$C/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; ${TIP:0:9}; static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

if step off; then
    off_check "$M/off" off "$C/bin" "tip:" "glu3:$SW=3 ${P}_RO8=1" "glu4:$SW=4 ${P}_RO8=2"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c793=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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

if step ptx; then
    # multitoken_bitid glu through the compute-6.1 PTX of the tip's build on one GPU (the CUDA runtime of the image)
    X=$M/ptx
    mkdir -p "$X"
    : "${GPU:?GPU=<the GPU index for the ptx step>}"
    BIN=$M/main/rig/multitoken_bitid
    KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"
    RO="${P}_DMT=1 ${P}_DMT_R=1 ${P}_Q5K_MT=1"
    for r in "on:$KP $RO ${P}_RO8=1 $SW=3" "nopf:$KP $RO ${P}_RO8=2 $SW=4" "np1:$KP $RO ${P}_RO8=1 $SW=3 ${SW}_NP=1" \
             "fault:$KP $RO ${P}_RO8=1 $SW=3 ${SW}_FAULT=1"; do
        tag=${r%%:*} envs=${r#*:}
        # shellcheck disable=SC2086
        locked docker run --rm --label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-} --network none --gpus "device=$GPU" \
            -v "$M:$M" -v "$SRC:$SRC:ro" "$IMAGE" bash -c \
            'env LD_LIBRARY_PATH='"$SRC"'/cuda12-libs:/usr/local/cuda/lib64 CUDA_FORCE_PTX_JIT=1 '"$P"'=2 '"$envs"' nice -n 19 '"$BIN"' glu > '"$X/$tag.log"' 2>&1; echo "rc=$?" >> '"$X/$tag.log"
        sum=$(grep -a 'comparisons, ' "$X/$tag.log" | tail -n 1)
        nd=$(echo "$sum" | sed -nE 's/.*comparisons, ([0-9]+) with differences.*/\1/p')
        # the new pairs at 5-8 tokens at the shared expert's shape (2560 -> 640, every one taken with _RO8): lines whose pair
        # has an IQ4_XS, Q4_K or Q5_K product
        new=$(grep -aE '^GLU/(apart|1-tok) +[^ ]*(q4_K|q5_K|iq4_xs)[^ ]* +n=[5-8] m=640 ' "$X/$tag.log")
        newn=$(echo "$new" | grep -c . ) newd=$(echo "$new" | grep -c DIFFERENT)
        if [ "$tag" = fault ]; then
            v=$([ "${nd:-0}" -gt 0 ] && [ "$newn" -gt 0 ] && [ "$newd" = "$newn" ] && echo CAUGHT || echo MISSED)
        else
            v=$([ -n "$nd" ] && [ "$nd" = 0 ] && echo SAME || echo DIFFERS)
        fi
        add ptx "glu $tag" "$v" "$envs; ${sum:-no summary}; new pairs at 5-8 tokens: $newn lines, $newd differ; $(grep -a '^rc=' "$X/$tag.log")"
    done
    # the set's C793 runs replayed through the PTX (the fault run must fail its cases)
    R8RUNS="glumt8-on glumt8-nopf glumt8-np1 glumt8-ro8only glumt8-ro8-off glumt8-m1 glumt8-off glumt8-fault"
    # shellcheck disable=SC2086
    locked docker run --rm --label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-} --network none --gpus "device=$GPU" \
        -v "$M:$M" "$IMAGE" bash -c 'env LD_LIBRARY_PATH=/usr/local/cuda/lib64 SET='"$M/sets/glumt"' BIN='"$T/bin"' \
            OUT='"$X/replay"' DEV=0 RUNS="'"$R8RUNS"'" ENV_ALL="'"$P"'=2 CUDA_FORCE_PTX_JIT=1" nice -n 19 \
            bash '"$REPO"'/scripts/replay_checks.sh' > "$X/replay-console.txt" 2>&1
    rc=$?
    add ptx "replay glumt8" "$([ $rc = 0 ] && ! grep -q WRONG "$X/replay/results.tsv" 2> /dev/null && [ "$(wc -l < "$X/replay/results.tsv" 2> /dev/null)" -ge 8 ] && echo OK || echo CHECK)" \
        "rc=$rc; $(awk -F'\t' '{ printf "%s %s %s; ", $1, $4, $8 }' "$X/replay/results.tsv" 2> /dev/null)"
fi

if step merged; then
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); merges since the base: $(git -C "$SRC" rev-list --merges --count "$BASE_COMMIT..$F")"
    MC=$M/merged
    [ -d "$MC/ccache" ] || cp -al "$C/ccache" "$MC/ccache" 2> /dev/null || true
    rc=$(cpu_build "$MC" "$F")
    add merged "cpu build" "$([ "$rc" = 0 ] && [ -x "$MC/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; ${F:0:9}; static, every target; $(grep -ac 'warning:' "$MC/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$MC/build.log") error lines"
    rm -rf "$MC/tree"
    off_check "$M/moff" merged "$MC/bin" "merged:" "merged-glu3:$SW=3 ${P}_RO8=1"
fi

log "done"
date -Is > "$M/DONE"
