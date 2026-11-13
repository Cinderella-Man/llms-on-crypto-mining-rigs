#!/usr/bin/env bash
# C826 (2026-11-13): GGML_CUDA_GRAPHS_PATCH (C779, off by default) compares a node's sources by kind, leaves a memset or
# memcpy whose executable holds its new parameters as it is, and names every refused update; on pascal-p106-v4
# 2144cf760, built once and checked on a build machine, CPU only. The driver is C779's
# (docs/data/2026-11-12_c779_graph_replay/run/checks.sh) with a ptx step. Every build and test runs in a container
# without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on
# a shared machine.
# Steps (STEPS, in this order by default):
#   cu      ggml-cuda.cu alone, compiled as the rig build compiles it (sm_61), in the CUDA image: a compile error stops
#           the chain before the long builds; the warning lines that name ggml-cuda.cu or graph-patch.h are counted
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in
#           bin/) in $M/main/rig
#   names   C770's switch list in the build (src/build/common/switch-names.cpp): its count, and GGML_CUDA_GRAPHS_PATCH,
#           GGML_CUDA_GRAPHS_PATCH_WINDOW and GGML_CUDA_GRAPHS_PATCH_MAX in it, the first also in bin/llama-server
#   sass    the change is host code only: cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE
#           (C822's checked rig tarball, cc97bb6b6, whose tree is 2144cf760's), anonymous-namespace hashes masked, cut
#           into functions (the day merge's sass_attrib.py): every function the base's, none missing or added
#   ptx     C779's ptx_names.sh: both PTX dumps whole, with the numbers in internal names masked too
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; test-llama-archs,
#           llama-server, test-cuda-graph-patch and test-cuda-graph-force in $M/cpu/bin
#   gp      test-cuda-graph-patch on the CPU, twice (the dirty groups, the chain's cut into groups, the decision, and the
#           model of the recording capture and the patch with its control), and test-cuda-graph-force once
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch, with GGML_CUDA_GRAPHS_FORCE=1, with the force and patch switches, and with both at window 0
#           and a 100 % limit (CUDA switches: the CPU results must be the base's)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip,
#           test-cuda-graph-patch, and test-llama-archs against the base's logits with no switch and with the force and
#           patch switches (the "every switch off" check of the merged tree; MERGED_STEPS="build off")
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (2144cf760 in full), BASE_TREE (cc97bb6b6's
#   unpacked rig tarball), BASE_CPU (bin/ of a static CPU build whose CPU code is 2144cf760's, with test-llama-archs:
#   65ac9a47d's, to which 2144cf760 adds CUDA host code only), SEED_SRC (a llama.cpp clone with the fork as remote
#   "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default pascal-p106-v4-b12246-c826),
#   FINAL_BRANCH (default pascal-p106-v4), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c826}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-2144cf7600aefd2b6810c14d9101553f9264592e}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-cu cpu gp rig names sass ptx off c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
SW=GGML_CUDA_GRAPHS_PATCH
FORCE="GGML_CUDA_GRAPHS_FORCE=1"
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c826.log"; }
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
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$M/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
log "C826 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1)"

if step cu; then
    # the changed CUDA source alone, as the rig build compiles it (sm_61, the same CMake options), so that a compile
    # error stops the chain in minutes
    U=$M/cu
    rm -rf "$U" && mkdir -p "$U/src"
    git -C "$SRC" archive "$TIP" | tar -x -C "$U/src"
    # shellcheck disable=SC2086
    locked docker run --rm $DA -v "$U:$U" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq cmake build-essential > /dev/null &&
        cd '"$U/src"' && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF \
            -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_CURL=OFF > /dev/null &&
        cd build/ggml/src/ggml-cuda && nice -n 19 make ggml-cuda.cu.o' > "$M/logs/cu.log" 2>&1
    rc=$?
    add cu "ggml-cuda.cu alone" "$([ $rc = 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -acE 'warning' "$M/logs/cu.log") warning lines, $(grep -aE 'warning' "$M/logs/cu.log" | grep -acE 'ggml-cuda\.cu|graph-patch\.h') of them naming ggml-cuda.cu or graph-patch.h; $(grep -aE 'error' "$M/logs/cu.log" | head -n 3 | cut -c1-300 | tr '\n' ';')"
    rm -rf "$U"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); llama-server md5 $(md5sum < "$T/bin/llama-server" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step names; then
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    inl=0
    for s in "$SW" "${SW}_WINDOW" "${SW}_MAX"; do
        inl=$((inl + $(grep -c "^    \"$s\",\$" "$L" 2> /dev/null)))
    done
    inb=$(grep -ac "$SW" "$T/bin/llama-server")
    add names "switch list" "$([ -n "$n" ] && [ "$inl" = 3 ] && [ "$inb" -gt 0 ] && echo OK || echo FAIL)" \
        "$n names in the build's list; $SW, ${SW}_WINDOW and ${SW}_MAX in it: $inl of 3; lines of bin/llama-server holding $SW: $inb"
fi

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
    dumpbin c826 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c826.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c826.$w.tsv" "$S/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c826.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    log "sass done"
fi

if step ptx; then
    P=$M/sass/ptxnames
    NEW="$T/bin/llama-server" BASE="${BASE_TREE:?}/bin/llama-server" OUT="$P" LOCK="$LOCK" IMAGE="$IMAGE" nice -n 19 \
        bash "$D/2026-11-12_c779_graph_replay/run/ptx_names.sh" > "$M/logs/ptx.log" 2>&1
    add ptx "llama-server PTX, internal names' numbers masked" "$(grep -q '^verdict: SAME' "$P/ptx_names.txt" 2> /dev/null && echo SAME || echo DIFFERS)" \
        "$(grep -E '^(lines|differing lines|masked md5)' "$P/ptx_names.txt" 2> /dev/null | tr '\n' ';'); names: $(sed -n '/^their internal names/,/^masked/p' "$P/ptx_names.txt" 2> /dev/null | grep -vE '^(their|masked)' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    rm -f "$P"/new.ptx "$P"/base.ptx
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
    cp "$C/tree/build/bin/test-llama-archs" "$C/tree/build/bin/llama-server" "$C/tree/build/bin/test-cuda-graph-patch" \
        "$C/tree/build/bin/test-cuda-graph-force" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && [ -x "$C/bin/llama-server" ] && [ -x "$C/bin/test-cuda-graph-patch" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cut -c1-140 "$C/tree.txt"); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

if step gp; then
    for i in 1 2; do
        box "$CPU_IMAGE" bash -c 'nice -n 19 '"$C"'/bin/test-cuda-graph-patch > '"$M"'/logs/gp'"$i"'.log 2>&1'
        rc=$?
        add gp "test-cuda-graph-patch run $i" "$([ $rc = 0 ] && grep -q ', 0 failed' "$M/logs/gp$i.log" && echo OK || echo FAIL)" \
            "rc=$rc; $(tail -n 2 "$M/logs/gp$i.log" | tr '\n' ' '); $(grep -c 'failed:' "$M/logs/gp$i.log") failed checks named"
    done
    cmp -s "$M/logs/gp1.log" "$M/logs/gp2.log"
    add gp "the two runs' output" "$([ $? = 0 ] && echo SAME || echo DIFFERS)" "the model's runs are seeded: the same counts both times"
    box "$CPU_IMAGE" bash -c 'nice -n 19 '"$C"'/bin/test-cuda-graph-force > '"$M"'/logs/gf.log 2>&1'
    rc=$?
    add gp "test-cuda-graph-force" "$([ $rc = 0 ] && grep -q ', 0 failed' "$M/logs/gf.log" && echo OK || echo FAIL)" \
        "rc=$rc; $(tail -n 1 "$M/logs/gf.log")"
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
    archs_arms "$M/off" "$C/bin" "tip:" "force:$FORCE" "patch:$FORCE ${SW}=1" \
        "patch-w0:$FORCE ${SW}=1 ${SW}_WINDOW=0 ${SW}_MAX=100"
    rm -f "$M"/off/base/logits-*.f32
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c826=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
        cp "$MC/tree/build/bin/test-llama-archs" "$MC/tree/build/bin/test-cuda-graph-patch" "$MC/bin/" 2> /dev/null
        add merged "cpu build" "$([ $rc = 0 ] && [ -x "$MC/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; ${F:0:9}; static, every target; $(grep -ac 'warning:' "$MC/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$MC/build.log") error lines"
        box "$CPU_IMAGE" bash -c 'nice -n 19 '"$MC"'/bin/test-cuda-graph-patch > '"$M"'/logs/merged-gp.log 2>&1'
        rc=$?
        add merged "test-cuda-graph-patch" "$([ $rc = 0 ] && grep -q ', 0 failed' "$M/logs/merged-gp.log" && echo OK || echo FAIL)" \
            "rc=$rc; $(tail -n 1 "$M/logs/merged-gp.log")"
        rm -rf "$MC/tree"
    fi
    if [[ " $MS " == *" off "* ]]; then
        archs_base "$M/moff"
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-patch:$FORCE ${SW}=1"
        rm -f "$M"/moff/base/logits-*.f32
    fi
fi

log "done"
date -Is > "$M/DONE"
