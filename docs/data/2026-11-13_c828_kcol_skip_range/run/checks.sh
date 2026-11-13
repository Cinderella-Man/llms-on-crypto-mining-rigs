#!/usr/bin/env bash
# C828 (2026-11-13): GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP only at 1 to 4 tokens (fork branch pascal-p106-v4-b12360-c828 on
# v4-rc1 da2920d45), built once and checked on a build machine, CPU only. The method is C813's
# (docs/data/2026-11-12_c813_hrow_half/run/checks.sh) with a routing step. Every build and test runs in a container
# without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19).
# Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1 EXTRA_TARGETS="test-cuda-keepalive
#           test-llama-archs", as v4-rc1's) in $M/main/rig; its build tree stays for the route step
#   route   run/kcol_skip_route.cpp linked against the rig build's static libraries, run with the switch unset, at the
#           preset's value q4_0,q5_0 and with seven types listed: the predicate the k-column hook asks, per type and
#           1 to 8 tokens, against what C828 means (a listed type skipped at 1 to 4 tokens only)
#   bitid   bench/mmvq_pascal/verify/build_multitoken_bitid.sh against the rig build's tree (the binary R155 runs on a
#           P106, copied to $M/bitid) and C755's check, BITID_CPU=1 multitoken_bitid shdown on ggml's CPU backend: 72
#           comparisons (2 types x 4 shapes x 2-4 tokens x 3 inputs), all bit-identical
#   sass    cuobjdump -sass and -ptx of bin/llama-server of the new tarball and of BASE_TREE (v4-rc1's rig tarball),
#           anonymous-namespace hashes masked, cut into functions (the day merge's sass_attrib.py); where the PTX
#           files differ, also with the numbers after internal names' hashes masked (C779's ptx_names.sh)
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target); test-llama-archs in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch and with the switch at q4_0,q5_0 (a CUDA switch: the CPU results must be the base's)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries, the rig tarball and its tree stay)
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (da2920d45 in full), BASE_TREE (v4-rc1's
#   unpacked rig tarball), BASE_CPU (bin/ of da2920d45's static CPU build: test-llama-archs), SEED_SRC (a llama.cpp
#   clone with the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default
#   pascal-p106-v4-b12360-c828), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/route/*.tsv, $M/sass, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12360-c828}
BASE_COMMIT=${BASE_COMMIT:-da2920d45cbb3630b6b6cceba215fcf49e0717b1}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-rig route bitid sass cpu off c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-13_c828_kcol_skip_range/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
# the rig build's binaries load libcuda; a container without GPUs takes the toolkit's stub (C703)
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
SW=GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c828.log"; }
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
log "C828 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1); $(git -C "$SRC" diff --name-only "$BASE_COMMIT" "$TIP" | tr '\n' ' ')"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" JOBS="$JOBS" DOCKER_ARGS="$DA" \
        nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); llama-server md5 $(md5sum < "$T/bin/llama-server" | cut -d' ' -f1); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); test-cuda-keepalive md5 $(md5sum < "$T/bin/test-cuda-keepalive" | cut -d' ' -f1); test-llama-archs md5 $(md5sum < "$T/bin/test-llama-archs" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    (cd "$T/bin" && md5sum ./* > "$M/logs/bin.md5")
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step route; then
    R=$M/route
    B=$SRC/build
    rm -rf "$R" && mkdir -p "$R"
    cp "$RUN/kcol_skip_route.cpp" "$R/"
    box "$IMAGE" bash -c '
        libs=$(find '"$B"'/ggml -name "libggml*.a" | sort | tr "\n" " ")
        echo "libs: $libs"
        nm -C '"$B"'/ggml/src/ggml-cuda/libggml-cuda.a 2> /dev/null | grep -c " T ggml_cuda_mmvq_pascal_kcol_skips" > '"$R"'/symbol.txt
        nice -n 19 g++ -std=c++17 -O1 -I'"$SRC"'/ggml/include '"$R"'/kcol_skip_route.cpp -o '"$R"'/kcol_skip_route \
            -Wl,--start-group $libs -Wl,--end-group -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
            -lcudart -lcublas -lcublasLt -lcuda -fopenmp -ldl -lpthread -lm' > "$R/build.log" 2>&1
    rc=$?
    add route build "$([ $rc = 0 ] && [ -x "$R/kcol_skip_route" ] && echo OK || echo FAIL)" \
        "rc=$rc; ggml_cuda_mmvq_pascal_kcol_skips defined in libggml-cuda.a: $(cat "$R/symbol.txt" 2> /dev/null); $(grep -acE 'error|undefined reference' "$R/build.log") error lines"
    ALL=q4_0,q5_0,q8_0,q2_0,q3_K,q4_K,iq4_xs
    for arm in "unset:" "preset:q4_0,q5_0" "all:$ALL"; do
        tag=${arm%%:*} val=${arm#*:}
        # the image's entrypoint prints a banner on stdout: the table goes to a file inside the container
        box "$IMAGE" bash -c "$STUB"'
            '"$([ -n "$val" ] && echo "export $SW=$val")"'
            nice -n 19 '"$R"'/kcol_skip_route > '"$R/route_$tag.tsv"' 2> '"$R/route_$tag.err" > "$R/route_$tag.out" 2>&1
        rc=$?
        # what C828 means: a listed type is skipped at 1 to 4 tokens only; nothing is skipped with the switch unset
        bad=$(awk -F'\t' -v list=",$val," 'NR > 1 { want = (index(list, "," $2 ",") > 0 && $3 <= 4) ? "yes" : "no"
                  if ($4 != want) print $2 "@" $3 }' "$R/route_$tag.tsv" | tr '\n' ' ')
        n=$(($(wc -l < "$R/route_$tag.tsv") - 1))
        yes=$(awk -F'\t' 'NR > 1 && $4 == "yes" { print $2 "@" $3 }' "$R/route_$tag.tsv" | tr '\n' ' ')
        k=$(awk -F'\t' 'NR > 1 && ($2 == "q4_0" || $2 == "q5_0") { print $3 ":" $5 }' "$R/route_$tag.tsv" | sort -u -t: -k1,1n | tr '\n' ';')
        add route "${val:-unset}" "$([ "$rc" = 0 ] && [ "$n" = 56 ] && [ -z "$bad" ] && echo OK || echo FAIL)" \
            "rc=$rc; $n rows (7 types x 8 token counts); not as designed: ${bad:-none}; skipped: ${yes:-none}; Q4_0 and Q5_0 tokens:kernel ${k}"
    done
    log "route done"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    BD=$M/bitid
    rm -rf "$BD" && mkdir -p "$BD"
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$BD/build.log" 2>&1
    rc=$?
    cp "$X" "$BD/" 2> /dev/null
    add bitid build "$([ $rc = 0 ] && [ -x "$BD/multitoken_bitid" ] && grep -aqw shdown "$BD/multitoken_bitid" && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$BD/multitoken_bitid" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$BD/multitoken_bitid" 2> /dev/null) B; linked against the rig build of ${TIP:0:9}; $(grep -ac warning "$BD/build.log") warning lines"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$BD:$BD" "$IMAGE" bash -c "$STUB
        BITID_CPU=1 nice -n 19 $BD/multitoken_bitid shdown > $BD/cpu_shdown.txt 2>&1"
    rc=$?
    n=$(grep -c ': bit-identical\|: DIFFERENT' "$BD/cpu_shdown.txt")
    d=$(grep -c ': DIFFERENT' "$BD/cpu_shdown.txt")
    add bitid "cpu shdown" "$([ $rc = 0 ] && [ "$n" = 72 ] && [ "$d" = 0 ] && grep -q '^backend: CPU' "$BD/cpu_shdown.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $n product lines (expected 72: 2 types x 4 shapes x 2-4 tokens x 3 inputs), $d differ; $(tail -n 1 "$BD/cpu_shdown.txt")"
    log "bitid done"
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    # <label> <llama-server>: its SASS and PTX, the masked listing kept and cut into functions
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" > "$S/$1.$w.txt"
            nice -n 19 python3 "$ATTRIB" dump $w < "$S/$1.$w.txt" > "$S/$1.$w.tsv"
        done
    }
    dumpbin c828 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c828.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c828.$w.txt" "$S/base.$w.txt" && echo "the same file" || echo "files differ")
        num=""
        if [ "$same" != "the same file" ]; then
            # C779's ptx_names.sh: the number nvcc puts after an internal name's hash, masked too
            cmp -s <(sed -E 's/_cu_X_[0-9]+/_cu_X_N/g' "$S/c828.$w.txt") <(sed -E 's/_cu_X_[0-9]+/_cu_X_N/g' "$S/base.$w.txt") \
                && num="; with the numbers after internal names' hashes masked: the same file" \
                || num="; with the numbers after internal names' hashes masked: files differ"
            diff "$S/base.$w.txt" "$S/c828.$w.txt" | grep '^[<>]' | head -n 200 > "$S/diff_$w.txt"
        fi
        add sass "llama-server $w against v4-rc1's" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && { [ "$same" = "the same file" ] || [[ "$num" == *"the same file" ]]; } && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c828.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); masked dumps: $same$num; md5 $(md5sum < "$S/c828.$w.txt" | cut -c1-8) and $(md5sum < "$S/base.$w.txt" | cut -c1-8); $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    for f in "$S"/c828.*.txt "$S"/base.*.txt; do gzip -f "$f"; done
    log "sass done"
fi

C=$M/cpu
if step cpu; then
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then mkdir -p "$C" && cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    rm -rf "$C/tree" "$C/bin" && mkdir -p "$C/tree/src" "$C/bin"
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

if step off; then
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "skip:${SW}=q4_0,q5_0"
    rm -f "$M"/off/base/logits-*.f32
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c828=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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

log "done"
date -Is > "$M/DONE"
