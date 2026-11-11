#!/usr/bin/env bash
# checks.sh: C727 (Q1286) on a build machine, CPU only: Q6_K's 2-4-token products in C722's read-once kernel (fork branch
# pascal-p106-v4-b12246-c727, GGML_CUDA_MMVQ_PASCAL_Q6K_MT) built and checked without a GPU. Every build and test in a
# container (the CPU ones without network) at low CPU priority (--cpu-shares=256, nice 19), one at a time under LOCK.
# From C726's and C767's checks.sh (2026-11-11_c726_q3k_multitoken/run/, 2026-11-11_c767_q5k_one_row/run/). The base is
# pascal-p106-v4-b12246-day (18d0fa309): its rig tarball and the random-model logits its static CPU build wrote.
# Steps (STEPS, in this order by default):
#   build   the rig tarball of BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK/rig, whose
#           src/ and ccache/ it uses; its md5
#   sass    CUDA 12.9's cuobjdump -sass of both tarballs' bin/llama-server (BASE_TARBALL's extracted), -sass and
#           -res-usage of the tip's mmvq-pascal-dmt.cu.o; run/compare_sass.py (every function the base's code but the 12
#           new Q6_K instantiations, none with stack or local memory) and run/sass_check.sh (their instruction counts)
#   cpu     a static native CPU build (GGML_CUDA=OFF) of the tip, every target, in WORK/cpu/tree with its ccache in
#           WORK/cpu/ccache (seeded from CPU_CCACHE_SEED when given)
#   off     test-llama-archs -b CPU -s 1 of the tip against a copy of BASE_LOGITS (every architecture's logits as the
#           base's static CPU build wrote them): with no switch, with GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 and with =1 and its
#           fault (CUDA switches: the CPU results must be the base's)
#   ctest   ctest in the tip's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   record  the replay set q6kmt (scripts/record_checks.sh SUITE=q6kmt) with the tarball's test-backend-ops
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid (mode q6k among them) linked against the tip's build tree
#   emul    run/q6k_emul.c built and run on the CPU: the one-token kernel's and the new kernel's Q6_K arithmetic from the
#           same bytes at Flash-Next's shape and every lane count (q6k_emul.tsv), and the exact-conversion claims over
#           every scale and every reachable integer (q6k_exact.tsv)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE_TARBALL=<the base's rig tarball>
#   BASE_LOGITS=<folder of the base's logits-*.f32> [CPU_CCACHE_SEED=<ccache folder>] [LOCK=<flock file>] [JOBS=8]
#   [DOCKER_ARGS=...] [STEPS=...] bash checks.sh
#   WORK/rig/src must be a llama.cpp clone with the fork as remote "fork" (https fetch URL), as build_llamacpp_docker.sh
#   wants; WORK/rig/ccache may be seeded from an earlier build's.
# Output: WORK/summary.tsv (check, verdict, detail), WORK/sass/ (sass_binary.tsv, sass_dmt.tsv, sass_differ.txt,
#   sass_counts.tsv), WORK/sets/q6kmt.tar, WORK/emul/ (q6k_emul.tsv, q6k_exact.tsv), logs in WORK/logs; done when
#   WORK/C727_DONE exists.
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}"
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass cpu off ctest record bitid emul}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
RUN=$REPO/docs/data/2026-11-11_c727_q6k_readonce/run
SRC=$WORK/rig/src
C=$WORK/cpu
mkdir -p "$WORK/logs" "$WORK/sass" "$C"
rm -f "$WORK/C727_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball() { ls -td "$WORK"/rig/llama.cpp-*-"${BRANCH##*/}" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== c727 checks started $(date -Is), $BRANCH, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    locked env WORK="$WORK/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/logs/build.log" 2>&1
    rc=$?
    T=$(tarball)
    verdict build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$WORK/logs/build.log") warning lines"
    [ $rc = 0 ] || { touch "$WORK/C727_DONE"; exit 1; }
fi
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")

if step sass; then
    echo "=== sass $(date -Is)"
    : "${BASE_TARBALL:?BASE_TARBALL=<the base rig tarball>}"
    T=$(tarball)
    S=$WORK/sass
    B=$WORK/base
    O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-dmt.cu.o
    rm -rf "$B" && mkdir -p "$B"
    tar xzf "$BASE_TARBALL" -C "$B" --wildcards '*/bin/llama-server'
    BB=$(ls -d "$B"/llama.cpp-*/bin)
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        cuobjdump --version | tail -n 1 > '"$S"'/cuobjdump.txt
        nice -n 19 cuobjdump -sass '"$BB"'/llama-server > '"$S"'/base.sass
        nice -n 19 cuobjdump -sass '"$T"'/bin/llama-server > '"$S"'/tip.sass
        nice -n 19 cuobjdump -sass '"$O"' > '"$S"'/dmt.sass
        nice -n 19 cuobjdump -res-usage '"$O"' > '"$S"'/dmt_res.txt 2>&1
        md5sum '"$BB"'/llama-server '"$T"'/bin/llama-server '"$O"' > '"$S"'/md5.txt
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$WORK/logs/sass.log" 2>&1
    rc=$?
    nice -n 19 python3 -I "$RUN/compare_sass.py" "$S/base.sass" "$S/tip.sass" "$S/dmt_res.txt" "$S" > "$S/compare.txt" 2>&1
    rc2=$?
    SASS=$S nice -n 19 bash "$RUN/sass_check.sh" > "$S/sass_counts.tsv" 2> "$S/sass_check.err"
    i2f=$(awk -F'\t' 'NR > 1 && $27 != 0' "$S/sass_counts.tsv" | wc -l)
    xmad=$(awk -F'\t' 'NR > 1 && $30 != 0' "$S/sass_counts.tsv" | wc -l)
    verdict sass "$([ $rc = 0 ] && [ $rc2 = 0 ] && [ "$i2f" = 0 ] && echo SAME || echo CHECK)" \
        "rc=$rc/$rc2; $(tail -n 1 "$S/compare.txt"); main loops with I2F: $i2f, with XMAD (addressing; none in the products): $xmad (of $(($(wc -l < "$S/sass_counts.tsv") - 1))); $(cat "$S/cuobjdump.txt" 2> /dev/null)"
    gzip -f "$S/base.sass" "$S/tip.sass"
    rm -rf "$B"
fi

if step cpu; then
    echo "=== cpu $(date -Is)"
    T=$C/tree
    rm -rf "$T" "$C/bin" && mkdir -p "$T/src" "$C/bin"
    [ -d "$C/ccache" ] || { if [ -n "${CPU_CCACHE_SEED:-}" ]; then cp -a "$CPU_CCACHE_SEED" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$C/tree.txt"
    if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
        printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
            | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$WORK/logs/image.log" 2>&1
    fi
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$CPU_IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$C/build.log"' 2>&1'
    rc=$?
    cp "$T/build/bin/test-llama-archs" "$C/bin/" 2> /dev/null
    verdict cpu "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$C/tree.txt"); all targets; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

if step off; then
    echo "=== off $(date -Is)"
    : "${BASE_LOGITS:?BASE_LOGITS=<folder of the base logits>}"
    A=$WORK/off
    rm -rf "$A" && mkdir -p "$A/base"
    cp "$BASE_LOGITS"/logits-*.f32 "$A/base/"
    nb=$(ls "$A"/base/logits-*.f32 | wc -l)
    for arm in "tip:" "q6kmt:GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1" "q6kmt-fault:GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT_FAULT=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$WORK:$WORK" "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- '"$envs"' nice -n 19 '"$C"'/bin/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        wrote=$(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log")
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && [ "$wrote" = 0 ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; against the base's $nb files: $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -rf "$A/base"
fi

if step ctest; then
    echo "=== ctest $(date -Is)"
    B=$C/tree/build
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$CPU_IMAGE" bash -c \
        "cd $B && nice -n 19 ctest -j $JOBS --timeout 1800 --output-on-failure -E '$CTEST_EXCLUDE' --output-junit $WORK/logs/ctest.xml" \
        > "$WORK/logs/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$WORK/logs/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$WORK/logs/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}; not run: $CTEST_EXCLUDE"
fi

if step record; then
    echo "=== record $(date -Is)"
    T=$(tarball)
    rm -rf "$WORK/sets/q6kmt" "$WORK/sets/q6kmt.tar"
    mkdir -p "$WORK/sets"
    locked env OUT="$WORK/sets" SUITE=q6kmt BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="--network none ${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$WORK/logs/record.log" 2>&1
    rc=$?
    M=$WORK/sets/q6kmt/manifest.tsv
    verdict record "$([ $rc = 0 ] && [ -s "$M" ] && [ -s "$WORK/sets/q6kmt.tar.md5" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(wc -l < "$M" 2> /dev/null) cases, $(wc -l < "$WORK/sets/q6kmt/runs.tsv" 2> /dev/null) runs; q6kmt.tar $(stat -c %s "$WORK/sets/q6kmt.tar" 2> /dev/null) B, md5 $(cut -d' ' -f1 "$WORK/sets/q6kmt.tar.md5" 2> /dev/null)"
fi

if step bitid; then
    echo "=== bitid $(date -Is)"
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$WORK/logs/bitid.log" 2>&1
    rc=$?
    verdict bitid-build "$([ $rc = 0 ] && [ -x "$X" ] && grep -qa 'q6k' "$X" && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), mode q6k in it; linked against ${TIP:0:9}"
fi

if step emul; then
    echo "=== emul $(date -Is)"
    E=$WORK/emul
    rm -rf "$E" && mkdir -p "$E"
    cp "$RUN/q6k_emul.c" "$E/"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$CPU_IMAGE" bash -c \
        "cd $E && gcc -O2 -ffp-contract=off -o q6k_emul q6k_emul.c -lm && nice -n 19 ./q6k_emul rows > q6k_emul.tsv && nice -n 19 ./q6k_emul exact > q6k_exact.tsv" \
        > "$WORK/logs/emul.log" 2>&1
    rc=$?
    d=$(awk -F'\t' 'NR > 1 { o += $7; d += $8; s += $9; x += $10; c1 += $11 > 0; c2 += $12 > 0; n++ }
        END { printf "%d products, %d outputs, %d differ; %d sub-blocks, %d with x0 + x1 or scales differing; controls differ in %d and %d of %d products", n, o, d, s, x, c1, c2, n }' \
        "$E/q6k_emul.tsv" 2> /dev/null)
    ex=$(awk -F'\t' 'NR > 1 { n += $2; b += $3 } END { printf "exact checks: %d cases, %d differ", n, b }' "$E/q6k_exact.tsv" 2> /dev/null)
    verdict emul "$([ $rc = 0 ] && awk -F'\t' 'NR > 1 { if ($8 != 0 || $10 != 0) bad = 1; c1 += $11; c2 += $12 } END { exit bad || c1 == 0 || c2 == 0 }' "$E/q6k_emul.tsv" && awk -F'\t' 'NR > 1 && $3 != 0 { bad = 1 } END { exit bad }' "$E/q6k_exact.tsv" && echo SAME || echo DIFFERS)" \
        "rc=$rc; $d; $ex"
fi

echo "=== done $(date -Is)"
touch "$WORK/C727_DONE"
