#!/usr/bin/env bash
# checks.sh: C757 (Q1333) on a build machine, CPU only: the one launch of a gate, an up and their SWIGLU at 2-8 tokens
# (fork branch pascal-p106-v4-b12246-c757, GGML_CUDA_MMVQ_PASCAL_GLU_MT) built and checked without a GPU. Every build and
# test in a container without GPUs at low CPU priority (--cpu-shares=256, nice 19), one at a time under LOCK. C726's
# checks.sh with this kernel's object and suite.
# Steps (STEPS, in this order by default):
#   build   the rig tarball of BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK, whose
#           src/ and ccache/ it uses; its md5
#   sass    cuobjdump -sass of every object of the CUDA backend in the tip's build tree (WORK/src/build) and in
#           BASE_BUILD (the base commit's rig build tree): per object the code with the anonymous namespaces' hashes
#           masked, the same or not (mmvq-pascal-glu.cu.o, new, has no base); the new kernels' register use and
#           instruction counts, and their SWIGLU against the separate GLU kernel's (run/sass_check.sh)
#   record  the replay set glumt (scripts/record_checks.sh SUITE=glumt) with the tarball's test-backend-ops
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the tip's build tree
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in WORK/cpu/tree
#   off     test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, the tip's compares, with
#           no switch and with GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 (a CUDA switch: the CPU results must be the same)
#   ctest   ctest in the tip's CPU build tree, without the tests that need a download, the network or torch (as C565)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE_BUILD=<base rig build tree, e.g. .../src/build>
#   BASE_CPU=<bin/ of the base commit's static native CPU build> [LOCK=<flock file>] [JOBS=8] [DOCKER_ARGS=...]
#   [CPU_CCACHE_SEED=<a ccache folder to start the CPU build's from>] [STEPS=...] bash checks.sh
#   WORK/src must be a llama.cpp clone with the fork as remote "fork" (https fetch URL), as build_llamacpp_docker.sh wants.
# Output: WORK/summary.tsv (check, verdict, detail), WORK/sass/ (sass_objects.tsv, sass_counts.tsv, swiglu.txt),
#   WORK/sets/glumt.tar, logs in WORK/logs; done when WORK/C757_DONE exists.
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}"
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass record bitid cpu off ctest}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/logs" "$WORK/sass" "$WORK/cpu"
rm -f "$WORK/C757_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball() { ls -td "$WORK"/llama.cpp-*-"${BRANCH##*/}" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== c757 checks started $(date -Is), $BRANCH, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    locked env WORK="$WORK" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/logs/build.log" 2>&1
    rc=$?
    T=$(tarball)
    verdict build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/C757_DONE"; exit 1; }
fi

if step sass; then
    echo "=== sass $(date -Is)"
    : "${BASE_BUILD:?BASE_BUILD=<base rig build tree>}"
    BASE_BUILD=$(realpath "$BASE_BUILD")
    O=CMakeFiles/ggml-cuda.dir
    TIPD=$WORK/src/build/ggml/src/ggml-cuda/$O
    BASED=$BASE_BUILD/ggml/src/ggml-cuda/$O
    S=$WORK/sass
    # one container: every object's SASS, with each function name's anonymous-namespace hash masked, as an md5 per
    # object and side; the new kernels' res-usage and SASS, and the GLU kernels' SASS, kept
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" -v "$BASE_BUILD:$BASE_BUILD:ro" "$IMAGE" bash -c '
        cd '"$TIPD"' && find . -name "*.cu.o" | sort > '"$S"'/objects.txt
        mask() { sed -E "s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g"; }
        while read -r o; do
            t=$(nice -n 19 cuobjdump -sass "'"$TIPD"'/$o" | mask | md5sum | cut -d" " -f1)
            if [ -f "'"$BASED"'/$o" ]; then
                b=$(nice -n 19 cuobjdump -sass "'"$BASED"'/$o" | mask | md5sum | cut -d" " -f1)
            else
                b=-
            fi
            printf "%s\t%s\t%s\n" "${o#./}" "$b" "$t"
        done < '"$S"'/objects.txt > '"$S"'/objects_md5.tsv
        cuobjdump -res-usage '"$TIPD"'/mmvq-pascal-glu.cu.o > '"$S"'/glu_res.txt 2>&1
        cuobjdump -sass '"$TIPD"'/mmvq-pascal-glu.cu.o > '"$S"'/glu.sass 2>&1
        cuobjdump -sass '"$TIPD"'/unary.cu.o > '"$S"'/unary.sass 2>&1
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$WORK/logs/sass.log" 2>&1
    rc=$?
    awk -F'\t' 'BEGIN { OFS = "\t"; print "object", "verdict", "base md5", "tip md5" }
        { print $1, ($2 == "-" ? "new" : ($2 == $3 ? "same" : "DIFFERS")), $2, $3 }' "$S/objects_md5.tsv" > "$S/sass_objects.tsv"
    n=$(tail -n +2 "$S/sass_objects.tsv" | wc -l)
    same=$(awk -F'\t' '$2 == "same"' "$S/sass_objects.tsv" | wc -l)
    new=$(awk -F'\t' '$2 == "new" { printf "%s ", $1 }' "$S/sass_objects.tsv")
    differ=$(awk -F'\t' '$2 == "DIFFERS" { printf "%s ", $1 }' "$S/sass_objects.tsv")
    verdict sass-others "$([ $rc = 0 ] && [ "$n" -gt 0 ] && [ -z "$differ" ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; $n objects: $same the same as the base, new: ${new:-none}; differ: ${differ:-none}"
    SASS=$S bash "$HERE/sass_check.sh" > "$S/sass_counts.tsv" 2> "$S/sass_check.err"
    spills=$(awk -F'\t' 'NR > 1 && ($5 != 0 || $6 != 0)' "$S/sass_counts.tsv" | wc -l)
    k=$(awk -F'\t' 'NR > 1' "$S/sass_counts.tsv" | wc -l)
    i2f=$(awk -F'\t' 'NR > 1 { s += $8 } END { print s + 0 }' "$S/sass_counts.tsv")
    verdict sass-glu "$([ "$k" = 14 ] && [ "$spills" = 0 ] && echo OK || echo CHECK)" \
        "$k instantiations, $spills with stack or local memory; registers $(awk -F'\t' 'NR > 1 { print $4 }' "$S/sass_counts.tsv" | sort -n | sed -n '1p;$p' | tr '\n' '-' | sed 's/-$//'); I2F $i2f in all (the staging's)"
    bad=$(awk -F'\t' 'NR > 1 { split($NF, a, " "); if (a[1] != a[3] || a[3] == 0) print }' "$S/sass_counts.tsv" | wc -l)
    verdict sass-swiglu "$([ "$k" -gt 0 ] && [ "$bad" = 0 ] && echo SAME || echo DIFFERS)" \
        "every instantiation's SWIGLU against unary_gated_op_kernel<op_silu, float>: $(awk -F'\t' 'NR > 1 { split($NF, a, " "); s += a[1]; t += a[3] } END { print s " of " t }' "$S/sass_counts.tsv") the same op sequence (swiglu.txt)"
fi

if step record; then
    echo "=== record $(date -Is)"
    T=$(tarball)
    mkdir -p "$WORK/sets"
    locked env OUT="$WORK/sets" SUITE=glumt BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$WORK/logs/record.log" 2>&1
    rc=$?
    verdict record "$([ $rc = 0 ] && [ -f "$WORK/sets/glumt.tar" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(tail -n 1 "$WORK/sets/glumt.log" 2> /dev/null); manifest lines $(wc -l < "$WORK/sets/glumt/manifest.tsv" 2> /dev/null), runs $(wc -l < "$WORK/sets/glumt/runs.tsv" 2> /dev/null); glumt.tar $(stat -c %s "$WORK/sets/glumt.tar" 2> /dev/null) B, md5 $(cut -d' ' -f1 "$WORK/sets/glumt.tar.md5" 2> /dev/null)"
fi

if step bitid; then
    echo "=== bitid $(date -Is)"
    locked env DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$WORK/src" "$WORK/src" \
        > "$WORK/logs/bitid_build.log" 2>&1
    rc=$?
    B=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    verdict bitid-build "$([ $rc = 0 ] && [ -x "$B" ] && echo OK || echo FAIL)" "rc=$rc; md5 $(md5sum < "$B" 2> /dev/null | cut -d' ' -f1)"
fi

C=$WORK/cpu
if step cpu; then
    echo "=== cpu $(date -Is)"
    TIP=$(git -C "$WORK/src" rev-parse "fork/$BRANCH")
    T=$C/tree
    rm -rf "$T" "$C/bin" && mkdir -p "$T/src" "$C/bin"
    [ -d "$C/ccache" ] || { if [ -n "${CPU_CCACHE_SEED:-}" ]; then cp -a "$CPU_CCACHE_SEED" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    git -C "$WORK/src" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$WORK/src" log -1 --format=%s "$TIP")" > "$C/tree.txt"
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
        "rc=$rc; $(cat "$C/tree.txt"); $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

if step off; then
    echo "=== off $(date -Is)"
    BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
    A=$WORK/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env>: rc
        local tag=$1 bin=$2 envs=$3
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$BASE_CPU" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    for arm in "tip:" "glumt:GGML_CUDA_MMVQ_PASCAL_GLU_MT=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        rc=$(archs "$tag" "$C/bin" "$envs")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -f "$A"/*.f32
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

echo "=== done $(date -Is)"
touch "$WORK/C757_DONE"
