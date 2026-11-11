#!/usr/bin/env bash
# checks.sh: C728 (Q1287) on a build machine, CPU only: the K tiles of the read-once IQ4_XS and Q4_K kernel (fork branch
# pascal-p106-v4-b12246-c728, GGML_CUDA_MMVQ_PASCAL_DMT_KT) built and checked without a GPU. Every build and test in a
# container without GPUs at low CPU priority (--cpu-shares=256, nice 19), one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   build    the rig tarball of BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK, whose
#            src/ and ccache/ it uses; its md5
#   sass     cuobjdump -sass and -ptx of the llama-server of the tarball and of BASE_TARBALL (the base commit's rig
#            tarball), the per-file hashes of anonymous namespaces masked, cut into functions (sass_attrib.py dump, from
#            docs/data/2026-11-11_v4_day_merge/run) and compared function by function: every function the base's code,
#            the new mul_mat_vec_q_pascal_dmt_kt<type, NT, M, PF> only in the tip; then the new kernels' register
#            use and instruction counts beside C722's kernels (run/sass_check.sh on mmvq-pascal-dmt.cu.o)
#   record   the replay set dmtkt (scripts/record_checks.sh SUITE=dmtkt) with the tarball's test-backend-ops; the cases a
#            set in LINK_FROM holds are hardlinked in
#   bitid    bench/mmvq_pascal/verify/multitoken_bitid linked against the tip's build tree (mode dmtkt)
#   cpubase  a static native CPU build of BASE_COMMIT (GGML_CUDA=OFF, every target) in WORK/cpubase/tree
#   cpu      the same of the tip in WORK/cpu/tree (the same ccache)
#   off      test-llama-archs -b CPU -s 1: the base's binary writes every architecture's logits, the tip's compares, with
#            no switch and with GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 (CUDA switches: the CPU results
#            must be the same)
#   ctest    ctest in the tip's CPU build tree, without the tests that need a download, the network or torch (as C565)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE_COMMIT=<base commit> BASE_TARBALL=<the base
#   commit's rig tarball> [LINK_FROM=<complete replay sets>] [LOCK=<flock file>] [JOBS=8] [DOCKER_ARGS=...]
#   [CPU_CCACHE_SEED=<a ccache folder to start the CPU builds' from>] [STEPS=...] bash checks.sh
#   WORK/src must be a llama.cpp clone with the fork as remote "fork" (https fetch URL), as build_llamacpp_docker.sh wants.
# Output: WORK/summary.tsv (check, verdict, detail), WORK/sass/ (sass_functions.tsv, sass_counts.tsv), WORK/sets/dmtkt.tar,
#   logs in WORK/logs; done when WORK/C728_DONE exists.
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}"
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass record bitid cpubase cpu off ctest}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
ATTRIB=$REPO/docs/data/2026-11-11_v4_day_merge/run/sass_attrib.py
mkdir -p "$WORK/logs" "$WORK/sass"
rm -f "$WORK/C728_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball() { ls -td "$WORK"/llama.cpp-*-"${BRANCH##*/}" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== c728 checks started $(date -Is), $BRANCH, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    locked env WORK="$WORK" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/logs/build.log" 2>&1
    rc=$?
    T=$(tarball)
    verdict build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/C728_DONE"; exit 1; }
fi

if step sass; then
    echo "=== sass $(date -Is)"
    BASE_TARBALL=$(realpath "${BASE_TARBALL:?BASE_TARBALL=<the base commit rig tarball>}")
    T=$(tarball)
    S=$WORK/sass
    B=$WORK/base
    mkdir -p "$B"
    [ -x "$B/llama-server" ] || tar -xzf "$BASE_TARBALL" -C "$B" --wildcards '*/bin/llama-server' --strip-components 2
    for side in base tip; do
        if [ $side = base ]; then bin=$B/llama-server; else bin=$T/bin/llama-server; fi
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$WORK:$WORK:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$bin" \
                | sed -E 's/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g' \
                | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$side.$w.tsv"
        done
    done
    for w in sass ptx; do
        # name<TAB>md5s per side: the same, new (tip only), gone (base only) or DIFFERS
        awk -F'\t' -v w=$w 'BEGIN { OFS = "\t" }
            NR == FNR { b[$1] = $2; next }
            { t[$1] = $2 }
            END {
                for (f in t) print w, f, (f in b ? (b[f] == t[f] ? "same" : "DIFFERS") : "new")
                for (f in b) if (!(f in t)) print w, f, "gone"
            }' "$S/base.$w.tsv" "$S/tip.$w.tsv" | sort -k1,1 -k3,3 -k2,2 > "$S/functions.$w.tsv"
        n=$(wc -l < "$S/functions.$w.tsv")
        same=$(awk -F'\t' '$3 == "same"' "$S/functions.$w.tsv" | wc -l)
        new=$(awk -F'\t' '$3 == "new"' "$S/functions.$w.tsv" | wc -l)
        newkt=$(awk -F'\t' '$3 == "new" && $2 ~ /mul_mat_vec_q_pascal_dmt_kt/' "$S/functions.$w.tsv" | wc -l)
        other=$(awk -F'\t' '$3 != "same" && !($3 == "new" && $2 ~ /mul_mat_vec_q_pascal_dmt_kt/) { printf "%s %s; ", $3, $2 }' "$S/functions.$w.tsv")
        verdict "functions-$w" "$([ "$n" -gt 0 ] && [ "$newkt" = 24 ] && [ -z "$other" ] && echo SAME || echo CHECK)" \
            "$n functions: $same the base's code, $new new ($newkt mul_mat_vec_q_pascal_dmt_kt); else: ${other:-none}"
    done
    { printf 'side\tfunction\tverdict\n'; cat "$S/functions.sass.tsv" "$S/functions.ptx.tsv"; } > "$S/sass_functions.tsv"
    O=$WORK/src/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        cuobjdump -res-usage '"$O"'/mmvq-pascal-dmt.cu.o > '"$S"'/dmt_res.txt 2>&1
        cuobjdump -sass '"$O"'/mmvq-pascal-dmt.cu.o > '"$S"'/dmt.sass 2>&1
        cuobjdump -sass '"$O"'/mmvq-pascal.cu.o | gzip > '"$S"'/mmvqp.sass.gz
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$WORK/logs/sass.log" 2>&1
    rc=$?
    SASS=$S bash "$HERE/sass_check.sh" > "$S/sass_counts.tsv" 2> "$S/sass_check.err"
    k=$(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt"' "$S/sass_counts.tsv" | wc -l)
    spills=$(awk -F'\t' 'NR > 1 && $1 ~ /^mul_mat_vec_q_pascal_dmt/ && ($7 != 0 || $8 != 0)' "$S/sass_counts.tsv" | wc -l)
    # I2F per kernel, those of an integer division (I2F.RP) left out, against C722's kernel of the same type and NT (the
    # staging's (float) sum for Q4_K; the step's conversions are exact without I2F)
    i2f=$(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt" { n[$2 FS $3] = $16 - $17 }
                      NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" && $16 - $17 != n[$2 FS $3]' "$S/sass_counts.tsv" | wc -l)
    verdict sass-kt "$([ $rc = 0 ] && [ "$k" = 24 ] && [ "$spills" = 0 ] && [ "$i2f" = 0 ] && echo OK || echo CHECK)" \
        "rc=$rc; $k instantiations, $spills kernels of the file with stack or local memory; registers $(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" { print $6 }' "$S/sass_counts.tsv" | sort -n | sed -n '1p;$p' | tr '\n' '-' | sed 's/-$//'); $i2f with other I2F counts than C722's kernel of their type and tokens, those of a division left out"
fi

if step record; then
    echo "=== record $(date -Is)"
    T=$(tarball)
    mkdir -p "$WORK/sets"
    locked env OUT="$WORK/sets" SUITE=dmtkt BIN="$T/bin" JOBS="$JOBS" LINK_FROM="${LINK_FROM:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$WORK/logs/record.log" 2>&1
    rc=$?
    verdict record "$([ $rc = 0 ] && [ -f "$WORK/sets/dmtkt.tar" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -a 'linked .* in all' "$WORK/sets/dmtkt.log" | tail -n 1 | cut -d' ' -f2-); $(tail -n 1 "$WORK/sets/dmtkt.log" 2> /dev/null | cut -d' ' -f2-); manifest lines $(wc -l < "$WORK/sets/dmtkt/manifest.tsv" 2> /dev/null), runs $(wc -l < "$WORK/sets/dmtkt/runs.tsv" 2> /dev/null)"
fi

if step bitid; then
    echo "=== bitid $(date -Is)"
    locked env DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$WORK/src" "$WORK/src" \
        > "$WORK/logs/bitid_build.log" 2>&1
    rc=$?
    B=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    verdict bitid-build "$([ $rc = 0 ] && [ -x "$B" ] && echo OK || echo FAIL)" \
        "rc=$rc; md5 $(md5sum < "$B" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$B" 2> /dev/null) B; modes: $(grep -ac 'dmtkt' "$B") dmtkt strings"
fi

cpu_build() { # <tag> <commit>: a static native CPU build of the commit in WORK/<tag>/tree, its test-llama-archs in bin/
    local tag=$1 commit=$2 C=$WORK/$1 T
    T=$C/tree
    rm -rf "$T" "$C/bin" && mkdir -p "$T/src" "$C/bin"
    [ -d "$WORK/cpu-ccache" ] || { if [ -n "${CPU_CCACHE_SEED:-}" ]; then cp -a "$CPU_CCACHE_SEED" "$WORK/cpu-ccache"; else mkdir -p "$WORK/cpu-ccache"; fi; }
    git -C "$WORK/src" archive "$commit" | tar -x -C "$T/src"
    echo "commit $(git -C "$WORK/src" rev-parse "$commit") $(git -C "$WORK/src" log -1 --format=%s "$commit")" > "$C/tree.txt"
    if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
        printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
            | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$WORK/logs/image.log" 2>&1
    fi
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" -e CCACHE_DIR="$WORK/cpu-ccache" -e CCACHE_BASEDIR="$T" "$CPU_IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$C/build.log"' 2>&1'
    local rc=$?
    cp "$T/build/bin/test-llama-archs" "$C/bin/" 2> /dev/null
    verdict "$tag" "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$C/tree.txt"); $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
}

if step cpubase; then
    echo "=== cpubase $(date -Is)"
    cpu_build cpubase "${BASE_COMMIT:?BASE_COMMIT=<base commit>}"
fi

if step cpu; then
    echo "=== cpu $(date -Is)"
    cpu_build cpu "fork/$BRANCH"
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$WORK/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env>: rc
        local tag=$1 bin=$2 envs=$3
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$WORK:$WORK" "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$WORK/cpubase/bin" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    for arm in "tip:" "kt:GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        rc=$(archs "$tag" "$WORK/cpu/bin" "$envs")
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
    B=$WORK/cpu/tree/build
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
touch "$WORK/C728_DONE"
