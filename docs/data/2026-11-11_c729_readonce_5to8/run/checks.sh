#!/usr/bin/env bash
# C729 (Q1288): the read-once kernels at 5 to 8 tokens (GGML_CUDA_MMVQ_PASCAL_RO8), checked on a build machine without a
# GPU: every build and test in a container (the CPU ones without network) at low CPU priority, one at a time under LOCK.
# From C767's and C758's checks.sh (2026-11-11_c767_q5k_one_row/run/, 2026-11-11_c758_dmt_default/run/). Steps (STEPS, in
# this order by default):
#   base    the base's rig build (scripts/build_llamacpp_docker.sh, TESTS=1, BASE_BRANCH) in $WORK/rig; its CUDA objects
#           move to $WORK/base/obj and its bin/llama-server to $WORK/base/bin, the rest of its tarball is deleted
#   build   the branch's rig tarball the same way in $WORK/rig (the base's ccache serves the unchanged files)
#   sass    CUDA 12.9's cuobjdump (-sass, -res-usage, -ptx) of every CUDA object of both builds, and -sass of both
#           bin/llama-server; run/compare_sass.py: every function the base's code but the 24 new instantiations, and per
#           instantiation of both read-once kernels its registers, stack, local memory, spills and instruction counts
#   pick    run/pick.cu built against the base's and the tip's own mmvq-pascal-dmt.cu and mmvq-pascal-q5k.cu (nvcc, PTX only,
#           nothing launched) and run under each setting (ARMS below): what each product of 1-8 tokens launches. The
#           base's and the tip's lines at 1-4 tokens must be the same under every setting, the tip's under RO8 too; at 5-8
#           tokens the tip launches only with RO8, at R = 1
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid (modes dmt8 and q5k8 among them) linked against the branch's build
#           tree (verify/build_multitoken_bitid.sh), into $REPO/bench/mmvq_pascal/build
#   cpu     static native CPU builds (GGML_CUDA=OFF, every target) of the base and the branch in $WORK/cpu/{base,tip}
#   off     test-llama-archs -b CPU -s 1: the base's binary writes every architecture's logits, the branch's compares them
#           with no switch and with GGML_CUDA_MMVQ_PASCAL_RO8=1 (a CUDA switch: the CPU results must be the same)
#   ctest   ctest in the branch's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   record  the replay set ro8 (scripts/record_checks.sh SUITE=ro8) with the tarball's test-backend-ops on the CPU, into
#           $WORK/sets
#   clean   the build trees, the CPU trees and the ccaches (the tarball, its tree, the set and the tables stay)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> [BASE_BRANCH=pascal-p106-v4-b12246-day] [REMOTE=fork]
#   [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04]
#   [CPU_IMAGE=pq-c565-cpu:24.04] [STEPS=...] bash checks.sh
#   $WORK/rig is scripts/build_llamacpp_docker.sh's work folder: its llama.cpp clone in $WORK/rig/src has the fork as
#   remote REMOTE with an https fetch URL, its ccache in $WORK/rig/ccache
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/sass/*.tsv, $WORK/pick/*.tsv, $WORK/sets/ro8.tar, logs in
#   $WORK/logs and $WORK/checks; done when $WORK/C729_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}"
BASE_BRANCH=${BASE_BRANCH:-pascal-p106-v4-b12246-day}
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
STEPS=${STEPS:-base build sass pick bitid cpu off ctest record}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
O=build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
SRC=$WORK/rig/src
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$WORK/base" "$WORK/cpu"
rm -f "$WORK/C729_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tree()    { ls -td "$WORK"/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# shellcheck disable=SC2086
box()     { locked docker run --rm $DA --network none -v "$WORK:$WORK" -v "$REPO:$REPO:ro" "$CPU_IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH" "$BASE_BRANCH:refs/remotes/$REMOTE/$BASE_BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
BASE=$(git -C "$SRC" rev-parse "$REMOTE/$BASE_BRANCH")
echo "=== c729 checks started $(date -Is), $BRANCH at ${TIP:0:9}, base $BASE_BRANCH at ${BASE:0:9}, steps: $STEPS"

if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi

rig_build() { # <branch> <log>: the rig tarball of a fork branch in $WORK/rig
    locked env WORK="$WORK/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "$REMOTE/$1" > "$2" 2>&1
}

if step base; then
    echo "=== base $(date -Is)"
    rig_build "$BASE_BRANCH" "$WORK/logs/base.log"
    rc=$?
    T=$(tree "$BASE_BRANCH")
    rm -rf "$WORK/base/obj" "$WORK/base/bin" && mkdir -p "$WORK/base/bin"
    if [ $rc = 0 ] && [ -n "$T" ] && [ -d "$SRC/$O" ]; then
        mv "$SRC/$O" "$WORK/base/obj"
        cp "$T/bin/llama-server" "$T/BUILDINFO.txt" "$WORK/base/bin/"
        md5=$(md5sum < "$T.tar.gz" | cut -d' ' -f1)
        rm -rf "$T" "$T.tar.gz"
        verdict base OK "$(basename "$T"): $(grep -E '^commit' "$WORK/base/bin/BUILDINFO.txt"); its tarball md5 $md5 (deleted); $(find "$WORK/base/obj" -name '*.cu.o' | wc -l) CUDA objects kept"
    else
        verdict base FAIL "rc=$rc; logs/base.log"
    fi
fi

if step build; then
    echo "=== build $(date -Is)"
    rig_build "$BRANCH" "$WORK/logs/build.log"
    rc=$?
    T=$(tree "$BRANCH")
    verdict build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/C729_DONE"; exit 1; }
fi

if step sass; then
    echo "=== sass $(date -Is)"
    T=$(tree "$BRANCH")
    S=$WORK/sass
    rm -rf "$S" && mkdir -p "$S"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        set -u
        dump() { # <out> <objects dir> <bin dir>
            local out=$1 obj=$2 bin=$3 o n
            mkdir -p "$out/obj"
            (cd "$obj" && find . -name "*.cu.o" | sort) > "$out/objects.txt"
            while read -r o; do
                n=$(basename "$o" .o)
                nice -n 19 cuobjdump -sass "$obj/$o" > "$out/obj/$n.sass" 2>> "$out/errors.txt"
                nice -n 19 cuobjdump -res-usage "$obj/$o" > "$out/obj/$n.res" 2>> "$out/errors.txt"
                nice -n 19 cuobjdump -ptx "$obj/$o" > "$out/obj/$n.ptx" 2>> "$out/errors.txt"
            done < "$out/objects.txt"
            nice -n 19 cuobjdump -sass "$bin/llama-server" > "$out/llama-server.sass" 2>> "$out/errors.txt"
        }
        cuobjdump --version | tail -n 1 > '"$S"'/cuobjdump.txt
        dump '"$S/base"' '"$WORK/base/obj"' '"$WORK/base/bin"'
        dump '"$S/tip"' '"$SRC/$O"' '"$T/bin"'
        ' > "$WORK/logs/sass.log" 2>&1
    rc=$?
    box bash -c "nice -n 19 python3 -I $HERE/compare_sass.py $S/base $S/tip $S" > "$K/compare_sass.txt" 2>&1
    rc2=$?
    # the new instantiations: no stack, no local memory, no spill instruction
    spill=$(awk -F'\t' 'NR > 1 && $8 == "new" && ($11 != 0 || $12 != 0 || $13 != 0 || $14 != 0) { printf "%s<%s,%s,%s,%s> ", $1, $2, $3, $4, $5 }' "$S/sass_kernels.tsv" 2> /dev/null)
    regs=$(awk -F'\t' 'NR > 1 && $8 == "new" { if ($10 > m) m = $10 } END { print m + 0 }' "$S/sass_kernels.tsv" 2> /dev/null)
    verdict sass "$([ $rc = 0 ] && [ $rc2 = 0 ] && echo SAME || echo DIFFERS)" \
        "rc=$rc/$rc2; $(tail -n 1 "$K/compare_sass.txt"); $(cat "$S/cuobjdump.txt" 2> /dev/null); $(cat "$S"/*/errors.txt 2> /dev/null | wc -l) error lines"
    verdict sass-new "$([ -s "$S/sass_kernels.tsv" ] && [ -z "$spill" ] && echo CLEAN || echo SPILLS)" \
        "the 24 new instantiations: at most $regs registers; with stack, local memory or spills: ${spill:-none}"
fi

P=$WORK/pick
if step pick; then
    echo "=== pick $(date -Is)"
    rm -rf "$P" && mkdir -p "$P/base" "$P/tip" "$P/arms"
    git -C "$SRC" archive "$BASE" ggml | tar -x -C "$P/base"
    git -C "$SRC" archive "$TIP" ggml | tar -x -C "$P/tip"
    cp "$HERE/pick.cu" "$P/pick.cu"
    # the expressions pick.cu repeats, in each tree
    B=$P/base/ggml/src/ggml-cuda T2=$P/tip/ggml/src/ggml-cuda
    lb=$(( $(grep -c 'if (dmt_mode() == 0 || ncols < 2 || ncols > DMT_MAX_COLS ||' "$B/mmvq-pascal-dmt.cu") + \
           $(grep -c '(ncols == 1 ? !q5kmt_one_row(nrows) : q5kmt_mode() == 0)' "$B/mmvq-pascal-q5k.cu") + \
           $(grep -c 'const int r   = q5kmt_r();' "$B/mmvq-pascal-q5k.cu") + $(grep -c 'const bool pf = q5kmt_pf();' "$B/mmvq-pascal-q5k.cu") ))
    lt=$(( $(grep -c '(ncols <= DMT_MAX_COLS ? dmt_mode() : dmt_ro8()) == 0' "$T2/mmvq-pascal-dmt.cu") + \
           $(grep -c '(ncols <= Q5KMT_MAX_COLS ? q5kmt_mode() : q5kmt_ro8()) == 0' "$T2/mmvq-pascal-q5k.cu") + \
           $(grep -c 'const int r   = q5kmt_r(nt);' "$T2/mmvq-pascal-q5k.cu") + $(grep -c 'const bool pf = q5kmt_pf(nt);' "$T2/mmvq-pascal-q5k.cu") ))
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$WORK/base/obj/flags.make")
    # name:switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out); every arm runs on both trees
    ARMS='none:
dmt:DMT=1 Q5K_MT=1
dmt-nopf:DMT=2 Q5K_MT=2
dmt-r2:DMT=1 DMT_R=2 Q5K_MT=1 Q5K_MT_R=2
dmt-pf:DMT=1 DMT_PF=1 Q5K_MT=1 Q5K_MT_PF=1
dmt-nw2:DMT=1 DMT_NW=2 Q5K_MT=1 Q5K_MT_NW=2
q5k1r:Q5K_1R=1
ro8:RO8=1
ro8-nopf:RO8=2
ro8-pf:RO8=1 Q5K_MT_PF=1 DMT_PF=1
ro8-r2:RO8=1 DMT_R=2 Q5K_MT_R=2
ro8-r0:RO8=1 DMT_R=0
ro8-nw2:RO8=1 DMT_NW=2 Q5K_MT_NW=2
ro8-dmt-nopf:RO8=1 DMT=2 Q5K_MT=2
ro8-all:RO8=1 DMT=1 Q5K_MT=1 Q5K_1R=1
ro8-all-nopf:RO8=2 DMT=2 Q5K_MT=2 Q5K_1R=1'
    printf '%s\n' "$ARMS" > "$P/arms.txt"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        set -e
        for side in base tip; do
            T='"$P"'/$side
            for k in dmt q5k; do
                nice -n 19 nvcc -std=c++17 -O1 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                    $([ $side = base ] && echo -DC729_BASE) $([ $k = q5k ] && echo -DPICK_Q5K) \
                    -I$T/ggml/include -I$T/ggml/src -I$T/ggml/src/ggml-cuda \
                    -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections \
                    -o $T/pick-$k $T/../pick.cu
            done
        done
        while IFS=: read -r name sw; do
            envs=""
            for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
            for side in base tip; do
                { env $envs '"$P"'/$side/pick-dmt; env $envs '"$P"'/$side/pick-q5k | tail -n +2; } > '"$P"'/arms/$name-$side.tsv
            done
        done < '"$P"'/arms.txt' > "$WORK/logs/pick.log" 2>&1
    rc=$?
    pickok=$([ $rc = 0 ] && [ "$lb" = 4 ] && [ "$lt" = 4 ] && [ -s "$P/arms/ro8-tip.tsv" ] && echo 1 || echo 0)
    verdict pick-build "$([ "$pickok" = 1 ] && echo OK || echo FAIL)" \
        "rc=$rc; base ${BASE:0:9}, tip ${TIP:0:9}; the expressions pick.cu repeats found: base $lb of 4, tip $lt of 4; $(ls "$P/arms" | wc -l) arm files, $(tail -n +2 "$P/arms/ro8-tip.tsv" 2> /dev/null | wc -l) products each"
fi
if step pick && [ "${pickok:-0}" = 1 ]; then
    A=$P/arms
    # 1. 1-4 tokens: the tip's lines are the base's under every arm (RO8 changes nothing there); 5-8 tokens: the base
    # launches nothing, the tip only under RO8
    bad14= bad58= n14=0 n58=0
    while IFS=: read -r name sw; do
        d=$(diff <(awk -F'\t' 'NR > 1 && $2 <= 4' "$A/$name-base.tsv") <(awk -F'\t' 'NR > 1 && $2 <= 4' "$A/$name-tip.tsv") | grep -c '^[<>]')
        [ "$d" = 0 ] || bad14="$bad14$name($d) "
        n14=$((n14 + $(awk -F'\t' 'NR > 1 && $2 <= 4' "$A/$name-tip.tsv" | wc -l)))
        b58=$(awk -F'\t' 'NR > 1 && $2 >= 5 && $5 != "off"' "$A/$name-base.tsv" | wc -l)
        t58=$(awk -F'\t' 'NR > 1 && $2 >= 5 && $5 != "off"' "$A/$name-tip.tsv" | wc -l)
        r58=$(awk -F'\t' 'NR > 1 && $2 >= 5 && $5 ~ /^[0-9]+$/ && $5 != 1' "$A/$name-tip.tsv" | wc -l)
        case " $sw " in *" RO8="*) want=on ;; *) want=off ;; esac
        if [ "$b58" != 0 ] || [ "$r58" != 0 ] || { [ $want = on ] && [ "$t58" = 0 ]; } || { [ $want = off ] && [ "$t58" != 0 ]; }; then
            bad58="$bad58$name(base $b58, tip $t58, R != 1 $r58) "
        fi
        n58=$((n58 + t58))
    done < "$P/arms.txt"
    verdict pick-1to4 "$([ -z "$bad14" ] && echo SAME || echo DIFFERS)" \
        "$(wc -l < "$P/arms.txt") arms x 1-4 tokens ($n14 product lines in all): the tip's launch the base's; differ: ${bad14:-none}"
    verdict pick-5to8 "$([ -z "$bad58" ] && echo OK || echo FAIL)" \
        "5-8 tokens: the base none in any arm; the tip only in the arms with RO8, every one at R = 1 ($n58 product lines taken in all); wrong: ${bad58:-none}"
    # 2. the tip's choices at 5-8 tokens under RO8=1 and =2, with the 4-token row of the same K for comparison
    awk -F'\t' 'BEGIN { OFS = "\t" } NR == 1 || $2 >= 4' "$A/ro8-tip.tsv" > "$P/ro8_launch.tsv"
    awk -F'\t' 'BEGIN { OFS = "\t" } NR == 1 || $2 >= 5' "$A/ro8-nopf-tip.tsv" > "$P/ro8_nopf_launch.tsv"
    # 3. Qwen3.8-Flash-Next's shapes (K = 2560 and 6144) at 4-8 tokens
    awk -F'\t' 'BEGIN { OFS = "\t" } NR == 1 || (($3 == 2560 || $3 == 6144) && $2 >= 4)' "$A/ro8-all-tip.tsv" > "$P/flashnext.tsv"
fi

if step bitid; then
    echo "=== bitid $(date -Is)"
    rm -f "$REPO/bench/mmvq_pascal/build/multitoken_bitid"
    locked env DOCKER_ARGS="$DA --network none" \
        nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$SRC" "$SRC" > "$WORK/logs/bitid.log" 2>&1
    rc=$?
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    # the weak references of dmt8 and q5k8 must resolve to the build's own functions (defined, T)
    nt=$(box bash -c "nm -C $X | grep -cE ' T ggml_cuda_mmvq_pascal_(dmt|q5k_mt)_takes\('" 2> /dev/null)
    verdict bitid-build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$nt" = 2 ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1); the two takes functions defined in it: $nt of 2; linked against $(git -C "$SRC" rev-parse --short=9 HEAD)"
fi

if step cpu; then
    echo "=== cpu $(date -Is)"
    for side in base tip; do
        R=$WORK/cpu/$side
        T=$R/tree
        rev=$([ $side = base ] && echo "$BASE" || echo "$TIP")
        rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin" "$WORK/cpu/ccache"
        git -C "$SRC" archive "$rev" | tar -x -C "$T/src"
        echo "commit $rev $(git -C "$SRC" log -1 --format=%s "$rev")" > "$R/tree.txt"
        # one path for both trees, so that the ccache serves the second
        box bash -c '
            mkdir -p /c && ln -s '"$T"' /c/tree
            export CCACHE_DIR='"$WORK/cpu/ccache"'
            cmake -S /c/tree/src -B /c/tree/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build /c/tree/build -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
        rc=$?
        cp "$T/build/bin/test-llama-archs" "$R/bin/" 2> /dev/null
        verdict "cpu-$side" "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(cat "$R/tree.txt"); all targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
    done
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$K/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env>: rc
        box bash -c 'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$3"' nice -n 19 '"$2"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$1.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$WORK/cpu/base/bin" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; the base's test-llama-archs: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    for arm in "tip:" "tip-ro8:GGML_CUDA_MMVQ_PASCAL_RO8=1"; do
        tag=${arm%%:*}
        rc=$(archs "$tag" "$WORK/cpu/tip/bin" "${arm#*:}")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${arm#*:}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -f "$A"/*.f32
fi

if step ctest; then
    echo "=== ctest $(date -Is)"
    box bash -c "mkdir -p /c && ln -s $WORK/cpu/tip/tree /c/tree && cd /c/tree/build && nice -n 19 ctest -j $JOBS --timeout 1800 --output-on-failure -E '$CTEST_EXCLUDE' --output-junit $K/ctest.xml" \
        > "$K/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$K/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$K/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}; not run: $CTEST_EXCLUDE"
fi

if step record; then
    echo "=== record $(date -Is)"
    T=$(tree "$BRANCH")
    rm -rf "$WORK/sets/ro8" "$WORK/sets/ro8.tar"
    locked env OUT="$WORK/sets" SUITE=ro8 BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="--network none $DA" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$WORK/logs/record.log" 2>&1
    rc=$?
    M=$WORK/sets/ro8/manifest.tsv
    verdict record "$([ $rc = 0 ] && [ -s "$M" ] && [ -s "$WORK/sets/ro8.tar.md5" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(wc -l < "$M" 2> /dev/null) cases, $(wc -l < "$WORK/sets/ro8/runs.tsv" 2> /dev/null) runs; ro8.tar $(stat -c %s "$WORK/sets/ro8.tar" 2> /dev/null) B, md5 $(cat "$WORK/sets/ro8.tar.md5" 2> /dev/null)"
fi

if step clean; then
    echo "=== clean $(date -Is)"
    rm -rf "$SRC/build" "$WORK/rig/ccache" "$WORK/cpu/base/tree" "$WORK/cpu/tip/tree" "$WORK/cpu/ccache" "$WORK/base/obj" \
        "$P/base" "$P/tip" "$WORK/sass/base/obj" "$WORK/sass/tip/obj" "$WORK/sass/base/llama-server.sass" \
        "$WORK/sass/tip/llama-server.sass" "$WORK/sets/ro8"
    verdict clean OK "build trees, CPU trees, ccaches, object dumps and the set's unpacked cases removed; $(du -sh "$WORK" | cut -f1) left"
fi

echo "=== done $(date -Is)"
touch "$WORK/C729_DONE"
