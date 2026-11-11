#!/usr/bin/env bash
# checks.sh: C758 (Q1335) on a build machine, CPU only: GGML_CUDA_MMVQ_PASCAL_DMT=1 at one row per lane group by
# default (fork branch pascal-p106-v4-b12246-c758, one host-code commit on BASE) built and checked without a GPU. Every
# build and test in a container without GPUs at low CPU priority (--cpu-shares=256, nice 19), one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   build   the rig tarball of BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK, whose
#           src/ and ccache/ it uses; its md5
#   sass    cuobjdump -sass of every object of the CUDA backend in the tip's build tree (WORK/src/build) and in
#           BASE_BUILD (BASE's rig build tree): per object the code with the anonymous namespaces' hashes masked, the
#           same or not; and per kernel of mmvq-pascal-dmt.cu.o (mul_mat_vec_q_pascal_dmt<type, NT, R, PF>) the same
#           way, with its registers, stack and local memory (sass/kernels.tsv)
#   pick    run/pick.cu built against BASE's and the tip's own mmvq-pascal-dmt.cu (nvcc, PTX only, nothing launched) and
#           run under each setting: what each product of 2-4 tokens launches (R, prefetch, warps, shared memory,
#           kernel). Checks that the tip's DMT=1 launches BASE's DMT=1 DMT_R=1 everywhere but at IQ4_XS, 2 tokens,
#           K = 2560, where it launches BASE's DMT=2 DMT_R=1; that every earlier setting has a spelling on the tip that
#           launches the same; and joins the kernels the tip's DMT=1 launches at Qwen3.8-Flash-Next's shapes with their
#           SASS in both builds (pick/flashnext.tsv)
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in WORK/cpu/tree
#   off     test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, the tip's compares, with
#           no switch and with GGML_CUDA_MMVQ_PASCAL_DMT=1 (a CUDA switch: the CPU results must be the same)
#   ctest   ctest in the tip's CPU build tree, without the tests that need a download, the network or torch (as C565)
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE=<base commit> BASE_BUILD=<BASE's rig build tree,
#   e.g. .../src/build> BASE_CPU=<bin/ of BASE's static native CPU build> [LOCK=<flock file>] [JOBS=8]
#   [DOCKER_ARGS=...] [CPU_CCACHE_SEED=<a ccache folder to start the CPU build's from>] [STEPS=...] bash checks.sh
#   WORK/src must be a llama.cpp clone with the fork as remote "fork" (https fetch URL), as build_llamacpp_docker.sh wants.
# Output: WORK/summary.tsv (check, verdict, detail), WORK/sass/, WORK/pick/, logs in WORK/logs; done when WORK/C758_DONE
#   exists.
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}"
JOBS=${JOBS:-8}
STEPS=${STEPS:-build sass pick cpu off ctest}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
O=CMakeFiles/ggml-cuda.dir
mkdir -p "$WORK/logs" "$WORK/sass" "$WORK/pick" "$WORK/cpu"
rm -f "$WORK/C758_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball() { ls -td "$WORK"/llama.cpp-*-"${BRANCH##*/}" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== c758 checks started $(date -Is), $BRANCH, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    locked env WORK="$WORK" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/logs/build.log" 2>&1
    rc=$?
    T=$(tarball)
    verdict build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/C758_DONE"; exit 1; }
fi

if step sass; then
    echo "=== sass $(date -Is)"
    : "${BASE_BUILD:?BASE_BUILD=<base rig build tree>}"
    BASE_BUILD=$(realpath "$BASE_BUILD")
    TIPD=$WORK/src/build/ggml/src/ggml-cuda/$O
    BASED=$BASE_BUILD/ggml/src/ggml-cuda/$O
    S=$WORK/sass
    # one container: every object's SASS, with each function name's anonymous-namespace hash masked, as an md5 per
    # object and side; then mmvq-pascal-dmt.cu.o split per function (one file per kernel, masked) on both sides
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" -v "$BASE_BUILD:$BASE_BUILD:ro" "$IMAGE" bash -c '
        mask() { sed -E "s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g"; }
        cd '"$TIPD"' && find . -name "*.cu.o" | sort > '"$S"'/objects.txt
        while read -r o; do
            t=$(nice -n 19 cuobjdump -sass "'"$TIPD"'/$o" | mask | md5sum | cut -d" " -f1)
            if [ -f "'"$BASED"'/$o" ]; then
                b=$(nice -n 19 cuobjdump -sass "'"$BASED"'/$o" | mask | md5sum | cut -d" " -f1)
            else
                b=-
            fi
            printf "%s\t%s\t%s\n" "${o#./}" "$b" "$t"
        done < '"$S"'/objects.txt > '"$S"'/objects_md5.tsv
        for side in base:'"$BASED"' tip:'"$TIPD"'; do
            d='"$S"'/dmt_${side%%:*}
            rm -rf "$d" && mkdir -p "$d"
            cuobjdump -res-usage "${side#*:}/mmvq-pascal-dmt.cu.o" > "$d/res.txt" 2>&1
            cuobjdump -sass "${side#*:}/mmvq-pascal-dmt.cu.o" | mask | awk -v d="$d" "
                /Function : / { if (f) close(f); n++; f = d \"/\" n \".sass\"; print n \"\t\" \$3 > (d \"/index.tsv\") }
                f { print > f }"
        done
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$WORK/logs/sass.log" 2>&1
    rc=$?
    awk -F'\t' 'BEGIN { OFS = "\t"; print "object", "verdict", "base md5", "tip md5" }
        { print $1, ($2 == "-" ? "new" : ($2 == $3 ? "same" : "DIFFERS")), $2, $3 }' "$S/objects_md5.tsv" > "$S/sass_objects.tsv"
    n=$(tail -n +2 "$S/sass_objects.tsv" | wc -l)
    same=$(awk -F'\t' '$2 == "same"' "$S/sass_objects.tsv" | wc -l)
    new=$(awk -F'\t' '$2 == "new" { printf "%s ", $1 }' "$S/sass_objects.tsv")
    differ=$(awk -F'\t' '$2 == "DIFFERS" { printf "%s ", $1 }' "$S/sass_objects.tsv")
    verdict sass-objects "$([ $rc = 0 ] && [ "$n" -gt 0 ] && [ -z "$differ$new" ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; $n objects: $same the same as the base, new: ${new:-none}; differ: ${differ:-none}"
    # per kernel: <type, NT, R, PF> from the mangled name (ggml_type 12 = q4_K, 23 = iq4_xs), registers from res-usage
    kernels() { # <side dir>: kernel, md5, REG, STACK, LOCAL
        local d=$1 i fn key name res
        while IFS=$'\t' read -r i fn; do
            if [[ $fn =~ dmtIL9ggml_type([0-9]+)ELi([0-9]+)ELi([0-9]+)ELb([01])E ]]; then
                key=${BASH_REMATCH[0]}
                name="mul_mat_vec_q_pascal_dmt<$(case ${BASH_REMATCH[1]} in 12) echo q4_K ;; 23) echo iq4_xs ;; *) echo "type${BASH_REMATCH[1]}" ;; esac), ${BASH_REMATCH[2]}, ${BASH_REMATCH[3]}, $([ "${BASH_REMATCH[4]}" = 1 ] && echo true || echo false)>"
            else
                key=$fn name="other: $fn"
            fi
            res=$(awk -v key="$key" 'index($0, "Function") { f = index($0, key) > 0; next } f && /REG:/ { print; exit }' "$d/res.txt")
            printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$(md5sum < "$d/$i.sass" | cut -d' ' -f1)" \
                "$(sed -nE 's/.*REG:([0-9]+).*/\1/p' <<< "$res")" "$(sed -nE 's/.*STACK:([0-9]+).*/\1/p' <<< "$res")" \
                "$(sed -nE 's/.*LOCAL:([0-9]+).*/\1/p' <<< "$res")"
        done < "$d/index.tsv" | LC_ALL=C sort
    }
    kernels "$S/dmt_base" > "$S/kernels_base.tsv"
    kernels "$S/dmt_tip" > "$S/kernels_tip.tsv"
    LC_ALL=C join -t $'\t' -a 1 -a 2 -e - -o 0,1.2,2.2,2.3,2.4,2.5 "$S/kernels_base.tsv" "$S/kernels_tip.tsv" \
        | awk -F'\t' 'BEGIN { OFS = "\t"; print "kernel", "verdict", "base md5", "tip md5", "REG", "STACK", "LOCAL" }
                      { print $1, ($2 == $3 ? "same" : "DIFFERS"), $2, $3, $4, $5, $6 }' > "$S/kernels.tsv"
    k=$(tail -n +2 "$S/kernels.tsv" | wc -l)
    kd=$(awk -F'\t' 'NR > 1 && $2 != "same" { printf "%s ", $1 }' "$S/kernels.tsv")
    verdict sass-dmt-kernels "$([ "$k" -gt 0 ] && [ -z "$kd" ] && echo SAME || echo DIFFERS)" \
        "$k functions in mmvq-pascal-dmt.cu.o, each one's SASS against the base's; differ or missing: ${kd:-none}"
fi

P=$WORK/pick
if step pick; then
    echo "=== pick $(date -Is)"
    : "${BASE:?BASE=<base commit>}" "${BASE_BUILD:?BASE_BUILD=<base rig build tree>}"
    TIP=$(git -C "$WORK/src" rev-parse "fork/$BRANCH")
    rm -rf "$P/base" "$P/tip" "$P/arms" && mkdir -p "$P/base" "$P/tip" "$P/arms"
    git -C "$WORK/src" archive "$BASE" ggml | tar -x -C "$P/base"
    git -C "$WORK/src" archive "$TIP" ggml | tar -x -C "$P/tip"
    cp "$HERE/pick.cu" "$P/pick.cu"
    # the launcher's prefetch line in each tree: pick.cu's PICK_PF repeats it (C758_BASE for the base)
    lb=$(grep -c 'const bool pf = dmt_mode() != 2;' "$P/base/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu")
    lt=$(grep -c 'const bool pf = dmt_pick_pf(type, nt, nsb, r);' "$P/tip/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu")
    # the build's own CUDA defines (no include paths), from the base's rig build tree
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$(realpath "$BASE_BUILD")/ggml/src/ggml-cuda/$O/flags.make")
    # base and tip arms: name, side, switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out)
    ARMS='base-on:base:DMT=1
base-nopf:base:DMT=2
base-r1:base:DMT=1 DMT_R=1
base-r2:base:DMT=1 DMT_R=2
base-r1nopf:base:DMT=2 DMT_R=1
base-r2nopf:base:DMT=2 DMT_R=2
base-nw2:base:DMT=1 DMT_NW=2
tip-default:tip:DMT=1
tip-r1:tip:DMT=1 DMT_R=1
tip-rule:tip:DMT=1 DMT_R=0
tip-rule-nopf:tip:DMT=2 DMT_R=0
tip-pf:tip:DMT=1 DMT_PF=1
tip-r2:tip:DMT=1 DMT_R=2
tip-nopf:tip:DMT=2
tip-r1nopf:tip:DMT=2 DMT_R=1
tip-r2nopf:tip:DMT=2 DMT_R=2
tip-rule-nw2:tip:DMT=1 DMT_R=0 DMT_NW=2'
    printf '%s\n' "$ARMS" > "$P/arms.txt"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        set -e
        for side in base tip; do
            T='"$P"'/$side
            nice -n 19 nvcc -std=c++17 -O1 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                $([ $side = base ] && echo -DC758_BASE) -I$T/ggml/include -I$T/ggml/src -I$T/ggml/src/ggml-cuda \
                -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections \
                -o $T/pick $T/../pick.cu
        done
        while IFS=: read -r name side sw; do
            envs=""
            for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
            env $envs '"$P"'/$side/pick > '"$P"'/arms/$name.tsv
        done < '"$P"'/arms.txt
        chown -R '"$(id -u):$(id -g)"' '"$P"'' > "$WORK/logs/pick.log" 2>&1
    rc=$?
    pickok=$([ $rc = 0 ] && [ "$lb" = 1 ] && [ "$lt" = 1 ] && [ -s "$P/arms/tip-default.tsv" ] && echo 1 || echo 0)
    verdict pick-build "$([ "$pickok" = 1 ] && echo OK || echo FAIL)" \
        "rc=$rc; base $BASE, tip $TIP; the launchers' prefetch lines found: base $lb, tip $lt; $(ls "$P/arms" | wc -l) arms, $(cat "$P/arms/tip-default.tsv" 2> /dev/null | tail -n +2 | wc -l) products each"
fi
if step pick && [ "${pickok:-0}" = 1 ]; then
    A=$P/arms
    # 1. the tip's default against the base's DMT=1 DMT_R=1, and at IQ4_XS, 2 tokens, K = 2560 its DMT=2 DMT_R=1
    awk -F'\t' 'BEGIN { OFS = "\t" }
        FILENAME ~ /base-r1\.tsv$/     { r1[FNR] = $0; next }
        FILENAME ~ /base-r1nopf\.tsv$/ { np[FNR] = $0; next }
        FNR == 1 { print $0, "the base arm it equals"; next }
        { want = ($1 == "iq4_xs" && $2 == 2 && $3 == 2560) ? "r1nopf" : "r1"
          ref = want == "r1" ? r1[FNR] : np[FNR]
          print $0, ($0 == ref ? "base " (want == "r1" ? "DMT=1 DMT_R=1" : "DMT=2 DMT_R=1") : "DIFFERS from base " want) }' \
        "$A/base-r1.tsv" "$A/base-r1nopf.tsv" "$A/tip-default.tsv" > "$P/default.tsv"
    nd=$(awk -F'\t' 'NR > 1' "$P/default.tsv" | wc -l)
    bad=$(awk -F'\t' 'NR > 1 && $NF ~ /DIFFERS/ { printf "%s/%s/%s ", $1, $2, $3 }' "$P/default.tsv")
    nnp=$(awk -F'\t' 'NR > 1 && $NF == "base DMT=2 DMT_R=1"' "$P/default.tsv" | wc -l)
    # where the base's r1 and the tip's default part (type/NT/K), there and nowhere else
    part=$(diff "$A/base-r1.tsv" "$A/tip-default.tsv" | awk -F'\t' '/^> / { sub(/^> /, ""); printf "%s/%s/%s ", $1, $2, $3 }')
    dr1=$(diff "$A/tip-default.tsv" "$A/tip-r1.tsv" | grep -c '^[<>]')
    verdict pick-default "$([ "$nd" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
        "tip DMT=1, $nd products: $((nd - nnp)) launch what base DMT=1 DMT_R=1 launched, $nnp what base DMT=2 DMT_R=1 launched; differ: ${bad:-none}; tip DMT=1 parts from base DMT=1 DMT_R=1 at (type/NT/K): ${part:-none}; tip DMT=1 DMT_R=1 against tip DMT=1: $dr1 lines differ"
    # 2. every earlier setting and its spelling on the tip
    printf 'base arm\tbase switches\ttip spelling\ttip switches\tproducts\tdiffering lines\n' > "$P/spellings.tsv"
    for pair in on:rule nopf:rule-nopf r1:pf r2:r2 r1nopf:nopf r1nopf:r1nopf r2nopf:r2nopf nw2:rule-nw2; do
        b=base-${pair%%:*} t=tip-${pair#*:}
        nl=$(diff "$A/$b.tsv" "$A/$t.tsv" | grep -c '^[<>]')
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${b#base-}" "$(grep "^$b:" "$P/arms.txt" | cut -d: -f3)" "${t#tip-}" \
            "$(grep "^$t:" "$P/arms.txt" | cut -d: -f3)" "$(($(wc -l < "$A/$t.tsv") - 1))" "$nl" >> "$P/spellings.tsv"
    done
    sbad=$(awk -F'\t' 'NR > 1 && $6 != 0 { printf "%s=%s(%s) ", $1, $3, $6 }' "$P/spellings.tsv")
    verdict pick-spellings "$([ -z "$sbad" ] && echo SAME || echo DIFFERS)" \
        "$(($(wc -l < "$P/spellings.tsv") - 1)) pairs, each earlier setting against its spelling on the tip; differ: ${sbad:-none}"
    # 3. the products the kernel takes: the same set in the base's arms with R = 1 or C722's rule and the tip's default
    # (R = 2 forced declines the rows where no warp of two rows a lane group fits)
    taken() { awk -F'\t' 'NR > 1 { print $1, $2, $3, ($5 ~ /declined/ ? "declined" : "taken") }' "$1"; }
    tk=$(for b in on nopf r1 r1nopf nw2; do diff <(taken "$A/base-$b.tsv") <(taken "$A/tip-default.tsv") > /dev/null || printf '%s ' "$b"; done)
    verdict pick-taken "$([ -z "$tk" ] && echo SAME || echo DIFFERS)" \
        "$(awk -F'\t' 'NR > 1 && $5 !~ /declined/' "$A/tip-default.tsv" | wc -l) products taken, $(awk -F'\t' 'NR > 1 && $5 ~ /declined/' "$A/tip-default.tsv" | wc -l) declined by the tip's default; base arms with another set: ${tk:-none}"
    # 4. Qwen3.8-Flash-Next's shapes (K = 2560 and 6144): the kernel the tip's DMT=1 launches, R79's arm that ran it,
    # and its SASS in both builds
    if [ -f "$WORK/sass/kernels.tsv" ]; then
        awk -F'\t' 'BEGIN { OFS = "\t"; print "type", "NT", "K", "kernel (tip DMT=1)", "R79 arm", "warps", "smem", "SASS", "base md5", "tip md5", "REG" }
            FILENAME ~ /kernels\.tsv$/ { v[$1] = $2; b[$1] = $3; t[$1] = $4; reg[$1] = $5; next }
            FNR > 1 && ($3 == 2560 || $3 == 6144) { arm = $6 == 1 ? "r1 (DMT=1 DMT_R=1)" : "r1nopf (DMT=2 DMT_R=1)"
                print $1, $2, $3, $11, arm, $7, $9, (v[$11] == "" ? "MISSING" : v[$11]), b[$11], t[$11], reg[$11] }' \
            "$WORK/sass/kernels.tsv" "$A/tip-default.tsv" > "$P/flashnext.tsv"
        fb=$(awk -F'\t' 'NR > 1 && $8 != "same"' "$P/flashnext.tsv" | wc -l)
        verdict pick-flashnext "$([ "$fb" = 0 ] && echo SAME || echo DIFFERS)" \
            "$(($(wc -l < "$P/flashnext.tsv") - 1)) Flash-Next products (2 types x 3 token counts x K 2560, 6144): each kernel the tip's DMT=1 launches has the base's SASS; $fb not"
    fi
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
    for arm in "tip:" "dmt:GGML_CUDA_MMVQ_PASCAL_DMT=1"; do
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
touch "$WORK/C758_DONE"
