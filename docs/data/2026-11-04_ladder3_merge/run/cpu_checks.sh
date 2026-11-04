#!/usr/bin/env bash
# C227 (Q780) on a build machine, CPU only: fork branch pascal-p106-v3-b12076-ladder3 (thirteen branches merged onto
# -ladder2) built for the rig, and every merged branch's own CPU checks re-run on the merged tree, in containers without
# a GPU (the CUDA toolkit's stub libcuda.so stands in for the driver, so the CUDA backend finds no device):
#   union     git only: every changed line of the merge against -ladder2 is a line one of the branches changed, and every
#             file only one branch touched is that branch's file byte for byte (exceptions listed)
#   tree      `git archive` of the merged commit (and of -ladder2) for the checks that read the source
#   decode    the kernels' host checks on the merged tree: C70/C93's k-column decoding (kcol_unit_check, the bench's
#             header replaced by the tree's) with its mutation check; C2's IQ2_XXS/IQ2_XS/IQ3_S decoding (iq_unit_check,
#             plus MUTATE=1); C163's k-column and multi-token grid-type decoding (iq_kcol_check, plus MUTATE=1); C6's
#             sparse attention (bench/fattn_sparse/check.sh); C135's transpose-reduce on an emulated warp
#   build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1, sm_61 + compute_61 PTX) and, from its build
#             tree, test-llama-archs, test-llama-context, test-draft-mem and test-hop-cast (a second tarball)
#   cpubuild  the merged tree without CUDA (static, native): llama-server, llama-quantize, llama-fit-params and the tests
#   sass      per-kernel SASS identity of the merged test-backend-ops against -ladder2 and every source branch's build
#             (docs/data/2026-11-04_ladder2_merge/run/sass_identity.sh)
#   identity  test-llama-archs -b CPU on every architecture, logits written by -ladder2's binary and compared by the
#             merged one: the rig builds (CUDA build, CPU backend) and the CPU builds, at the defaults and with every
#             switch the merge brings off; all must be bit for bit
#   gdn       C19/C20's check: random qwen4exp on the CPU, fused graph against unfused, n_rs_seq 0 and 3
#   draft     Q564's test-draft-mem
#   ctx       C22/C28's checks (docs/data/2026-11-04_ckpt_cards/run/checks.sh) with the rig build's binaries
#   vgraphs   C151's cpu_checks.sh steps ops sched models ctx archs spec ckpt, on the merged CPU build
#   hops      C200/C201's cpu_checks.sh steps ops models unit model identity ctx server, the merged CPU build against
#             -ladder2's (BASE_CPU)
#   bytesw    Q618/Q619's cpu_check.sh step 1 (random qwen4exp heads, every switch arm against off)
#   qsa       C6/C23's cpu_e2e.sh on the merged tree and on -ladder2's: the qwen4exp rows OK on both, same numbers
#   ckptrc    C154/C155's chain.sh step checks (context, server, mem, fit) with the rig build and the merged CPU build
#   specdump  Q688's cpu_check.sh (its own CPU build of a clone of the merged commit)
#   adraft    Q687's cpu_check.sh (its own CPU builds of the merged commit and of -ladder2)
#   taper     C218's cpu_checks.sh steps unit batchalloc models taper identity ctx server replay, the merged CPU build
#             against -ladder2's (BASE_CPU)
#   lightprof C142's test-cuda-profile-light, and its rig script's dry run
#   ccflags   C131's cmake_hook_check.sh on the merged tree: GGML_CUDA_PASCAL_FLAGS reaches exactly the Pascal kernel files
# Usage: WORK=<dir> REPO=<this repository> BASE_BUILD=<-ladder2's unpacked rig tarball, extras in bin/> \
#   BASE_CPU=<-ladder2's static native CPU build: llama-server, test-llama-archs> SASS_REFS='<base tag>=<dir> ...' \
#   [BRANCH=pascal-p106-v3-b12076-ladder3] [BASE=a330672c1] [BRANCHES='<name>=<tip> ...'] [CCACHE_CPU_SEED='<dirs>'] \
#   [JOBS=6] [THREADS=8] [LOCK=<file>] [DOCKER_ARGS=...] [STEPS='union tree ...'] bash cpu_checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src needs the fork as remote
#             "fork" with an https fetch URL, and the branches' tips (BRANCHES) fetched
#   SASS_REFS as sass_identity.sh takes them: the base first, then <tag>[:<cut-from tag>]=<build dir> per source build
#   CCACHE_CPU_SEED  ccache folders of earlier CPU builds of the source branches, copied into the specdump and adraft
#             work folders before their builds (optional)
#   LOCK      a file to flock around each build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra-<sha>.tar.gz, $WORK/cpu/bin/,
#   $WORK/checks/<check>/, $WORK/b/<branch>/ (the branches' own scripts' work folders), $WORK/logs/, $WORK/summary.tsv
#   (check, verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-ladder3}
BASE=${BASE:-a330672c1}
BRANCHES=${BRANCHES:-kcol=af46311af iqkcol=0fd6ce5c1 iq2xs=6c88b6991 qsa=bdb902d95 fap=ff8e1ec24 bytesw=fd549b754 ccflags=9e1a8492e vgraphs=a476ab83f specdump=dafb09a4a adraft=b52064398 ckptrc=3e49981fb hops=5b69b0574 lightprof=973bcf09d taper=7b1632aee}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
STEPS=${STEPS:-union tree decode build extra cpubuild sass identity gdn draft ctx vgraphs hops bytesw qsa ckptrc specdump adraft taper lightprof ccflags}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
PYIMAGE=python:3.12-slim-bookworm
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
DATA=$REPO/docs/data
S=$WORK/src
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks" "$WORK/cpu" "$WORK/b"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
ok_if() { if [ "$1" = 0 ]; then echo OK; else echo FAIL; fi; }
# a container without a GPU whose CUDA backend loads against the stub driver
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
L3=$(git -C "$S" rev-parse "fork/$BRANCH")
L3S=$(git -C "$S" rev-parse --short=9 "$L3")
echo "=== chain started $(date -Is), $BRANCH at $L3S, steps: $STEPS"
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
# every switch the merged branches bring whose default is not -ladder2's behaviour, at -ladder2's behaviour (the kernel
# switches act only on compute 6.1 and the profiler only when GGML_CUDA_PROFILE is set, so they do not appear here)
OFF_ENV="LLAMA_GRAPH_PER_SIZE=0 LLAMA_GRAPH_META_CALLOC=0 LLAMA_FIT_DRAFT_CTX=0 LLAMA_FIT_NEXTN_SKIP=0"

if step union; then
    echo "=== union $(date -Is)"
    U=$WORK/checks/union
    rm -rf "$U" && mkdir -p "$U/diff"
    lines() { git -C "$S" diff -U0 "$1" "$2" -- "$3" | grep -a "^$4" | grep -av "^$4$4$4 " | LC_ALL=C sort -u; }
    printf 'file\tbranches\tadded_extra\tadded_missing\tremoved_extra\tremoved_missing\tsingle_branch_identical\n' > "$U/union.tsv"
    nbad=0
    for f in $(git -C "$S" diff --name-only "$BASE" "$L3"); do
        bs=""
        : > "$U/u_add" && : > "$U/u_rem"
        for bt in $BRANCHES; do
            b=${bt%%=*} t=${bt#*=}
            mb=$(git -C "$S" merge-base "$BASE" "$t")
            git -C "$S" diff --quiet "$mb" "$t" -- "$f" && continue
            bs="$bs $b"
            lines "$mb" "$t" "$f" + >> "$U/u_add"
            lines "$mb" "$t" "$f" - >> "$U/u_rem"
        done
        LC_ALL=C sort -u -o "$U/u_add" "$U/u_add"; LC_ALL=C sort -u -o "$U/u_rem" "$U/u_rem"
        lines "$BASE" "$L3" "$f" + > "$U/m_add"; lines "$BASE" "$L3" "$f" - > "$U/m_rem"
        ae=$(LC_ALL=C comm -23 "$U/m_add" "$U/u_add" | wc -l); am=$(LC_ALL=C comm -13 "$U/m_add" "$U/u_add" | wc -l)
        re=$(LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem" | wc -l); rm_=$(LC_ALL=C comm -13 "$U/m_rem" "$U/u_rem" | wc -l)
        si=""
        set -- $bs
        if [ $# = 1 ]; then
            t=$(for bt in $BRANCHES; do [ "${bt%%=*}" = "$1" ] && echo "${bt#*=}"; done)
            [ "$(git -C "$S" rev-parse "$L3:$f" 2>/dev/null)" = "$(git -C "$S" rev-parse "$t:$f" 2>/dev/null)" ] && si=yes || si=no
        fi
        if [ "$ae$am$re$rm_" != 0000 ]; then
            nbad=$((nbad + 1))
            { echo "== $f (branches:$bs)"; echo "-- added in the merge, by no branch:"; LC_ALL=C comm -23 "$U/m_add" "$U/u_add"
              echo "-- added by a branch, not in the merge:"; LC_ALL=C comm -13 "$U/m_add" "$U/u_add"
              echo "-- removed in the merge, by no branch:"; LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem"
              echo "-- removed by a branch, not in the merge:"; LC_ALL=C comm -13 "$U/m_rem" "$U/u_rem"; } > "$U/diff/$(echo "$f" | tr / _).txt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "${bs# }" "$ae" "$am" "$re" "$rm_" "$si" >> "$U/union.tsv"
    done
    rm -f "$U"/u_add "$U"/u_rem "$U"/m_add "$U"/m_rem
    nf=$(($(wc -l < "$U/union.tsv") - 1))
    nsi=$(awk -F'\t' '$7 == "yes"' "$U/union.tsv" | wc -l); nsn=$(awk -F'\t' '$7 == "no"' "$U/union.tsv" | wc -l)
    verdict union "$([ $nbad = 0 ] && echo OK || echo REVIEW)" \
        "$nf files changed; $nbad with lines outside the union (checks/union/diff/: $(ls "$U/diff" | tr '\n' ' ')); single-branch files identical to the branch's: $nsi, not: $nsn ($(awk -F'\t' '$7 == "no" { printf "%s ", $1 }' "$U/union.tsv"))"
fi

if step tree; then
    echo "=== tree $(date -Is)"
    for x in "tree:$L3" "tree-base:$BASE"; do
        d=$WORK/${x%%:*}
        rm -rf "$d" && mkdir -p "$d" && git -C "$S" archive "${x#*:}" | tar -x -C "$d"
    done
    verdict tree OK "tree = $L3S, tree-base = $BASE"
fi

if step decode; then
    echo "=== decode $(date -Is)"
    K=$WORK/checks/decode
    rm -rf "$K" && mkdir -p "$K"
    # C70 / C93: the bench's copy of the k-column header (and the grid decoder it includes) replaced by the merged tree's;
    # the diff against the published copy is what the merge changed in them
    R=$K/kcolrepo/bench/mmvq_pascal
    mkdir -p "$R" && cp -r "$REPO/bench/mmvq_pascal/src" "$REPO/bench/mmvq_pascal/verify" "$R/"
    { head -n 3 "$REPO/bench/mmvq_pascal/src/kcol_mm.cuh"; cat "$WORK/tree/ggml/src/ggml-cuda/mmvq-pascal-kcol-impl.cuh"; } > "$R/src/kcol_mm.cuh"
    cp "$WORK/tree/ggml/src/ggml-cuda/mmvq-pascal-iq.cuh" "$R/src/mmvq-pascal-iq.cuh"
    { diff "$REPO/bench/mmvq_pascal/src/kcol_mm.cuh" "$R/src/kcol_mm.cuh"; diff "$REPO/bench/mmvq_pascal/src/mmvq-pascal-iq.cuh" "$R/src/mmvq-pascal-iq.cuh"; } > "$K/header.diff"
    DOCKER_ARGS="$D" locked nice -n 19 bash "$R/verify/kcol_unit_check.sh" 40 > "$K/kcol.txt" 2>&1
    rc1=$?
    REPO=$K/kcolrepo WORK=$K/kcolmut DOCKER_ARGS="$D" locked nice -n 19 bash "$DATA/2026-11-04_kcol_coverage/run/mutation_check.sh" > "$K/kcol_mut.txt" 2>&1
    rc2=$?
    # the mutation check passes when exactly the three planted types fail (its own exit status does not say)
    mf=$(awk '/ cases +worst err\/tol / && $NF == "FAIL" { print $1 }' "$K/kcol_mut.txt" | LC_ALL=C sort | tr '\n' ' ')
    mp=$(awk '/ cases +worst err\/tol / && $NF == "PASS"' "$K/kcol_mut.txt" | wc -l)
    verdict decode-kcol "$([ $rc1 = 0 ] && grep -q 'every type matches' "$K/kcol.txt" && [ $rc2 = 0 ] && [ "$mf" = "iq4_nl q4_0 q5_0 " ] && echo OK || echo FAIL)" \
        "header diff vs the published copy: $(grep -c '^[<>]' "$K/header.diff") lines ($(grep '^[<>]' "$K/header.diff" | tr -s ' ' | tr '\n' '|' | cut -c 1-160)); unit rc $rc1: $(grep -c ' PASS$' "$K/kcol.txt") types PASS, $(tail -n 1 "$K/kcol.txt"); mutation rc $rc2: fail ${mf:-none}, $mp pass"
    # C2: IQ2_XXS, IQ2_XS, IQ3_S (one token); C163: the grid types and IQ1_M in the k-column and multi-token kernels
    for c in iq_unit_check:200 iq_kcol_check:100; do
        n=${c%%:*}
        JOBS=$JOBS DOCKER_ARGS="$D" locked nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/$n.sh" "$WORK/tree" "${c#*:}" > "$K/$n.txt" 2>&1
        rc1=$?
        MUTATE=1 JOBS=$JOBS DOCKER_ARGS="$D" locked nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/$n.sh" "$WORK/tree" 20 > "$K/${n}_mut.txt" 2>&1
        rc2=$?
        note=""
        if [ "$n" = iq_unit_check ] && [ $rc2 = 3 ]; then
            # C163 gave mmvq-pascal-iq.cuh a second copy of IQ3_S's index line (decode32), so MUTATE=1's sed edits three
            # lines where the script counts two and stops; a copy that accepts every site of the two edits runs the
            # same mutations (on the one-token functions the check exercises, and on decode32)
            R=$K/iqrepo/bench/mmvq_pascal
            mkdir -p "$R" && cp -r "$REPO/bench/mmvq_pascal/verify" "$R/" && mkdir -p "$R/build"
            perl -pi -e 's/\| grep -c "\^>"\)" = 2 \]/| grep -c "^>")" -ge 2 ]/' "$R/verify/iq_unit_check.sh"
            grep -q -- '-ge 2 ]' "$R/verify/iq_unit_check.sh" && \
            MUTATE=1 JOBS=$JOBS DOCKER_ARGS="$D" GGML_LIB=$REPO/bench/mmvq_pascal/build/iq_ggml/ggml/src/libggml-base.a \
                locked nice -n 19 bash "$R/verify/iq_unit_check.sh" "$WORK/tree" 20 > "$K/${n}_mut_sites.txt" 2>&1
            rc2=$?
            note="; as published: the mutations did not apply (3 sites); every site mutated:"
        fi
        verdict "decode-${n%_check}" "$([ $rc1 = 0 ] && [ $rc2 = 0 ] && echo OK || echo FAIL)" \
            "rc $rc1: $(tail -n 3 "$K/$n.txt" | tr '\n' ' ' | cut -c 1-200)$note MUTATE=1 rc $rc2: $(tail -n 2 "$K/${n}_mut${note:+_sites}.txt" | tr '\n' ' ' | cut -c 1-160)"
    done
    # C6: the sparse decode attention's host model, faults and mutations
    SRC=$WORK/tree JOBS=$JOBS DOCKER_ARGS="$D" locked nice -n 19 bash "$REPO/bench/fattn_sparse/check.sh" > "$K/fasp.txt" 2>&1
    rc=$?
    cp "$REPO/bench/fattn_sparse/build/check.txt" "$K/fasp_check.txt" 2>/dev/null
    verdict decode-fasp "$(ok_if $rc)" "rc $rc; $(tail -n 4 "$K/fasp.txt" | tr '\n' ' ' | cut -c 1-240)"
    # C135: the transpose-reduce (both forms) on an emulated warp
    mkdir -p "$K/reduce" && cp "$DATA/2026-11-04_fap_spills/run/reduce_check.cpp" "$K/reduce/"
    locked docker run --rm $D -v "$K/reduce:/r" -w /r "$IMAGE" bash -c 'g++ -O2 -std=c++17 -o reduce_check reduce_check.cpp && ./reduce_check > reduce_check.txt 2>&1; rc=$?; chown -R '"$(id -u):$(id -g)"' /r; exit $rc'
    rc=$?
    verdict decode-fap-reduce "$(ok_if $rc)" "rc $rc; $(tail -n 2 "$K/reduce/reduce_check.txt" | tr '\n' ' ')"
fi

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
}
extra() {
    # the test binaries the rig build leaves out, from the same build tree
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs test-llama-context test-draft-mem test-hop-cast \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        rm -rf /work/extra && mkdir -p /work/extra
        cp build/bin/test-llama-archs build/bin/test-llama-context build/bin/test-draft-mem build/bin/test-hop-cast /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
cpubuild() {
    # the same checkout as the rig build, built outside the tree without CUDA (the flags of C151's and C200's CPU builds)
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        git -C /work/src log -1 --format="cpu build of %H (%ci)"
        T="llama-server llama-quantize llama-fit-params test-backend-ops test-opt test-alloc test-llama-archs test-llama-context
           test-draft-mem test-hop-cast test-bytes-switches test-spec-dump test-spec-adapt test-cuda-profile-light
           test-ubatch-taper test-batch-alloc"
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target $T > /work/logs/cpubuild.log 2>&1 \
            || { tail -40 /work/logs/cpubuild.log; exit 1; }
        rm -rf /work/cpu/bin && mkdir -p /work/cpu/bin
        for t in $T; do cp /work/cpu/build/bin/$t /work/cpu/bin/; done
        git -C /work/src rev-parse HEAD > /work/cpu/bin/COMMIT
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build extra cpubuild
export WORK BRANCH JOBS IMAGE D REPO

if step build; then
    echo "=== build $(date -Is)"
    locked bash -c build
    rc=$?
    echo "=== build ended $(date -Is) rc=$rc"
    if [ $rc != 0 ]; then verdict build FAIL "see logs/build.log"; date -Is > "$WORK/DONE"; exit 1; fi
    T=$(tarball_tree)
    (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
    verdict build OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"
fi
T=$(tarball_tree)

if step extra; then
    echo "=== extra $(date -Is)"
    locked bash -c extra > "$WORK/logs/extra.console" 2>&1
    rc=$?
    if [ $rc != 0 ]; then verdict extra FAIL "rc $rc; see logs/extra.log"; else
        (cd "$WORK" && tar czf "extra-$L3S.tar.gz" extra && md5sum "extra-$L3S.tar.gz" && stat -c '%s bytes' "extra-$L3S.tar.gz") | tee -a "$WORK/md5.txt"
        cp "$WORK"/extra/test-* "$T/bin/"
        verdict extra OK "extra-$L3S.tar.gz: $(ls "$WORK/extra" | tr '\n' ' ')"
    fi
fi

if step cpubuild; then
    echo "=== cpubuild $(date -Is)"
    locked bash -c cpubuild > "$WORK/logs/cpubuild.console" 2>&1
    rc=$?
    verdict cpubuild "$(ok_if $rc)" "rc $rc; $(grep -m 1 "cpu build of" "$WORK/logs/cpubuild.console")"
    [ $rc = 0 ] || { date -Is > "$WORK/DONE"; exit 1; }
fi

if step sass; then
    echo "=== sass $(date -Is)"
    # shellcheck disable=SC2086
    WORK=$WORK/checks DOCKER_ARGS="${DOCKER_ARGS:-}" nice -n 19 bash "$DATA/2026-11-04_ladder2_merge/run/sass_identity.sh" "$T" ${SASS_REFS:?} \
        > "$WORK/logs/sass.log" 2>&1
    rc=$?
    verdict sass "$(grep -q '^verdict: every kernel' "$WORK/checks/sass/summary.txt" 2>/dev/null && echo OK || echo REVIEW)" \
        "rc $rc; $(grep -E '^verdict|carried|merged build' "$WORK/checks/sass/summary.txt" 2>/dev/null | tr '\n' ' ')"
fi

if step identity; then
    echo "=== identity $(date -Is)"
    K=$WORK/checks/identity
    rm -rf "$K" && mkdir -p "$K"
    # <name> <base bin dir> <merged bin dir> <env for the merged runs, ;-separated arms name:K=V,...>
    ident() {
        local n=$1 bb=$2 mb=$3 arms=$4
        mkdir -p "$K/$n"
        docker run --rm $D -v "$bb:/xb:ro" -v "$mb:/x:ro" -v "$K/$n:/k" -e ARMS="$arms" "$IMAGE" bash -c "$STUB"'
            run() { # <name> <bin dir> <env...>
                local a=$1 b=$2; shift 2
                env "$@" LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $b/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > /k/$a.log
                echo "$a rc=${PIPESTATUS[0]}"
            }
            run base /xb
            ls /k/logits-*.f32 | wc -l > /k/n_base.txt
            IFS=";" read -ra A <<< "$ARMS"
            for x in "${A[@]}"; do run "${x%%:*}" /x $(echo "${x#*:}" | tr , " "); done
            rm -f /k/*.f32
            chown -R '"$(id -u):$(id -g)"' /k' > "$K/$n/rc.txt" 2>&1
        grep 'logits against' "$K/$n/base.log" > "$K/$n/base_self.txt"
        printf 'run\tarch\tnmse\tmax_abs_diff\tdiffer\tof\tbase_self\n' > "$K/$n/identity.tsv"
        for a in base $(echo "$arms" | tr ';' '\n' | cut -d: -f1); do
            grep 'logits against' "$K/$n/$a.log" | awk -v r="$a" 'NR == FNR { s[$0] = 1; next }
                { if (match($0, /^[a-z0-9_.-]+: logits against [^:]*: nmse [^,]*, max abs diff [^,]*, [0-9]+ of [0-9]+ differ/)) {
                      split($0, f, /: logits against [^:]*: nmse |, max abs diff |, | of | differ/)
                      printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\n", r, f[1], f[2], f[3], f[4], f[5], ($0 in s) } }' \
                "$K/$n/base_self.txt" - >> "$K/$n/identity.tsv"
        done
        local nb ns ok=1 d=""
        nb=$(cat "$K/$n/n_base.txt" 2>/dev/null || echo 0); ns=$(wc -l < "$K/$n/base_self.txt")
        for a in $(echo "$arms" | tr ';' '\n' | cut -d: -f1); do
            c=$(awk -F'\t' -v r="$a" '$1 == r && !$7' "$K/$n/identity.tsv" | wc -l)
            s=$(awk -F'\t' -v r="$a" '$1 == r && $7' "$K/$n/identity.tsv" | wc -l)
            x=$(awk -F'\t' -v r="$a" '$1 == r && !$7 && $5 > 0 { printf "%s ", $2 }' "$K/$n/identity.tsv")
            [ "$c" = "$nb" ] && [ "$c" -gt 0 ] && [ "$s" = "$ns" ] && [ -z "$x" ] || ok=0
            d="$d$a: $c of $nb archs compared, differ: ${x:-none}, $(grep -m 1 '^Summary' "$K/$n/$a.log"), FAIL rows $(grep -c '^|.*FAIL' "$K/$n/$a.log"); "
        done
        verdict "identity-$n" "$([ $ok = 1 ] && echo OK || echo FAIL)" "base wrote $nb archs ($(grep -m 1 '^Summary' "$K/$n/base.log")); $d$(tr '\n' ' ' < "$K/$n/rc.txt")"
    }
    ident rig "$BASE_BUILD/bin" "$T/bin" "default:;off:${OFF_ENV// /,}"
    ident cpu "$BASE_CPU" "$WORK/cpu/bin" "default:;off:${OFF_ENV// /,}"
fi

if step gdn; then
    echo "=== gdn $(date -Is)"
    K=$WORK/checks/gdn
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $D -v "$T/bin:/x:ro" -v "$K:/k" "$IMAGE" bash -c "$STUB"'
        for rs in 0 3; do
            for fuse in 0 1; do
                echo "--- n_rs_seq=$rs LLAMA_GDN_FUSE=$fuse"
                LLAMA_ARCHS_N_RS_SEQ=$rs LLAMA_GDN_FUSE=$fuse LLAMA_ARCHS_LOGITS=/k/rs$rs- \
                    nice -n 19 /x/test-llama-archs -a "^qwen4exp$" -b CPU -s 1 > /k/archs-rs$rs-fuse$fuse.log 2>&1
                echo "rc=$?"
                grep -E "logits against|one batch against|wrote|FAIL|OK" /k/archs-rs$rs-fuse$fuse.log | sed "s/\x1b\[[0-9;]*m//g"
            done
        done
        rm -f /k/*.f32
        chown -R '"$(id -u):$(id -g)"' /k' > "$K/console.txt" 2>&1
    nrc=$(grep -c '^rc=0' "$K/console.txt"); nfail=$(grep -c 'FAIL' "$K/console.txt")
    verdict gdn "$([ "$nrc" = 4 ] && [ "$nfail" = 0 ] && echo OK || echo FAIL)" \
        "$nrc/4 runs rc 0, $nfail FAIL lines; $(grep -E 'logits against' "$K/console.txt" | tr -s ' ' | tr '\n' ';')"
fi

if step draft; then
    echo "=== draft $(date -Is)"
    mkdir -p "$WORK/checks/draft"
    docker run --rm $D -v "$T/bin:/x:ro" "$IMAGE" bash -c "$STUB"' && nice -n 19 /x/test-draft-mem' > "$WORK/checks/draft/test-draft-mem.log" 2>&1
    rc=$?
    verdict draft "$(ok_if $rc)" "rc $rc; $(tail -n 2 "$WORK/checks/draft/test-draft-mem.log" | tr '\n' ' ')"
fi

if step ctx; then
    echo "=== ctx $(date -Is)"
    K=$WORK/checks/ctx
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $D -v "$T/bin:/x:ro" -v "$DATA/2026-11-04_ckpt_cards/run:/ckrun:ro" -v "$K:/k" -e THREADS="$THREADS" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        '"$STUB"'
        cd /k && BIN=/x RUN=/ckrun OUT=/k THREADS=$THREADS nice -n 19 bash /ckrun/checks.sh
        rc=$?
        chown -R '"$(id -u):$(id -g)"' /k
        exit $rc' > "$K/checks.log" 2>&1
    rc=$?
    st=$(awk -F'\t' 'NR > 1 && $2 ~ /^store/ { print $1 ":" $2 "=" $3 }' "$K/context.tsv" 2>/dev/null | tr '\n' ' ')
    tally=$(awk -F'\t' 'NR > 1 { n[$3]++ } END { for (v in n) printf "%s %d; ", v, n[v] }' "$K/context.tsv" 2>/dev/null)
    verdict ctx "$(ok_if $rc)" "rc $rc; cases: $tally store cases: $st; server: $(head -n 1 "$K/server/verdict.txt" 2>/dev/null)"
fi

# a branch script's work folder with the merged CPU build (and -ladder2's) as its own; its verdicts join ours
bwork() { # <name>
    local w=$WORK/b/$1
    rm -rf "$w" && mkdir -p "$w/cpu" "$w/cpu-base"
    cp -al "$WORK/cpu/bin" "$w/cpu/bin"
    cp -a "$BASE_CPU" "$w/cpu-base/bin"
    echo "$w"
}
fold() { # <name> <summary.tsv>
    [ -f "$2" ] || { verdict "$1" FAIL "no $2"; return; }
    awk -F'\t' -v p="$1" 'NR > 1 { printf "%s:%s\t%s\t%s\n", p, $1, $2, $3 }' "$2" | tee -a "$WORK/summary.tsv"
}

if step vgraphs; then
    echo "=== vgraphs $(date -Is)"
    w=$(bwork vgraphs)
    WORK=$w REPO=$REPO THREADS=$THREADS DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS="ops sched models ctx archs spec ckpt" \
        bash "$DATA/2026-11-04_vgraphs/run/cpu_checks.sh" > "$WORK/logs/vgraphs.log" 2>&1
    fold vgraphs "$w/summary.tsv"
fi

if step hops; then
    echo "=== hops $(date -Is)"
    w=$(bwork hops)
    WORK=$w REPO=$REPO THREADS=$THREADS DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS="ops models unit model identity ctx server" \
        bash "$DATA/2026-11-04_card_hops/run/cpu_checks.sh" > "$WORK/logs/hops.log" 2>&1
    fold hops "$w/summary.tsv"
fi

if step bytesw; then
    echo "=== bytesw $(date -Is)"
    K=$WORK/checks/bytesw
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $D -v "$WORK/cpu/bin:/x:ro" -v "$DATA/2026-11-04_bytes_switches/run:/r:ro" -v "$K:/o" -e THREADS="$THREADS" "$IMAGE" \
        bash -c 'BIN=/x OUT=/o THREADS=$THREADS nice -n 19 bash /r/cpu_check.sh; rc=$?; rm -f /o/random/*.gguf; chown -R '"$(id -u):$(id -g)"' /o; exit $rc' \
        > "$K/console.txt" 2>&1
    rc=$?
    n0=$(grep -c ': exit 0;' "$K/console.txt")
    verdict bytesw "$([ $rc = 0 ] && [ "$n0" = 4 ] && echo OK || echo FAIL)" "rc $rc, $n0/4 arms exit 0; $(grep ': exit' "$K/console.txt" | cut -c 1-110 | tr '\n' '|')"
fi

if step qsa; then
    echo "=== qsa $(date -Is)"
    K=$WORK/checks/qsa
    rm -rf "$K" && mkdir -p "$K"
    for x in merged:tree base:tree-base; do
        SRC=$WORK/${x#*:} OUT=$K/${x%%:*} JOBS=$JOBS DOCKER_ARGS="$D" locked nice -n 19 bash "$DATA/2026-11-04_qsa_sparse_attn/run/cpu_e2e.sh" \
            > "$K/${x%%:*}.console" 2>&1
        echo "rc=$?" >> "$K/${x%%:*}.console"
        rm -rf "$K/${x%%:*}/build" "$K/${x%%:*}/models"
    done
    # the qwen4exp rows of test-llama-archs (logits, roundtrip, chunked against one batch) and test-llama-context's totals
    for a in merged base; do
        sed 's/\x1b\[[0-9;]*m//g' "$K/$a/archs.txt" 2>/dev/null | grep -iE 'qwen4exp|against' | sed 's/[0-9.]*ms//g' > "$K/$a.rows"
    done
    same=$(cmp -s "$K/merged.rows" "$K/base.rows" && echo "same rows as -ladder2" || echo "rows DIFFER from -ladder2")
    nf=$(cat "$K"/merged/archs.txt "$K"/merged/context.txt 2>/dev/null | grep -c 'FAIL')
    verdict qsa "$([ "$nf" = 0 ] && grep -q 'rc=0' "$K/merged.console" && [ -s "$K/merged.rows" ] && cmp -s "$K/merged.rows" "$K/base.rows" && echo OK || echo REVIEW)" \
        "$same; FAIL lines $nf; context: $(grep -E 'passed|failed' "$K/merged/context.txt" 2>/dev/null | tr -s ' ' | tr '\n' ' ' | cut -c 1-160); stats lines: $(grep -c 'QSA k-pool inputs' "$K/merged/archs_stats.txt" 2>/dev/null)"
fi

if step ckptrc; then
    echo "=== ckptrc $(date -Is)"
    w=$WORK/b/ckptrc
    rm -rf "$w" && mkdir -p "$w/extra/$BRANCH" "$w/old"
    cp -al "$T" "$w/" && ln "$T.tar.gz" "$w/"
    for t in llama-server llama-fit-params test-llama-context test-llama-archs; do cp -l "$WORK/cpu/bin/$t" "$w/extra/$BRANCH/"; done
    cp -a "$BASE_BUILD/bin/." "$w/old/"
    WORK=$w REPO=$REPO BRANCH=$BRANCH THREADS=$THREADS STEPS=checks PARTS="context server mem fit" OLD_BIN=$w/old MOUNTS=$w/old \
        DOCKER_ARGS="${DOCKER_ARGS:-}" bash "$DATA/2026-11-04_ckpt_refcount/run/chain.sh" > "$WORK/logs/ckptrc.log" 2>&1
    f=$(grep -o 'checks ended .* fail=[0-9]*' "$w/logs/checks.log" | tail -n 1)
    verdict ckptrc "$([[ "$f" == *fail=0 ]] && echo OK || echo FAIL)" \
        "$f; $(grep -E ': exit|PASS|FAIL|verdict' "$w/logs/checks.log" | grep -v '^+' | tr -s ' ' | tr '\n' '|' | cut -c 1-300)"
fi

# specdump and adraft build from a clone of the merged commit in their own work folder, as they did on their branches
bclone() { # <name>
    local w=$WORK/b/$1
    rm -rf "$w" && mkdir -p "$w/ccache-cpu"
    git clone -q --local --no-checkout "$S" "$w/src" && git -C "$w/src" checkout -q "$L3"
    for c in ${CCACHE_CPU_SEED:-}; do [ -d "$c" ] && cp -a "$c/." "$w/ccache-cpu/"; done
    echo "$w"
}

if step specdump; then
    echo "=== specdump $(date -Is)"
    w=$(bclone specdump)
    WORK=$w REPO=$REPO JOBS=$JOBS THREADS=$THREADS LOCK=${LOCK:-} DOCKER_ARGS="${DOCKER_ARGS:-}" \
        bash "$DATA/2026-11-04_tree_replay/run/cpu_check.sh" > "$WORK/logs/specdump.log" 2>&1
    fold specdump "$w/cpu/summary.tsv"
fi

if step adraft; then
    echo "=== adraft $(date -Is)"
    w=$(bclone adraft)
    [ -d "$WORK/b/specdump/ccache-cpu" ] && cp -a "$WORK/b/specdump/ccache-cpu/." "$w/ccache-cpu/"
    WORK=$w REPO=$REPO BASE_REF=$BASE JOBS=$JOBS LOCK=${LOCK:-} DOCKER_ARGS="${DOCKER_ARGS:-}" \
        bash "$DATA/2026-11-04_adaptive_draft/run/cpu_check.sh" > "$WORK/logs/adraft.log" 2>&1
    fold adraft "$w/cpu/summary.tsv"
fi

if step taper; then
    echo "=== taper $(date -Is)"
    w=$(bwork taper)
    WORK=$w REPO=$REPO THREADS=$THREADS DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS="unit batchalloc models taper identity ctx server replay" \
        bash "$DATA/2026-11-04_prompt_taper/run/cpu_checks.sh" > "$WORK/logs/taper.log" 2>&1
    fold taper "$w/summary.tsv"
fi

if step lightprof; then
    echo "=== lightprof $(date -Is)"
    K=$WORK/checks/lightprof
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $D -v "$WORK/cpu/bin:/x:ro" "$IMAGE" /x/test-cuda-profile-light > "$K/test-cuda-profile-light.log" 2>&1
    rc1=$?
    docker run --rm $D -v "$REPO:/r:ro" "$PYIMAGE" bash /r/docs/data/2026-11-04_light_profiler/run/dryrun.sh > "$K/dryrun.txt" 2>&1
    rc2=$?
    verdict lightprof "$([ $rc1 = 0 ] && grep -q 'dry run: pass' "$K/dryrun.txt" && echo OK || echo FAIL)" \
        "test-cuda-profile-light rc $rc1: $(tail -n 1 "$K/test-cuda-profile-light.log"); dry run rc $rc2: $(grep 'dry run:' "$K/dryrun.txt")"
fi

if step ccflags; then
    echo "=== ccflags $(date -Is)"
    K=$WORK/checks/ccflags
    rm -rf "$K" && mkdir -p "$K"
    SRC=$WORK/tree LOCK=${LOCK:-} DOCKER_ARGS="${DOCKER_ARGS:-}" bash "$DATA/2026-11-04_ccflags/run/cmake_hook_check.sh" > "$K/cmake_hook_check.txt" 2>&1
    rc=$?
    n=$(cd "$WORK/tree/ggml/src/ggml-cuda" && ls mmvq-pascal*.cu mmvf-pascal*.cu fattn-pascal.cu fattn-tile-pascal.cu moe-persistent.cu gdn-fused.cu 2>/dev/null | wc -l)
    got=$(grep -m 1 'CUDA files carry the options' "$K/cmake_hook_check.txt" | awk '{ print $1 }')
    got2=$(grep 'CUDA files carry the options' "$K/cmake_hook_check.txt" | sed -n 2p | awk '{ print $1 }')
    verdict ccflags "$([ "$got" = "$n" ] && [ "$got2" = 0 ] && grep -q 'compile exit 0' "$K/cmake_hook_check.txt" && echo OK || echo FAIL)" \
        "rc $rc; with the option $got CUDA files carry it (the tree has $n Pascal kernel files), without it $got2; $(grep 'compile exit' "$K/cmake_hook_check.txt")"
fi

cat "$WORK/md5.txt" 2>/dev/null
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
