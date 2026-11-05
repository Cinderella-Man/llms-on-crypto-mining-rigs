#!/usr/bin/env bash
# The -ladder4 merge on a build machine, CPU only: fork branch pascal-p106-v3-b12076-ladder4 (-kshape with every
# off-by-default feature branch of the fork merged in: BRANCHES; -adraft already in -kshape) built for the rig, and the
# merged branches' own CPU checks re-run on the merged tree, every binary in a container of the build image:
#   union     git only: every line the merge changes against -kshape is a line one of the merged branches changed, or a
#             line of the merge's own commits (MERGE_OWN: the tiered layer's graph-level cut, route once's histogram
#             before the tail's cut, and the -rpc merge with GGML_OP_COUNT 105)
#   build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, no AVX, with
#             test-backend-ops) and, from its build tree, the extra tests (EXTRA) as a second tarball
#   cpubuild  the merged tree without CUDA (static, native): CPU_TARGETS into $WORK/cpu/bin
#   identity  test-llama-archs -b CPU on every architecture, the logits written by the base and compared by the merge:
#             the rig builds (BASE_BUILD, the -kshape tarball, against this one; the CUDA backend finds no device) and
#             the CPU builds (BASE_CPU, a CPU build whose CPU code is -kshape's, against $WORK/cpu/bin); nothing the
#             merge brings is on by default, so every architecture must match bit for bit
#   moeskip   -moeskip's run/cpu_checks.sh, steps switch and archs, with the merged CPU build: the tail op against the
#             graph-level ops it replaces, bit for bit
#   coldtier  -coldtier's run/cpu_checks.sh, steps random and moeids, with the merged CPU build: the tiered random models
#             against the untiered ones, byte for byte; every MUL_MAT_ID's ids distinct per token
#   combo     the two together, on coldtier's sharp random model (32 experts, 8 used) and its tiered copy (hot 24), with
#             LLAMA_MOE_TAIL=$CUT: untiered, the op equals the graph-level ops; tiered, the op as asked equals the
#             graph-level ops (a tiered layer takes them: its tier lookup reads a row per id), equals the untiered
#             model's, and differs from no cut; test-moe-ids on the tiered model with the cut; at -ub 128, 4 and 1
#   adraft    test-spec-adapt (Q687's unit test) from the merged CPU build
#   units     the other merged branches' unit tests from the merged CPU build: test-spec-block (-blockver),
#             test-spec-dump (-specdump2), test-draft-mem (-draft1card), test-quantize-fns (-n2p's four types),
#             test-hop-cast --unit-only (-hopcast4), and test-route-once on coldtier's sharp random model (-routeonce;
#             run the coldtier step first)
# Usage: WORK=<dir> REPO=<this repository> BASE_BUILD=<-kshape's unpacked rig tarball, extras in bin/> \
#   BASE_CPU=<bin/ of a static native CPU build with -kshape's CPU code> TEXT=<a text file for llama-perplexity> \
#   [BRANCH=pascal-p106-v3-b12076-ladder4] [REMOTE=fork] [BASE=d27845b67] [BRANCHES='<name>=<tip> ...'] \
#   [MERGE_OWN='<commit> ...'] [CUT=p=0.8,renorm=1] [JOBS=8] [THREADS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] \
#   [CPUS=<cpuset>] [MOUNTS='<dirs>'] [STEPS='union build ...'] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL, and the branches' tips fetched
#   MOUNTS    more folders the containers mount at their own paths (TEXT's, BASE_*'s if outside WORK)
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz}, $WORK/extra-<sha>.tar.gz, $WORK/md5.txt, $WORK/cpu/bin/,
#   $WORK/checks/<step>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-ladder4}
REMOTE=${REMOTE:-fork}
BASE=${BASE:-d27845b67}
BRANCHES=${BRANCHES:-moeskip=f3072cbe8 coldtier=f5e600bd2 blockver=b8f42a3ff specdump2=3e22fd7c1 spancount=f6c157897 \
routeonce=b3825c215 rpc=ba068ae6e replay=a89d6b281 mailbox=5aebbddd9 hopcast4=3dbf1105a draft1card=648c720c9 \
n2p=1a0832375 hybrid=8c25952e4}
MERGE_OWN=${MERGE_OWN:-6e57e208b 6fd71e172 6a9021389}
CUT=${CUT:-p=0.8,renorm=1}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-union build cpubuild identity moeskip coldtier combo adraft units}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
EXTRA=${EXTRA:-test-llama-archs test-llama-context test-draft-mem test-hop-cast test-bytes-switches test-moe-ids test-spec-adapt \
test-hop-mailbox test-route-once}
CPU_TARGETS=${CPU_TARGETS:-llama-server llama-perplexity llama-quantize test-backend-ops test-llama-archs test-bytes-switches \
test-moe-ids test-spec-adapt test-spec-block test-spec-dump test-draft-mem test-quantize-fns test-hop-cast test-route-once}
D="--cpu-shares=256 ${CPUS:+--cpuset-cpus $CPUS} ${DOCKER_ARGS:-}"
DATA=$REPO/docs/data
S=$WORK/src
mkdir -p "$WORK/logs" "$WORK/checks" "$WORK/cpu"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
mounts="-v $WORK:$WORK"
for d in ${BASE_BUILD:+"$BASE_BUILD"} ${BASE_CPU:+"$BASE_CPU"} ${TEXT:+"$(dirname "$TEXT")"} ${MOUNTS:-}; do
    mounts="$mounts -v $d:$d"
done
# shellcheck disable=SC2086
box() { docker run --rm $D $mounts "$@"; }
# a container without a GPU whose CUDA backend loads against the stub driver (the rig build's binaries)
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
L4=$(git -C "$S" rev-parse "$REMOTE/$BRANCH")
echo "=== checks started $(date -Is), $BRANCH at ${L4:0:9}, steps: $STEPS"

if step union; then
    echo "=== union $(date -Is)"
    U=$WORK/checks/union
    rm -rf "$U" && mkdir -p "$U/diff"
    lines() { git -C "$S" diff -U0 "$1" "$2" -- "$3" | grep -a "^$4" | grep -av "^$4$4$4 " | LC_ALL=C sort -u; }
    printf 'file\tbranches\tadded_extra\tadded_missing\tremoved_extra\tremoved_missing\n' > "$U/union.tsv"
    nbad=0
    for f in $(git -C "$S" diff --name-only "$BASE" "$L4"); do
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
        for c in $MERGE_OWN; do
            git -C "$S" diff --quiet "$c^" "$c" -- "$f" && continue
            bs="$bs own:${c:0:9}"
            lines "$c^" "$c" "$f" + >> "$U/u_add"
            lines "$c^" "$c" "$f" - >> "$U/u_rem"
        done
        LC_ALL=C sort -u -o "$U/u_add" "$U/u_add"; LC_ALL=C sort -u -o "$U/u_rem" "$U/u_rem"
        lines "$BASE" "$L4" "$f" + > "$U/m_add"; lines "$BASE" "$L4" "$f" - > "$U/m_rem"
        ae=$(LC_ALL=C comm -23 "$U/m_add" "$U/u_add" | wc -l); am=$(LC_ALL=C comm -13 "$U/m_add" "$U/u_add" | wc -l)
        re=$(LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem" | wc -l); rm_=$(LC_ALL=C comm -13 "$U/m_rem" "$U/u_rem" | wc -l)
        # a line the merge's own commit replaced is "added by a branch, not in the merge": expected, listed for review
        if [ "$ae$re" != 00 ] || { [ "$am$rm_" != 00 ] && [[ "$bs" != *own:* ]]; }; then
            nbad=$((nbad + 1))
            { echo "== $f (branches:$bs)"; echo "-- added in the merge, by no branch:"; LC_ALL=C comm -23 "$U/m_add" "$U/u_add"
              echo "-- added by a branch, not in the merge:"; LC_ALL=C comm -13 "$U/m_add" "$U/u_add"
              echo "-- removed in the merge, by no branch:"; LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem"
              echo "-- removed by a branch, not in the merge:"; LC_ALL=C comm -13 "$U/m_rem" "$U/u_rem"; } > "$U/diff/$(echo "$f" | tr / _).txt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "${bs# }" "$ae" "$am" "$re" "$rm_" >> "$U/union.tsv"
    done
    rm -f "$U"/u_add "$U"/u_rem "$U"/m_add "$U"/m_rem
    nf=$(($(wc -l < "$U/union.tsv") - 1))
    verdict union "$([ $nbad = 0 ] && echo OK || echo REVIEW)" \
        "$nf files changed against ${BASE:0:9}; $nbad with lines outside the union$([ $nbad = 0 ] || echo " (checks/union/diff/: $(ls "$U/diff" | tr '\n' ' '))")"
fi

if step build; then
    echo "=== build $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
        "$REMOTE/$BRANCH" > "$WORK/logs/rig.log" 2>&1
    rc=$?
    T=$(tarball_tree)
    if [ $rc = 0 ] && [ -n "$T" ]; then
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        verdict build-rig OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/rig.log") error lines in logs/rig.log"
    fi
    # the extra tests from the same build tree (the CUDA build of the merged commit)
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e EXTRA="$EXTRA" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target $EXTRA > /work/logs/extra-build.log 2>&1 \
            || { tail -40 /work/logs/extra-build.log; exit 1; }
        rm -rf /work/extra && mkdir -p /work/extra
        for t in $EXTRA; do cp build/bin/$t /work/extra/; done
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs' > "$WORK/logs/extra.log" 2>&1
    rc=$?
    if [ $rc = 0 ]; then
        sha=$(git -C "$S" rev-parse --short=9 HEAD)
        tar czf "$WORK/extra-$sha.tar.gz" -C "$WORK" extra
        [ -n "$T" ] && cp "$WORK"/extra/* "$T/bin/"
        (cd "$WORK" && md5sum "extra-$sha.tar.gz" && stat -c '%n %s bytes' "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
        verdict build-extra OK "extra-$sha.tar.gz: $EXTRA"
    else
        verdict build-extra FAIL "rc=$rc; logs/extra.log"
    fi
fi

if step cpubuild; then
    echo "=== cpubuild $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e T="$CPU_TARGETS" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cpu-cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target $T > /work/logs/cpu-build.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf /work/cpu/bin && mkdir -p /work/cpu/bin
            for t in $T; do cp /work/cpu/build/bin/$t /work/cpu/bin/; done
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs /work/ccache
        exit $rc'
    rc=$?
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $CPU_TARGETS; commit $(git -C "$S" rev-parse --short=9 HEAD)"
fi

if step identity; then
    echo "=== identity $(date -Is)"
    K=$WORK/checks/identity
    rm -rf "$K" && mkdir -p "$K"
    T=$(tarball_tree)
    # <name> <base bin dir> <merged bin dir>: the base writes the logits, the merge compares
    ident() {
        local n=$1 bb=$2 mb=$3
        mkdir -p "$K/$n"
        box -v "$K/$n:/k" "$IMAGE" bash -c "$STUB"'
            run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                        | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
            run base '"$bb"'
            ls /k/logits-*.f32 | wc -l > /k/n_base.txt
            run merged '"$mb"'
            rm -f /k/*.f32
            chown -R '"$(id -u):$(id -g)"' /k' > "$K/$n/rc.txt" 2>&1
        # a build from before C565's fix (fork commit fd8276bbc) gives an architecture's Dense and MoE models one logits
        # file, so its MoE model compares against the Dense one's logits and fails (7 architectures): those lines are
        # the base's own; a build with the fix writes the MoE models' files apart, passes them, and with a base from
        # before the fix writes them itself (not compared)
        grep 'logits against' "$K/$n/base.log" | grep -v ', 0 of ' > "$K/$n/base_self.txt"
        local nb nc nw bad sb sm
        nb=$(cat "$K/$n/n_base.txt" 2>/dev/null || echo 0)
        nc=$(grep -c 'logits against' "$K/$n/merged.log")
        nw=$(grep -c 'logits_dump_or_compare: wrote' "$K/$n/merged.log")
        bad=$(grep 'logits against' "$K/$n/merged.log" | grep -v ', 0 of ' | grep -vxFf "$K/$n/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
        sb=$(grep -m 1 '^Summary' "$K/$n/base.log") sm=$(grep -m 1 '^Summary' "$K/$n/merged.log")
        verdict "identity-$n" "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && [ -z "$bad" ] && { [ "$sb" = "$sm" ] || [[ "$sm" == 'Summary: all '* ]]; } && echo OK || echo FAIL)" \
            "base wrote $nb logits files; $nc compared, differ: ${bad:-none}; written by the merge only: $nw; base: $sb; merged: $sm; $(tr '\n' ' ' < "$K/$n/rc.txt")"
    }
    if [ -n "${BASE_BUILD:-}" ] && [ -n "$T" ]; then ident rig "$BASE_BUILD/bin" "$T/bin"; else verdict identity-rig FAIL "no BASE_BUILD or no tarball tree"; fi
    if [ -n "${BASE_CPU:-}" ]; then ident cpu "$BASE_CPU" "$WORK/cpu/bin"; else verdict identity-cpu FAIL "no BASE_CPU"; fi
fi

if step moeskip; then
    echo "=== moeskip $(date -Is)"
    K=$WORK/checks/moeskip
    rm -rf "$K/out" && mkdir -p "$K/bin" "$K/logs"
    for t in test-bytes-switches test-llama-archs llama-quantize test-backend-ops; do cp "$WORK/cpu/bin/$t" "$K/bin/"; done
    WORK=$K THREADS=$THREADS DOCKER_ARGS="$D" STEPS="switch archs" nice -n 19 bash "$DATA/2026-11-04_moe_skip/run/cpu_checks.sh" \
        > "$K/console.txt" 2>&1
    while IFS=$'\t' read -r s v d; do verdict "moeskip-$s" "$v" "$d"; done < "$K/out/verdicts.tsv"
fi

if step coldtier; then
    echo "=== coldtier $(date -Is)"
    K=$WORK/checks/coldtier
    mkdir -p "$K"
    : > "$K/checks.tsv"
    : "${TEXT:?coldtier needs TEXT}"
    # env: as prefix assignments, ${CPUS:+...} would end the assignments (the next one runs as the command) and BIN and
    # MOUNTS would expand $WORK as $K
    env WORK="$K" REPO="$REPO" BIN="$WORK/cpu/bin" TEXT="$TEXT" THREADS="$THREADS" ${CPUS:+CPUS=$CPUS} \
        MOUNTS="$WORK $(dirname "$TEXT") ${MOUNTS:-}" STEPS="random moeids" \
        nice -n 19 bash "$DATA/2026-11-04_cold_tier/run/cpu_checks.sh" > "$K/console.txt" 2>&1
    while IFS=$'\t' read -r _t s v d; do verdict "coldtier-$s" "$v" "$d"; done < "$K/checks.tsv"
fi

if step combo; then
    echo "=== combo $(date -Is)"
    : "${TEXT:?combo needs TEXT}"
    K=$WORK/checks/combo
    rm -rf "$K" && mkdir -p "$K"
    UM=$WORK/checks/coldtier/random/qwen4exp-moe-sharp.gguf
    TM=$WORK/checks/coldtier/checks/random/qwen4exp-moe-sharp-h24/qwen4exp-moe-sharp.gguf
    if [ ! -f "$UM" ] || [ ! -f "$TM" ]; then
        verdict combo FAIL "no random models (run the coldtier step first): $UM $TM"
    else
        # <tag> <model> <LLAMA_MOE_TAIL or ""> <ubatch>: llama-perplexity's KL base file of the run
        ppl() { box ${3:+-e LLAMA_MOE_TAIL=$3} "$IMAGE" nice -n 19 "$WORK/cpu/bin/llama-perplexity" -m "$2" -f "$TEXT" \
                    -c 128 -b 128 -ub "$4" --chunks 4 -t "$THREADS" --no-repack --kl-divergence-base "$K/$1.kld" \
                    > "$K/$1.log" 2>&1 || echo "$1 rc=$?" >> "$K/rc.txt"; }
        same() { [ -s "$K/$1.kld" ] && cmp -s "$K/$1.kld" "$K/$2.kld"; }
        for ub in 128 4 1; do
            ppl u0-$ub "$UM" ""               $ub   # untiered, no cut
            ppl ug-$ub "$UM" "$CUT,graph=1"   $ub   # untiered, graph-level cut
            ppl uo-$ub "$UM" "$CUT"           $ub   # untiered, the op (-1 ids up to 8 tokens)
            ppl t0-$ub "$TM" ""               $ub   # tiered, no cut
            ppl tg-$ub "$TM" "$CUT,graph=1"   $ub   # tiered, graph-level cut
            ppl to-$ub "$TM" "$CUT"           $ub   # tiered, the op as asked: the tiered layers take the graph's cut
            r=""
            same uo-$ub ug-$ub || r="$r untiered op != graph;"
            same to-$ub tg-$ub || r="$r tiered op != tiered graph;"
            same tg-$ub ug-$ub || r="$r tiered graph != untiered graph;"
            same t0-$ub u0-$ub || r="$r tiered != untiered without the cut;"
            same tg-$ub t0-$ub && r="$r the cut changes nothing;"
            verdict "combo ub $ub" "$([ -z "$r" ] && echo OK || echo FAIL)" "${r:-op = graph untiered and tiered, tiered = untiered, the cut acts} ($CUT)"
        done
        box -e LLAMA_MOE_TAIL="$CUT" "$IMAGE" nice -n 19 "$WORK/cpu/bin/test-moe-ids" -m "$TM" -t "$THREADS" > "$K/moeids.log" 2>&1
        rc=$?
        verdict "combo moeids" "$([ $rc = 0 ] && echo OK || echo FAIL)" "tiered model with LLAMA_MOE_TAIL=$CUT: $(grep -E '^test-moe-ids:|checks,' "$K/moeids.log" | tr '\n' ' ')"
        rm -f "$K"/*.kld
    fi
fi

if step adraft; then
    echo "=== adraft $(date -Is)"
    K=$WORK/checks/adraft
    mkdir -p "$K"
    box "$IMAGE" nice -n 19 "$WORK/cpu/bin/test-spec-adapt" > "$K/unit.log" 2>&1
    rc=$?
    verdict adraft "$([ $rc = 0 ] && echo OK || echo FAIL)" "test-spec-adapt rc=$rc: $(tail -n 1 "$K/unit.log")"
fi

if step units; then
    echo "=== units $(date -Is)"
    K=$WORK/checks/units
    rm -rf "$K" && mkdir -p "$K"
    # <check> <binary> [args]: exit status 0 passes; the log's last line goes into the detail
    unit() {
        local n=$1 b=$2; shift 2
        box "$IMAGE" nice -n 19 "$WORK/cpu/bin/$b" "$@" > "$K/$n.log" 2>&1
        local rc=$?
        verdict "unit $n" "$([ $rc = 0 ] && echo OK || echo FAIL)" "$b $* rc=$rc: $(grep -av '^ *$' "$K/$n.log" | tail -n 1 | cut -c1-160)"
    }
    unit blockver   test-spec-block
    unit specdump2  test-spec-dump
    unit draft1card test-draft-mem
    unit n2p        test-quantize-fns
    unit hopcast4   test-hop-cast --unit-only
    UM=$WORK/checks/coldtier/random/qwen4exp-moe-sharp.gguf
    if [ -f "$UM" ]; then
        unit routeonce test-route-once -m "$UM" -t "$THREADS"
    else
        verdict "unit routeonce" FAIL "no random model (run the coldtier step first): $UM"
    fi
fi

date -Is > "$WORK/DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
