#!/usr/bin/env bash
# C815 (2026-11-12): LLAMA_SPEC_ADAPT's rule taken while drafting (LLAMA_SPEC_ADAPT_ONLINE, off by default), on
# pascal-p106-v4 b9a1c1fa7: built once and checked on a build machine, CPU only. The method is C779's
# (docs/data/2026-11-12_c779_graph_replay/run/checks.sh). Every build and test runs in a container without GPUs, waits
# for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on a shared machine.
# Steps (STEPS, in this order by default):
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; test-llama-archs,
#           llama-server and test-spec-adapt in $M/cpu/bin
#   bsrv    the same of BASE_COMMIT, llama-server and test-llama-archs only, in $M/bsrv (one ccache with cpu)
#   unit    test-spec-adapt twice: the rule, and C815's common_spec_adapt_continue and _keeps_next against
#           common_spec_adapt_choose on 20,000 random drafts (seeded: the same output both times)
#   model   the random qwen4exp MoE model with one MTP block (the tip's test-llama-archs -o --mtp)
#   server  server_check.sh: the base's llama-server with LLAMA_SPEC_ADAPT=1 against the tip's with the switch as built,
#           =1 and =2, with LLAMA_VERIFY_PREPARE, the backend sampler and probabilistic drafting, and the control
#   off     test-llama-archs -b CPU -s 1: the base's (bsrv) writes every architecture's logits, the tip's compares with
#           no switch and with LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2 (nothing there drafts: the same logits)
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX) in $M/main/rig
#   rigbase the same build of BASE_COMMIT in the same folder (its CUDA objects come from the first build's ccache)
#   names   C770's switch list in the rig build: LLAMA_SPEC_ADAPT_ONLINE and LLAMA_SPEC_ADAPT_ONLINE_FAULT in it and in
#           bin/llama-server
#   sass    the change is host code only: cuobjdump -sass and -ptx of the two tarballs' bin/llama-server,
#           anonymous-namespace hashes masked, cut into functions (the day merge's sass_attrib.py): every function the
#           base's, none missing or added
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build trees removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip,
#           test-spec-adapt, and test-llama-archs against the base's logits with no switch and with the switch (the
#           "every switch off" check of the merged tree; MERGED_STEPS="build off server" adds server_check.sh's d arms)
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (b9a1c1fa7 in full), SEED_SRC (a llama.cpp
#   clone with the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default
#   pascal-p106-v4-b12246-c815), FINAL_BRANCH (default pascal-p106-v4), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE,
#   CPU_IMAGE, SRV_IMAGE (CPU_IMAGE with curl and jq), STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/server, $M/sass, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c815}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-b9a1c1fa7}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-cpu bsrv unit model server off rig names rigbase sass c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-12_c815_adapt_while_drafting/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
SW=LLAMA_SPEC_ADAPT_ONLINE
ON="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2"
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c815.log"; }
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
# <dir> <commit> <ccache> <targets or "all">: a static native CPU build in <dir>/tree, the named binaries into <dir>/bin
cpu_build() {
    local B=$1 commit=$2 cc=$3 targets=$4 tgt=""
    rm -rf "$B/tree" "$B/bin" && mkdir -p "$B/tree/src" "$B/bin"
    git -C "$SRC" archive "$commit" | tar -x -C "$B/tree/src"
    echo "commit $commit $(git -C "$SRC" log -1 --format=%s "$commit")" > "$B/tree.txt"
    [ "$targets" = all ] || tgt="--target $targets"
    box -e CCACHE_DIR="$cc" -e CCACHE_BASEDIR="$B/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$B/tree/src"' -B '"$B/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$B/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$B/tree/build"' -j '"$JOBS"' '"$tgt"' -- -k > '"$B/build.log"' 2>&1'
}

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
BASE_FULL=$(git -C "$SRC" rev-parse "$BASE_COMMIT^{commit}") || { echo "no $BASE_COMMIT"; exit 1; }
git -C "$SRC" merge-base --is-ancestor "$BASE_FULL" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
for img in "$CPU_IMAGE" "$SRV_IMAGE"; do
    docker image inspect "$img" > /dev/null 2>&1 || { echo "no image $img"; exit 1; }
done
log "C815 checks: $BRANCH at ${TIP:0:9}, base ${BASE_FULL:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_FULL" "$TIP" | tail -n 1); under ggml/: $(git -C "$SRC" diff --name-only "$BASE_FULL" "$TIP" -- ggml | wc -l)"

C=$M/cpu
BS=$M/bsrv
if step cpu; then
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then mkdir -p "$C" && cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    cpu_build "$C" "$TIP" "$C/ccache" all
    rc=$?
    cp "$C/tree/build/bin/test-llama-archs" "$C/tree/build/bin/llama-server" "$C/tree/build/bin/test-spec-adapt" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && [ -x "$C/bin/llama-server" ] && [ -x "$C/bin/test-spec-adapt" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cut -c1-140 "$C/tree.txt"); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi

if step bsrv; then
    cpu_build "$BS" "$BASE_FULL" "$C/ccache" "llama-server test-llama-archs"
    rc=$?
    cp "$BS/tree/build/bin/llama-server" "$BS/tree/build/bin/test-llama-archs" "$BS/bin/" 2> /dev/null
    add bsrv build "$([ $rc = 0 ] && [ -x "$BS/bin/llama-server" ] && [ -x "$BS/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cut -c1-140 "$BS/tree.txt"); static, llama-server and test-llama-archs; $(grep -acE 'error:|undefined reference' "$BS/build.log") error lines"
    rm -rf "$BS/tree"
fi

if step unit; then
    for i in 1 2; do
        box "$CPU_IMAGE" bash -c 'nice -n 19 '"$C"'/bin/test-spec-adapt > '"$M"'/logs/unit'"$i"'.log 2>&1'
        rc=$?
        add unit "test-spec-adapt run $i" "$([ $rc = 0 ] && grep -qE '^([0-9]+)/\1 checks passed' "$M/logs/unit$i.log" && echo OK || echo FAIL)" \
            "rc=$rc; $(grep -a '^online:' "$M/logs/unit$i.log" | head -n 1); $(tail -n 1 "$M/logs/unit$i.log")"
    done
    cmp -s "$M/logs/unit1.log" "$M/logs/unit2.log"
    add unit "the two runs' output" "$([ $? = 0 ] && echo SAME || echo DIFFERS)" "seeded: the same counts both times"
fi

S=$M/server
if step model; then
    rm -rf "$S/models" && mkdir -p "$S/models"
    box "$CPU_IMAGE" bash -c "nice -n 19 $C/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp > $S/models/save.log 2>&1"
    rc=$?
    add model "qwen4exp with an MTP block" "$([ $rc = 0 ] && [ -s "$S/models/qwen4exp-moe-mtp.gguf" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(ls -l "$S/models/qwen4exp-moe-mtp.gguf" 2> /dev/null | awk '{ print $5 " B" }'); md5 $(md5sum < "$S/models/qwen4exp-moe-mtp.gguf" | cut -d' ' -f1)"
fi

if step server; then
    rm -rf "$S/out"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$REPO:$REPO:ro" "$SRV_IMAGE" bash -c \
        "BASE=$BS/bin BIN=$C/bin MODEL=$S/models/qwen4exp-moe-mtp.gguf OUT=$S/out THREADS=$THREADS \
         SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    add server "server_check.sh" "$([ $rc = 0 ] && tail -n 1 "$S/out/verdict.txt" | grep -qx PASS && echo PASS || echo FAIL)" \
        "rc=$rc; $(grep -c ' = ' "$S/out/verdict.txt" 2> /dev/null) arms equal to their reference; $(grep -E 'pfault|dvpb' "$S/out/verdict.txt" | cut -c1-160 | tr '\n' ';')"
    log "server rc=$rc"
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
    local A=$1 rc nb
    rm -rf "$A" && mkdir -p "$A/base"
    box "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- nice -n 19 '"$BS"'/bin/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A"'/base.log; exit ${PIPESTATUS[0]}'
    rc=$?
    nb=$(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l)
    add "$(basename "$A")" "base writes" "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" "rc=$rc; ${BASE_FULL:0:9}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log")"
}

if step off; then
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "online:$ON"
    rm -f "$M"/off/base/logits-*.f32
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

# before rigbase, which builds the base in the same source folder
if step names; then
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    inl=0
    for s in "$SW" "${SW}_FAULT"; do
        inl=$((inl + $(grep -c "^    \"$s\",\$" "$L" 2> /dev/null)))
    done
    inb=$(grep -ac "$SW" "$T/bin/llama-server")
    add names "switch list" "$([ -n "$n" ] && [ "$inl" = 2 ] && [ "$inb" -gt 0 ] && echo OK || echo FAIL)" \
        "$n names in the build's list; $SW and ${SW}_FAULT in it: $inl of 2; lines of bin/llama-server holding $SW: $inb"
fi

if step rigbase; then
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "$BASE_FULL" > "$M/logs/rigbase.log" 2>&1
    rc=$?
    TB=$(tree "${BASE_FULL:0:9}")
    add rigbase build "$([ $rc = 0 ] && [ -n "$TB" ] && [ -x "$TB/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$TB").tar.gz md5 $(md5sum < "$TB.tar.gz" | cut -d' ' -f1); $(grep -E '^(commit)' "$TB/BUILDINFO.txt" | tr '\n' ';')"
fi
TB=$(tree "${BASE_FULL:0:9}")

if step sass; then
    SD=$M/sass
    rm -rf "$SD" && mkdir -p "$SD"
    # <label> <llama-server>: its SASS and PTX cut into functions (cuobjdump in the CUDA image, the cutting on the host)
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$SD/$1.$w.tsv"
        done
    }
    dumpbin c815 "$T/bin/llama-server"
    dumpbin base "$TB/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$SD/c815.$w.tsv" "$SD/base.$w.tsv" > "$SD/attrib_$w.txt" 2>&1
        same=$(cmp -s "$SD/c815.$w.tsv" "$SD/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$SD/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$SD/c815.$w.tsv") functions ($(wc -l < "$SD/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$SD/attrib_$w.txt"); $(sed -n '2,$p' "$SD/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    log "sass done"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c815=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
    rm -rf "$C/tree" "$BS/tree"
    add trim "CPU trees" OK "removed (the binaries in cpu/bin and bsrv/bin stay); $(du -sh "$M" | cut -f1) in $M; $(free_gb) GB free"
fi

if step merged; then
    MS=${MERGED_STEPS:-build off}
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); commits since the base: $(git -C "$SRC" rev-list --count "$BASE_FULL..$F"); merges since the base: $(git -C "$SRC" log --merges --format=%s "$BASE_FULL..$F" | cut -c1-40 | tr '\n' ';')"
    MC=$M/merged
    if [[ " $MS " == *" build "* ]]; then
        cpu_build "$MC" "$F" "$C/ccache" all
        rc=$?
        cp "$MC/tree/build/bin/test-llama-archs" "$MC/tree/build/bin/test-spec-adapt" "$MC/tree/build/bin/llama-server" "$MC/bin/" 2> /dev/null
        add merged "cpu build" "$([ $rc = 0 ] && [ -x "$MC/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; ${F:0:9}; static, every target; $(grep -ac 'warning:' "$MC/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$MC/build.log") error lines"
        box "$CPU_IMAGE" bash -c 'nice -n 19 '"$MC"'/bin/test-spec-adapt > '"$M"'/logs/merged-unit.log 2>&1'
        rc=$?
        add merged "test-spec-adapt" "$([ $rc = 0 ] && grep -qE '^([0-9]+)/\1 checks passed' "$M/logs/merged-unit.log" && echo OK || echo FAIL)" \
            "rc=$rc; $(tail -n 1 "$M/logs/merged-unit.log")"
        rm -rf "$MC/tree"
    fi
    if [[ " $MS " == *" off "* ]]; then
        archs_base "$M/moff"
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-online:$ON"
        rm -f "$M"/moff/base/logits-*.f32
    fi
    if [[ " $MS " == *" server "* ]]; then
        rm -rf "$S/mout"
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$M:$M" -v "$REPO:$REPO:ro" "$SRV_IMAGE" bash -c \
            "BASE=$BS/bin BIN=$MC/bin MODEL=$S/models/qwen4exp-moe-mtp.gguf OUT=$S/mout THREADS=$THREADS ARMS='dbase dbuilt don1 don2 dvp' \
             SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/mconsole.txt" 2>&1
        rc=$?
        add merged "server_check.sh, d arms" "$([ $rc = 0 ] && tail -n 1 "$S/mout/verdict.txt" | grep -qx PASS && echo PASS || echo FAIL)" \
            "rc=$rc; $(grep -c ' = ' "$S/mout/verdict.txt" 2> /dev/null) arms equal to dbase"
    fi
fi

log "done"
date -Is > "$M/DONE"
