#!/usr/bin/env bash
# C803 (Q1407) on a build machine, CPU only: LLAMA_SCHED_PER_SIZE_RESERVE=1 plans each of the MTP draft's per-size
# schedulers (LLAMA_MTP_GRAPH_KEEP=1) once over the whole memory, so a growing context never makes the draft's graph
# re-plan. Every build and test runs in a container without GPUs at --cpu-shares=256 and nice 19, behind LOCK (flock),
# one at a time.
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build of the branch's tip (test-llama-archs, llama-server, test-backend-ops) at
#            $M/cpu/tip/tree, one ccache
#   replan   test-llama-archs --mtp-replan (the shared scheduler alone, LLAMA_MTP_GRAPH_KEEP=1, and both switches over
#            six turns of a growing context): must pass
#   keep     test-llama-archs --mtp-graph-keep (C619's check) without the switch and with LLAMA_SCHED_PER_SIZE_RESERVE=1
#            for the whole process: both must pass, with the same graph builds
#   server   run/trace.sh: llama-server on the CPU with test-llama-archs's tiny qwen4exp model and its separate MTP draft,
#            FN84's requests, in ARMS (by default R60's line and R85's, each without and with LLAMA_MTP_GRAPH_KEEP=1 and
#            with both switches); the draft's re-plans per arm, and every arm's replies against its line's first arm
#   off      test-llama-archs -b CPU -s 1: BASE_CPU (the base's static CPU build) writes every architecture's logits,
#            the tip compares with no switch and with LLAMA_SCHED_PER_SIZE_RESERVE=1
#   rig      the rig tarball of the branch (scripts/build_llamacpp_docker.sh, TESTS=1) in $M/main/rig
#   c565     2026-11-05_c565_ladder4fix/run/cpu_tests.sh on the tip, static and then shared libraries
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_CPU (bin/ of the base's static CPU build, with
#   test-llama-archs), SEED_SRC (a llama.cpp clone with the fork as remote "fork", https); optional SEED_CCACHE (the rig
#   build's), SEED_CPU_CCACHE, SEED_C565_CCACHE; BRANCH (default pascal-p106-v4-b12246-c803), BASE_COMMIT (default
#   c96561876 in full), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS, ARMS (run/trace.sh's)
# Output: $M/summary.tsv (row, check, verdict, detail), $M/logs/c803.log, each step's folder; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c803}
BASE_COMMIT=${BASE_COMMIT:-c96561876e1d32548bacf1df687b410214d8bb52}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-cpu replan keep server off rig c565}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
RUN=$REPO/docs/data/2026-11-12_c803_draft_realloc/run
SRC=$M/main/rig/src
SW=LLAMA_SCHED_PER_SIZE_RESERVE
SL=LLAMA_DRAFT_SHORTLIST=100,8,2
QSA=LLAMA_QSA_INDEXER_ROWS=1,LLAMA_QSA_KPOOL_DEV=1
ARMS=${ARMS:-r60@$QSA,$SL@--spec-draft-n-ubatch~512 r60keep@$QSA,$SL,LLAMA_MTP_GRAPH_KEEP=1@--spec-draft-n-ubatch~512 \
r60plan@$QSA,$SL,LLAMA_MTP_GRAPH_KEEP=1,$SW=1@--spec-draft-n-ubatch~512 r85@$SL@--spec-draft-n-ubatch~128 \
r85keep@$SL,LLAMA_MTP_GRAPH_KEEP=1@--spec-draft-n-ubatch~128 r85plan@$SL,LLAMA_MTP_GRAPH_KEEP=1,$SW=1@--spec-draft-n-ubatch~128}
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c803.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# shellcheck disable=SC2086
box()    { locked docker run --rm $DA --network none -v "$M:$M" "$@"; }

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
BASE_COMMIT=$(git -C "$SRC" rev-parse "$BASE_COMMIT")
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
log "C803 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add branch commits OK "$(git -C "$SRC" log --format='%h %s' "$BASE_COMMIT..$TIP" | tr '\n' ';'); $(git -C "$SRC" diff --shortstat "$BASE_COMMIT" "$TIP")"
B=$M/cpu/tip/bin

if step cpu; then
    [ -d "$M/cpu/ccache" ] || { mkdir -p "$M/cpu"; if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -a "$SEED_CPU_CCACHE" "$M/cpu/ccache"; else mkdir -p "$M/cpu/ccache"; fi; }
    A=$M/cpu/tip
    targets="test-llama-archs llama-server test-backend-ops"
    rm -rf "$A" && mkdir -p "$A/tree/src" "$A/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$A/tree/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$A/tree.txt"
    box -e CCACHE_DIR="$M/cpu/ccache" -e CCACHE_BASEDIR="$A/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$A/tree/src"' -B '"$A/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$A/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$A/tree/build"' -j '"$JOBS"' --target '"$targets"' > '"$A/build.log"' 2>&1'
    rc=$?
    ok=1
    for t in $targets; do cp "$A/tree/build/bin/$t" "$A/bin/" 2> /dev/null || ok=0; done
    add cpu "tip build" "$([ $rc = 0 ] && [ $ok = 1 ] && echo OK || echo FAIL)" \
        "rc=$rc; $(tr '\n' ';' < "$A/tree.txt") static; $targets; $(grep -ac 'warning:' "$A/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$A/build.log") error lines"
    rm -rf "$A/tree/build"
fi

if step replan; then
    P=$M/replan
    mkdir -p "$P"
    box "$CPU_IMAGE" bash -c 'nice -n 19 '"$B"'/test-llama-archs --mtp-replan -s 1 -v 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$P/replan.log"'; exit ${PIPESTATUS[0]}'
    rc=$?
    add replan "--mtp-replan" "$([ $rc = 0 ] && grep -aq '^Summary: all .* passed' "$P/replan.log" && echo PASS || echo FAIL)" \
        "rc=$rc; $(grep -a '^|' "$P/replan.log" | tr -s ' ' | tr '\n' ';'); $(grep -a '^Summary' "$P/replan.log")"
fi

if step keep; then
    P=$M/keep
    mkdir -p "$P"
    for arm in "off:" "on:$SW=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        box "$CPU_IMAGE" bash -c 'env '"$envs"' nice -n 19 '"$B"'/test-llama-archs --mtp-graph-keep -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$P/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        grep -a '^|' "$P/$tag.log" | tr -s ' ' > "$P/$tag.rows"
        add keep "$tag" "$([ $rc = 0 ] && grep -aq '^Summary: all .* passed' "$P/$tag.log" && echo PASS || echo FAIL)" \
            "rc=$rc; ${envs:-no switch}; $(grep -a '^Summary' "$P/$tag.log")"
    done
    # the graph builds and the drafts as the table prints them; the compute column differs by the switch's plans
    if cmp -s <(cut -d'|' -f1-9 "$P/off.rows") <(cut -d'|' -f1-9 "$P/on.rows"); then
        add keep "off against on" SAME "the same rows up to the builds ($(wc -l < "$P/off.rows") rows); compute KiB off: $(cut -d'|' -f10 "$P/off.rows" | tr -d ' ' | tr '\n' ' '), on: $(cut -d'|' -f10 "$P/on.rows" | tr -d ' ' | tr '\n' ' ')"
    else
        add keep "off against on" DIFFERS "$(diff <(cut -d'|' -f1-9 "$P/off.rows") <(cut -d'|' -f1-9 "$P/on.rows") | grep -c '^[<>]') lines"
    fi
fi

if step server; then
    S=$M/server
    mkdir -p "$S"
    locked env WORK="$S" BIN="$B" ARMS="$ARMS" THREADS=4 IMAGE="$CPU_IMAGE" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        bash "$RUN/trace.sh" > "$S/console.txt" 2>&1
    rc=$?
    for a in $ARMS; do
        name=${a%%@*}
        line=${name%keep}; line=${line%plan}
        n=$(awk -F'\t' -v n="$name" '$1 == n && $4 == "draft" && $8 == 1' "$S/trace.tsv" | wc -l)
        g=$(grep -ac 'galloc-grow' "$S/$name.server.log")
        p=$(grep -ac 'sched_small_plan: ' "$S/$name.server.log")
        rep=$(docker run --rm --network none -v "$S:$S:ro" -v "$RUN:$RUN:ro" "$CPU_IMAGE" python3 -I "$RUN/draft_trace.py" replies "$S/$line.replies.json" "$S/$name.replies.json")
        v=OK
        [ "${name%plan}" != "$name" ] && [ "$n" != 0 ] && v=FAIL
        [ "${rep%% *}" != identical ] && v=FAIL
        add server "$name" "$v" "rc=$rc; draft re-plans $n; $(grep -a -P "^$name\tdraft\tevaluations" "$S/trace_summary.txt" | cut -f3- | tr '\t' ' '); buffer growths $g; per-size plans $p; replies against $line: $rep"
    done
fi

if step off; then
    A=$M/off
    BC=${BASE_CPU:?BASE_CPU=<bin of the base static CPU build>}
    rm -rf "$A" && mkdir -p "$A/base"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$BC:$BC:ro" "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- nice -n 19 '"$BC"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A"'/base.log; exit ${PIPESTATUS[0]}'
    rc=$?
    nb=$(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l)
    add off "base writes" "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" "rc=$rc; $BC: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log")"
    for arm in "tip:" "plan:$SW=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        box "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- '"$envs"' nice -n 19 '"$B"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        wrote=$(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log")
        add off "$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && [ "$wrote" = 0 ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; against the base's $nb files: $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -rf "$A/base"
fi

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        W=$M/c565-$v
        mkdir -p "$W"
        [ -d "$W/ccache" ] || [ -z "${SEED_C565_CCACHE:-}" ] || cp -a "$SEED_C565_CCACHE" "$W/ccache"
        env SRC="$SRC" WORK="$W" TREES="c803=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="--label pascal-quants=1 ${DOCKER_ARGS:-}" nice -n 19 \
            bash "$REPO/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        tail -n +2 "$W/summary.tsv" 2> /dev/null | awk -F'\t' -v r="c565 $v" 'BEGIN { OFS = "\t" }
            NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next } { print r, $1, $2, $3 }' \
            | tee -a "$M/summary.tsv"
        rm -rf "$W/tree"
    done
fi

log "done"
date -Is > "$M/DONE"
