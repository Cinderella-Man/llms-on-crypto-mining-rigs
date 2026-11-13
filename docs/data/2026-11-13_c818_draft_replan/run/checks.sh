#!/usr/bin/env bash
# C818 (2026-11-13): LLAMA_SCHED_PER_SIZE_RESERVE=2, C803's per-size plans plus the shared scheduler planned over the
# whole memory again whenever it re-plans (off by default), on pascal-p106-v4 65ac9a47d: built once and checked on a
# build machine, CPU only. The method is C813's (docs/data/2026-11-12_c813_hrow_half/run/checks.sh) with C803's
# re-plan, graph-keep and server checks (docs/data/2026-11-12_c803_draft_realloc/run/checks.sh). Every build and test
# runs in a container without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so
# one runs at a time on a shared machine.
# A CPU has one buffer type, so the shared scheduler's cards never change there; LLAMA_SCHED_TEST_REPLAN=1 (a test
# switch of the branch) makes the first shared-scheduler pass of more than 8 tokens plan it from its own graph over the
# memory as it is, which is what the rig's first catch-up does after its change of cards (R130).
# Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1 EXTRA_TARGETS=test-llama-archs: sm_61, no AVX,
#           test-backend-ops and test-llama-archs in bin/) in $M/main/rig
#   names   C770's switch list in the build (src/build/common/switch-names.cpp): its count, and LLAMA_SCHED_TEST_REPLAN in
#           it and in the tarball's binaries
#   sass    the change is host and test code only (under ggml/ two plan-count getters in ggml-alloc.c and
#           ggml-backend.cpp, no CUDA file): cuobjdump -sass and -ptx of
#           bin/llama-server of the new tarball and of BASE_TREE (the tip tarball of 65ac9a47d, C825), anonymous namespace
#           hashes masked, cut into functions (the day merge's sass_attrib.py); the PTX also with the numbers after the
#           hashes masked (C779's ptx_names.sh)
#   cpu     a static native CPU build of the tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; test-llama-archs and
#           llama-server in $M/cpu/bin
#   replan  test-llama-archs --mtp-replan (C803's three arms and C818's three: plan+cards, whole+cards, whole) of the CPU
#           build on seeds SEEDS and of the rig tarball (its CPU code, no GPU) on seed 1
#   keep    test-llama-archs --mtp-graph-keep (C619's) of the CPU build with LLAMA_SCHED_PER_SIZE_RESERVE unset, at 1, at 2,
#           and at 2 with LLAMA_SCHED_TEST_REPLAN=1: each must pass, with the same rows up to the graph builds
#   server  C803's run/trace.sh: llama-server on the CPU with test-llama-archs's tiny qwen4exp model and its separate MTP
#           draft, FN84's requests, in ARMS (R60's and R85's lines with both switches at 1 and at 2, each also with
#           LLAMA_SCHED_TEST_REPLAN=1); the draft's re-plans in passes of more than 8 tokens per arm, the shared
#           scheduler's plans over the whole memory, and every arm's replies against its line's first arm
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the new one compares
#           with no switch, LLAMA_SCHED_PER_SIZE_RESERVE=2, LLAMA_SCHED_TEST_REPLAN=1, LLAMA_SCHED_PER_SIZE=1 and the last
#           with =2 and with =2 and the test switch (the switch active in every context)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   merged  FINAL_BRANCH (pascal-p106-v4) once BRANCH is merged into it: a static native CPU build of its tip,
#           test-llama-archs against the base's logits with no switch and with the switch active (the "every switch off"
#           check of the merged tree), and --mtp-replan -s 1 (MERGED_STEPS="build off replan")
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (65ac9a47d in full), BASE_TREE (the base's rig
#   tarball tree), BASE_CPU (bin/ of 65ac9a47d's static CPU build: test-llama-archs), SEED_SRC (a llama.cpp clone with
#   the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default
#   pascal-p106-v4-b12246-c818), FINAL_BRANCH (default pascal-p106-v4), JOBS (8), THREADS (8), SEEDS ("1 2 3"),
#   DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS, ARMS (trace.sh's)
# Output: $M/summary.tsv (row, check, verdict, detail), $M/replan, $M/keep, $M/server, $M/off, $M/sass, $M/logs; done
#   when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c818}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-65ac9a47d6933025bc5d11db82158700d1352ddb}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
SEEDS=${SEEDS:-1 2 3}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-rig names sass cpu replan keep server off c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
PTXNAMES=$D/2026-11-12_c779_graph_replay/run/ptx_names.sh
C803RUN=$D/2026-11-12_c803_draft_realloc/run
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
# the rig build's binaries load libcuda; a container without GPUs takes the toolkit's stub (C703)
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
SW=LLAMA_SCHED_PER_SIZE_RESERVE
TS=LLAMA_SCHED_TEST_REPLAN
SL=LLAMA_DRAFT_SHORTLIST=100,8,2
QSA=LLAMA_QSA_INDEXER_ROWS=1,LLAMA_QSA_KPOOL_DEV=1
KP=LLAMA_MTP_GRAPH_KEEP=1
# <line><arm>@<switches>@<server args>: the line's first arm is its reference for the replies
ARMS=${ARMS:-r60plan@$QSA,$SL,$KP,$SW=1@--spec-draft-n-ubatch~512 r60whole@$QSA,$SL,$KP,$SW=2@--spec-draft-n-ubatch~512 \
r60plancards@$QSA,$SL,$KP,$SW=1,$TS=1@--spec-draft-n-ubatch~512 r60wholecards@$QSA,$SL,$KP,$SW=2,$TS=1@--spec-draft-n-ubatch~512 \
r85plan@$SL,$KP,$SW=1@--spec-draft-n-ubatch~128 r85whole@$SL,$KP,$SW=2@--spec-draft-n-ubatch~128 \
r85plancards@$SL,$KP,$SW=1,$TS=1@--spec-draft-n-ubatch~128 r85wholecards@$SL,$KP,$SW=2,$TS=1@--spec-draft-n-ubatch~128}
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c818.log"; }
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
log "C818 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1); under ggml/: $(git -C "$SRC" diff --name-only "$BASE_COMMIT" "$TIP" -- ggml | wc -l)"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 EXTRA_TARGETS=test-llama-archs JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); llama-server md5 $(md5sum < "$T/bin/llama-server" | cut -d' ' -f1); test-llama-archs in bin/: $([ -x "$T/bin/test-llama-archs" ] && echo yes || echo no); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step names; then
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    inl=$(grep -c "^    \"$TS\",\$" "$L" 2> /dev/null)
    ins=$(grep -c "^    \"$SW\",\$" "$L" 2> /dev/null)
    inb=$(cat "$T"/bin/llama-server "$T"/lib/*.so* 2> /dev/null | grep -ac "$TS")
    add names "switch list" "$([ -n "$n" ] && [ "$inl" = 1 ] && [ "$ins" = 1 ] && [ "$inb" -gt 0 ] && echo OK || echo FAIL)" \
        "$n names in the build's list; $TS in it: $inl, $SW: $ins; lines of bin/llama-server and lib/ holding $TS: $inb"
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    # <label> <llama-server>: its SASS and PTX cut into functions (cuobjdump in the CUDA image, the cutting on the host)
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin c818 "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/c818.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/c818.$w.tsv" "$S/base.$w.tsv" && echo "the same file" || echo "files differ")
        add sass "llama-server $w against the base" "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/c818.$w.tsv") functions ($(wc -l < "$S/base.$w.tsv") in the base's); dumps: $same; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
    done
    NEW="$T/bin/llama-server" BASE="$BASE_TREE/bin/llama-server" OUT="$S/ptxnames" LOCK="$LOCK" IMAGE="$IMAGE" \
        nice -n 19 bash "$PTXNAMES" > "$M/logs/ptxnames.log" 2>&1
    add sass "llama-server ptx, internal-name numbers masked" "$(grep -q '^verdict: SAME' "$S/ptxnames/ptx_names.txt" && echo SAME || echo DIFFERS)" \
        "$(grep -v '^verdict' "$S/ptxnames/ptx_names.txt" | tr -s ' ' | tr '\n' ';')"
    rm -f "$S"/ptxnames/*.ptx
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
    cp "$C/tree/build/bin/test-llama-archs" "$C/tree/build/bin/llama-server" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && [ -x "$C/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cut -c1-140 "$C/tree.txt"); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

# <dir> <row> <label> <bin> <image> <prefix> <flag> <env> <seeds>: test-llama-archs <flag> per seed, one summary row each,
# and the table rows of every log (with the seed) in <dir>/<label>.rows
archs_check() {
    local K=$1 row=$2 name=$3 bin=$4 img=$5 pre=$6 flag=$7 envs=$8 seeds=$9 s rc sum
    mkdir -p "$K"
    if [ ! -x "$bin/test-llama-archs" ]; then
        add "$row" "$name" SKIP "no test-llama-archs in $bin"
        return
    fi
    : > "$K/$name.rows"
    for s in $seeds; do
        box "$img" bash -c "$pre"'
            env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs '"$flag"' -s '"$s"' -v 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" \
                > '"$K/$name-s$s.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        sum=$(grep -a -m 1 '^Summary' "$K/$name-s$s.log")
        add "$row" "$name -s $s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" \
            "rc=$rc; ${envs:-no switch}; $flag; $sum"
        grep -a '^|' "$K/$name-s$s.log" | tr -s ' ' | sed "s/^/s$s/" >> "$K/$name.rows"
    done
}

if step replan; then
    log "replan"
    K=$M/replan
    rm -rf "$K" && mkdir -p "$K"
    archs_check "$K" replan cpu "$C/bin" "$CPU_IMAGE" true --mtp-replan "" "$SEEDS"
    archs_check "$K" replan rig "${T:+$T/bin}" "$IMAGE" "$STUB" --mtp-replan "" "1"
    log "replan done"
fi

if step keep; then
    log "keep"
    K=$M/keep
    rm -rf "$K" && mkdir -p "$K"
    for arm in "off:" "plan:$SW=1" "whole:$SW=2" "wholecards:$SW=2 $TS=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        archs_check "$K" keep "$tag" "$C/bin" "$CPU_IMAGE" true --mtp-graph-keep "$envs" "1"
    done
    # the graph builds and the drafts as the table prints them; the compute column differs by the switch's plans
    for tag in plan whole wholecards; do
        if cmp -s <(cut -d'|' -f1-9 "$K/off.rows") <(cut -d'|' -f1-9 "$K/$tag.rows"); then
            add keep "off against $tag" SAME "the same rows up to the builds ($(wc -l < "$K/off.rows") rows); compute KiB off: $(cut -d'|' -f10 "$K/off.rows" | tr -d ' ' | tr '\n' ' '), $tag: $(cut -d'|' -f10 "$K/$tag.rows" | tr -d ' ' | tr '\n' ' ')"
        else
            add keep "off against $tag" DIFFERS "$(diff <(cut -d'|' -f1-9 "$K/off.rows") <(cut -d'|' -f1-9 "$K/$tag.rows") | grep -c '^[<>]') lines"
        fi
    done
    log "keep done"
fi

if step server; then
    log "server"
    S=$M/server
    mkdir -p "$S"
    locked env WORK="$S" BIN="$C/bin" ARMS="$ARMS" THREADS=4 IMAGE="$CPU_IMAGE" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$C803RUN/trace.sh" > "$S/console.txt" 2>&1
    rc=$?
    for a in $ARMS; do
        name=${a%%@*}
        ref=${name:0:3}plan
        # the draft's re-plans in passes of more than 8 tokens (the shared scheduler's) and in all passes
        nsh=$(awk -F'\t' -v n="$name" '$1 == n && $4 == "draft" && $5 > 8 && $8 == 1' "$S/trace.tsv" | wc -l)
        nall=$(awk -F'\t' -v n="$name" '$1 == n && $4 == "draft" && $8 == 1' "$S/trace.tsv" | wc -l)
        # the plan lines, given to the evaluation whose trace line follows them: draft (mtp) or target
        read -r dtest dwhole ttest twhole < <(awk '/LLAMA_SCHED_TEST_REPLAN: the shared scheduler planned/ { t = 1 }
            /sched_shared_plan: / { w++ }
            /ubatch-trace: mtp / { dt += t; dw += w; t = 0; w = 0 }
            /ubatch-trace: (dec|enc) / { tt += t; tw += w; t = 0; w = 0 }
            END { printf "%d %d %d %d\n", dt, dw, tt, tw }' "$S/$name.server.log")
        again=$(grep -ac 'needed a new plan after the plan over the whole memory' "$S/$name.server.log")
        g=$(grep -ac 'galloc-grow' "$S/$name.server.log")
        p=$(grep -ac 'sched_small_plan: ' "$S/$name.server.log")
        rep=$(docker run --rm --network none -v "$S:$S:ro" -v "$C803RUN:$C803RUN:ro" "$CPU_IMAGE" python3 -I "$C803RUN/draft_trace.py" replies "$S/$ref.replies.json" "$S/$name.replies.json")
        v=OK
        case $name in
            *wholecards) [ "$nsh" = 1 ] && [ "$nall" = 1 ] && [ "$dtest" = 1 ] && [ "$dwhole" = 1 ] && [ "$again" = 0 ] || v=FAIL ;;
            *plancards)  [ "$nsh" -gt 0 ] && [ "$dtest" = 1 ] && [ "$dwhole" = 0 ] || v=FAIL ;;
            *whole)      [ "$nall" = 0 ] && [ "$dwhole" = 0 ] && [ "$dtest" = 0 ] || v=FAIL ;;
            *plan)       [ "$nall" = 0 ] && [ "$dwhole" = 0 ] || v=FAIL ;;
        esac
        [ "${rep%% *}" != identical ] && v=FAIL
        add server "$name" "$v" "rc=$rc; draft re-plans $nall, in passes of more than 8 tokens $nsh; draft plans from a graph in hand (test) $dtest, over the whole memory $dwhole, a new plan after one $again; target's $ttest, $twhole; $(grep -a -P "^$name\tdraft\tevaluations" "$S/trace_summary.txt" | cut -f3- | tr '\t' ' '); buffer growths $g; per-size plans $p; replies against $ref: $rep"
    done
    log "server done"
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
            "rc=$rc; ${envs:-no switch}; against the base's $(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l) files: $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; test plans $(grep -ac 'LLAMA_SCHED_TEST_REPLAN: the shared scheduler planned' "$A/$tag.log"), whole plans $(grep -ac 'sched_shared_plan: ' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
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
    log "off"
    archs_base "$M/off"
    archs_arms "$M/off" "$C/bin" "tip:" "whole:$SW=2" "test:$TS=1" "persize:LLAMA_SCHED_PER_SIZE=1" \
        "persize-whole:LLAMA_SCHED_PER_SIZE=1 $SW=2" "persize-whole-test:LLAMA_SCHED_PER_SIZE=1 $SW=2 $TS=1"
    rm -f "$M"/off/base/logits-*.f32
    log "off done"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c818=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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

if step merged; then
    MS=${MERGED_STEPS:-build off replan}
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    git -C "$SRC" merge-base --is-ancestor "$TIP" "$F" || { add merged "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH ${F:0:9} does not hold ${TIP:0:9}"; date -Is > "$M/DONE"; exit 1; }
    add merged "$FINAL_BRANCH" OK "${F:0:9} ($(git -C "$SRC" log -1 --format=%s "$F" | cut -c1-120)); merges since the base: $(git -C "$SRC" rev-list --merges --count "$BASE_COMMIT..$F"); $(git -C "$SRC" log --merges --format='%h %s' "$BASE_COMMIT..$F" | cut -c1-60 | tr '\n' ';')"
    MC=$M/merged
    if [[ " $MS " == *" build "* ]]; then
        rm -rf "$MC/tree" "$MC/bin" && mkdir -p "$MC/tree/src" "$MC/bin"
        [ -d "$MC/ccache" ] || cp -al "$C/ccache" "$MC/ccache" 2> /dev/null || mkdir -p "$MC/ccache"
        git -C "$SRC" archive "$F" | tar -x -C "$MC/tree/src"
        box -e CCACHE_DIR="$MC/ccache" -e CCACHE_BASEDIR="$MC/tree" "$CPU_IMAGE" bash -c '
            cmake -S '"$MC/tree/src"' -B '"$MC/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$MC/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$MC/tree/build"' -j '"$JOBS"' -- -k > '"$MC/build.log"' 2>&1'
        rc=$?
        cp "$MC/tree/build/bin/test-llama-archs" "$MC/bin/" 2> /dev/null
        add merged "cpu build" "$([ $rc = 0 ] && [ -x "$MC/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; ${F:0:9}; static, every target; $(grep -ac 'warning:' "$MC/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$MC/build.log") error lines"
        rm -rf "$MC/tree"
    fi
    if [[ " $MS " == *" off "* ]]; then
        archs_base "$M/moff"
        archs_arms "$M/moff" "$MC/bin" "merged:" "merged-persize-whole-test:LLAMA_SCHED_PER_SIZE=1 $SW=2 $TS=1"
        rm -f "$M"/moff/base/logits-*.f32
    fi
    if [[ " $MS " == *" replan "* ]]; then
        archs_check "$M/mreplan" merged replan "$MC/bin" "$CPU_IMAGE" true --mtp-replan "" "1"
    fi
fi

log "done"
date -Is > "$M/DONE"
