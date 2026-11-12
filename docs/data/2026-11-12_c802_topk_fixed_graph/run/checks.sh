#!/usr/bin/env bash
# C802 (Q1406) on a build machine, CPU only: C691's one top-k over the output rows (LLAMA_SAMPLING_TOP_K_ROWS=1) with a
# graph whose nodes and their allocation stay the same in every micro-batch, so ggml's allocator never plans the compute
# buffers again between a prompt's middle and its end (FN84). Every build and test runs in a container without GPUs
# at --cpu-shares=256 and nice 19, behind LOCK (flock), one at a time.
# Arms, static native CPU builds from the fork's branch:
#   test  TEST_COMMIT: BASE_COMMIT with C802's test only (the code before the fix)
#   view  the tip with run/view.patch: the fix replaced by a view of all the logits' rows in the pad's place (the node
#         count kept), with the tip's test and trace
#   tip   the branch's tip: the fix, and the allocator's trace line for a place the plan gave no memory
# Steps (STEPS, in this order by default):
#   cpu      the three arms' static CPU builds (test-backend-sampler; tip also test-llama-archs and test-backend-ops),
#            each in its own tree $M/cpu/<arm>/tree, one ccache
#   fixed    test_top_k_rows_fixed_graph alone with LLAMA_UBATCH_TRACE=1 (the allocator's trace, read once a process),
#            on each arm: test and view must fail it, tip must pass it
#   rig      the rig tarball of the branch (scripts/build_llamacpp_docker.sh, TESTS=1) in $M/main/rig
#   sampler  every test of test-backend-sampler in a process of its own on MODEL: tip with the switch unset, tip with it
#            on for the whole process, and the test arm unset (BASE_COMMIT's code)
#   graph    test_multi_output_cpu_suffix's graphs as the scheduler prints them (GGML_SCHED_DEBUG=2): the test arm and
#            tip with the switch off, tip with it on; the TOP_K, PAD, computed PAD and GET_ROWS lines counted, and the
#            test arm's lines against tip's with the switch off
#   off      test-llama-archs -b CPU -s 1: BASE_CPU (BASE_COMMIT's static CPU build) writes every architecture's logits,
#            tip compares with no switch and with LLAMA_SAMPLING_TOP_K_ROWS=1
#   replay   C691's recorded set top-k-rows (SET) replayed on the CPU with the rig build's test-backend-ops: its runs tk
#            (TOP_K at 248,320 columns) and tk-rows (TOP_K_ROWS)
#   c565     2026-11-05_c565_ladder4fix/run/cpu_tests.sh on tip, static and then shared libraries
# Env: M (the work root), REPO (this repository as run), LOCK, MODEL (a small GGUF with a vocabulary), SET (the
#   top-k-rows set's folder), BASE_CPU (bin/ of BASE_COMMIT's static CPU build, with test-llama-archs), SEED_SRC (a
#   llama.cpp clone with the fork as remote "fork", https); optional SEED_CCACHE (the rig build's), SEED_CPU_CCACHE,
#   SEED_C565_CCACHE; BRANCH (default pascal-p106-v4-b12246-c802), BASE_COMMIT (default d097d6d88 in full), TEST_COMMIT
#   (default the branch's first commit after BASE_COMMIT), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/logs/c802.log, each step's folder; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
MODEL=$(realpath "${MODEL:?MODEL=<a small GGUF>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c802}
BASE_COMMIT=${BASE_COMMIT:-d097d6d88a3ee3fb2e3dbdb90819d320557d9809}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-cpu fixed rig sampler graph off replay c565}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
RUN=$REPO/docs/data/2026-11-12_c802_topk_fixed_graph/run
SRC=$M/main/rig/src
SW=LLAMA_SAMPLING_TOP_K_ROWS
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c802.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# shellcheck disable=SC2086
box()    { locked docker run --rm $DA --network none -v "$M:$M" -v "$MODEL:/m.gguf:ro" "$@"; }
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
STUB='s=/usr/local/cuda/lib64/stubs/libcuda.so; if [ -e $s ]; then mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1;
      export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi'

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
TEST_COMMIT=${TEST_COMMIT:-$(git -C "$SRC" rev-list --reverse "$BASE_COMMIT..$TIP" | head -n 1)}
log "C802 checks: $BRANCH at ${TIP:0:9} (test ${TEST_COMMIT:0:9}), base ${BASE_COMMIT:0:9}, steps: $STEPS"
add branch commits OK "$(git -C "$SRC" log --format='%h %s' "$BASE_COMMIT..$TIP" | tr '\n' ';'); $(git -C "$SRC" diff --shortstat "$BASE_COMMIT" "$TIP")"

if step cpu; then
    [ -d "$M/cpu/ccache" ] || { mkdir -p "$M/cpu"; if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -a "$SEED_CPU_CCACHE" "$M/cpu/ccache"; else mkdir -p "$M/cpu/ccache"; fi; }
    for arm in "test:$TEST_COMMIT::test-backend-sampler" "view:$TIP:view.patch:test-backend-sampler" \
               "tip:$TIP::test-backend-sampler test-backend-ops test-llama-archs"; do
        IFS=: read -r tag commit patch targets <<< "$arm"
        A=$M/cpu/$tag
        rm -rf "$A" && mkdir -p "$A/tree/src" "$A/bin"
        git -C "$SRC" archive "$commit" | tar -x -C "$A/tree/src"
        echo "commit $commit $(git -C "$SRC" log -1 --format=%s "$commit")" > "$A/tree.txt"
        if [ -n "$patch" ]; then
            # outside any repository, git apply patches the folder like patch -p1
            (cd "$A/tree/src" && git apply "$RUN/$patch") || { add cpu "$tag patch" FAIL "$patch does not apply"; continue; }
            echo "patch $patch md5 $(md5sum < "$RUN/$patch" | cut -d' ' -f1)" >> "$A/tree.txt"
        fi
        box -e CCACHE_DIR="$M/cpu/ccache" -e CCACHE_BASEDIR="$A/tree" "$CPU_IMAGE" bash -c '
            cmake -S '"$A/tree/src"' -B '"$A/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$A/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$A/tree/build"' -j '"$JOBS"' --target '"$targets"' > '"$A/build.log"' 2>&1'
        rc=$?
        ok=1
        for t in $targets; do cp "$A/tree/build/bin/$t" "$A/bin/" 2> /dev/null || ok=0; done
        add cpu "$tag build" "$([ $rc = 0 ] && [ $ok = 1 ] && echo OK || echo FAIL)" \
            "rc=$rc; $(tr '\n' ';' < "$A/tree.txt") static; $targets; $(grep -ac 'warning:' "$A/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$A/build.log") error lines"
        rm -rf "$A/tree/build"
    done
fi
bin() { echo "$M/cpu/$1/bin"; }

if step fixed; then
    F=$M/fixed
    mkdir -p "$F"
    for tag in test view tip; do
        b=$(bin "$tag")
        box "$CPU_IMAGE" bash -c 'env LLAMA_UBATCH_TRACE=1 nice -n 19 stdbuf -oL -eL '"$b"'/test-backend-sampler --device cpu \
            --test test_top_k_rows_fixed_graph /m.gguf > '"$F/$tag.log"' 2>&1; echo "rc=$?"' > "$F/$tag.rc" 2>&1
        rc=$(sed -n 's/^rc=//p' "$F/$tag.rc" | tail -n 1)
        grep -a -E '^switch |^the same |fixed graph test PASSED|GGML_ASSERT|galloc-trigger' "$F/$tag.log" > "$F/$tag.lines"
        sum=$(grep -a -E 'node count|^the same |PASSED|GGML_ASSERT' "$F/$tag.lines" | tr '\n' ';')
        nt=$(grep -ac 'galloc-trigger' "$F/$tag.log")
        # the compute buffer the reserve planned and the one each context ended with (the switch off, then on)
        rs=$(grep -a 'sched_reserve: .*compute buffer size =' "$F/$tag.log" | sed -E 's/.*= *([0-9.]+) MiB.*/\1/' | tr '\n' ' ')
        en=$(grep -a '~llama_context: .*compute buffer size is' "$F/$tag.log" | grep -v ' 0\.0000 MiB' | sed -E 's/.*size is *([0-9.]+) MiB.*/\1/' | tr '\n' ' ')
        sum="compute buffer (off, on) MiB: reserved ${rs:-?}, at the end ${en:-?}; $sum"
        case $tag in
            # before the fix: the switch on has more than one node count
            test) v=$([ "$rc" != 0 ] && grep -aq '^switch on: MORE THAN ONE node count' "$F/$tag.lines" && grep -aq '^switch off: one node count (.*re-plans: 0$' "$F/$tag.lines" && echo "FAILS AS EXPECTED" || echo UNEXPECTED) ;;
            # the view: one node count, but re-plans
            view) v=$([ "$rc" != 0 ] && grep -aqE '^switch on: one node count \(.*re-plans: [1-9]' "$F/$tag.lines" && grep -aq '^switch off: one node count (.*re-plans: 0$' "$F/$tag.lines" && echo "FAILS AS EXPECTED" || echo UNEXPECTED) ;;
            tip)  v=$([ "$rc" = 0 ] && grep -aq '^switch on: one node count (.*re-plans: 0$' "$F/$tag.lines" && grep -aq 'fixed graph test PASSED' "$F/$tag.lines" && echo PASS || echo FAIL) ;;
        esac
        add fixed "$tag" "$v" "rc=$rc; $nt allocator trigger lines; $sum"
    done
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
T=$(tree "$BRANCH")

if step sampler; then
    S=$M/sampler
    mkdir -p "$S"
    # each test in a process of its own (the build's own list), so that a failing test does not end the others
    box "$CPU_IMAGE" bash -c '
        run() { # <name> <bin dir> [env...]
            local n=$1 b=$2 t; shift 2
            for t in $($b/test-backend-sampler --test list /m.gguf 2>&1 | sed -n "s/^  \(test_[a-z0-9_]*\)$/\1/p"); do
                env "$@" nice -n 19 stdbuf -oL -eL $b/test-backend-sampler --device cpu --test $t /m.gguf > '"$S"'/$n-$t.log 2>&1
                printf "%s\t%s\t%s\n" "$n" "$t" "$?"
            done
        }
        run off '"$(bin tip)"'
        run on '"$(bin tip)"' '"$SW"'=1
        run base '"$(bin test)"'' > "$S/rc.out" 2>&1
    grep -a -P '^(off|on|base)\t' "$S/rc.out" > "$S/rc.tsv"
    # test_logit_bias expects a +10 bias to win, which it does not on random weights (C691); the base fails C802's test
    for n in off on base; do
        x=test_logit_bias; [ $n = base ] && x="$x test_top_k_rows_fixed_graph"
        add sampler "$n" "$(awk -F'\t' -v n=$n -v x=" $x " '$1 == n && $3 != 0 && index(x, " " $2 " ") == 0' "$S/rc.tsv" | grep -q . && echo FAIL || echo OK)" \
            "$(awk -F'\t' -v n=$n '$1 == n' "$S/rc.tsv" | wc -l) tests, $(awk -F'\t' -v n=$n '$1 == n && $3 == 0' "$S/rc.tsv" | wc -l) pass; failing: $(awk -F'\t' -v n=$n '$1 == n && $3 != 0 { printf "%s (rc %s) ", $2, $3 }' "$S/rc.tsv")"
    done
    for n in off on; do
        add sampler "$n test_top_k_rows_switch" "$(grep -aq 'top-k rows switch test PASSED' "$S/$n-test_top_k_rows_switch.log" && echo PASS || echo FAIL)" \
            "$(grep -a -E '^chains ' "$S/$n-test_top_k_rows_switch.log" | tr '\n' ';')"
        add sampler "$n test_top_k_rows_fixed_graph" "$(grep -aq 'fixed graph test PASSED' "$S/$n-test_top_k_rows_fixed_graph.log" && echo PASS || echo FAIL)" \
            "$(grep -a -E 'node count|^the same ' "$S/$n-test_top_k_rows_fixed_graph.log" | tr '\n' ';')"
    done
fi

if step graph; then
    # the graphs of test_multi_output_cpu_suffix (a top-k first, two outputs a sequence in its second context) as the
    # scheduler prints them (view nodes are not printed); the node lines without their addresses, base against off
    G=$M/graph
    mkdir -p "$G"
    for n in base off on; do
        b=$(bin tip); e=$SW=0
        [ $n = base ] && b=$(bin test)
        [ $n = on ] && e=$SW=1
        box "$CPU_IMAGE" bash -c 'env '"$e"' GGML_SCHED_DEBUG=2 nice -n 19 '"$b"'/test-backend-sampler --device cpu \
            --test test_multi_output_cpu_suffix /m.gguf > '"$G/$n.log"' 2>&1; echo "rc=$?"' > "$G/$n.rc" 2>&1
        grep -a 'node #' "$G/$n.log" | sed -E 's/0x[0-9a-f]+//g' > "$G/$n.nodes"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$(wc -l < "$G/$n.nodes")" "$(grep -a -c 'TOP_K)' "$G/$n.nodes")" \
            "$(grep -a -c '(       PAD)' "$G/$n.nodes")" "$(grep -a '(       PAD)' "$G/$n.nodes" | grep -a -c ',c=1:')" \
            "$(grep -a -c 'GET_ROWS)' "$G/$n.nodes")" "$(sed -n 's/^rc=//p' "$G/$n.rc" | tail -n 1)" >> "$G/counts.tsv"
        add graph "$n" "$([ "$(sed -n 's/^rc=//p' "$G/$n.rc" | tail -n 1)" = 0 ] && echo OK || echo FAIL)" \
            "$e; node lines $(wc -l < "$G/$n.nodes"), TOP_K $(grep -a -c 'TOP_K)' "$G/$n.nodes"), PAD $(grep -a -c '(       PAD)' "$G/$n.nodes") (computed $(grep -a '(       PAD)' "$G/$n.nodes" | grep -a -c ',c=1:')), GET_ROWS $(grep -a -c 'GET_ROWS)' "$G/$n.nodes")"
        gzip -f -9 "$G/$n.log"
    done
    if cmp -s "$G/base.nodes" "$G/off.nodes"; then add graph "base against off" SAME "the same $(wc -l < "$G/off.nodes") node lines"; else
        add graph "base against off" DIFFERS "$(diff "$G/base.nodes" "$G/off.nodes" | grep -c '^[<>]') lines"; fi
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
    for arm in "tip:" "tkr:$SW=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        box "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- '"$envs"' nice -n 19 '"$(bin tip)"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
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

if step replay; then
    # C691's two runs of the set: tk (the 39 TOP_K cases at 248,320 columns) and tk-rows (the 75 TOP_K_ROWS cases)
    SET=$(realpath "${SET:?SET=<the top-k-rows set folder>}")
    R=$M/replay
    mkdir -p "$R"
    for f in 'tk|TOP_K|ne=\[248320,' 'tk-rows|TOP_K_ROWS|.'; do
        IFS='|' read -r run op pat <<< "$f"
        # shellcheck disable=SC2086
        locked docker run --rm $DA --network none -v "$T:$T:ro" -v "$SET:$SET:ro" -v "$R:/k" "$IMAGE" sh -c "$STUB"'
            nice -n 19 '"$T"'/bin/test-backend-ops -b CPU -o '"$op"' -p "'"$pat"'" --replay '"$SET"' --replay-log /k/'"$run"'.cases.log \
                > /k/'"$run"'.log 2>&1; echo "rc=$?"' > "$R/$run.rc" 2>&1
        rc=$(sed -n 's/^rc=//p' "$R/$run.rc" | tail -n 1)
        add replay "$run" "$([ "$rc" = 0 ] && ! grep -aq 'FAIL' "$R/$run.log" && echo OK || echo FAIL)" \
            "rc=$rc; -o $op -p '$pat'; $(grep -a -c 'FAIL' "$R/$run.log") FAIL lines; $(grep -a -m 1 'tests passed' "$R/$run.log" | tr -s ' '); $(grep -a 'replay:' "$R/$run.log" | tail -n 1)"
    done
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        W=$M/c565-$v
        mkdir -p "$W"
        [ -d "$W/ccache" ] || [ -z "${SEED_C565_CCACHE:-}" ] || cp -a "$SEED_C565_CCACHE" "$W/ccache"
        env SRC="$SRC" WORK="$W" TREES="c802=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
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
