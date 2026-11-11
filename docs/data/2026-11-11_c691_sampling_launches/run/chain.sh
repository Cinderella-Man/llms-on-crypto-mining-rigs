#!/usr/bin/env bash
# C691 (Q1223) on a build machine, CPU only: the rig tarball of the branch with its tests, the CPU checks of the backend
# samplers' one top-k over the output rows (LLAMA_SAMPLING_TOP_K_ROWS=1), and the recorded CUDA set a P106 replays.
# Every step that builds or tests takes LOCK, so that one runs at a time on a shared machine.
# Steps, in order:
#   build    the rig tarball of fork/BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK/w
#   cpu      static native CPU builds of test-backend-sampler and test-backend-ops: the branch (WORK/w/cpu/bin) and
#            BASE (WORK/w/cpu-base/bin), from the same clone and image
#   tbo      test-backend-ops -b CPU -o TOP_K,TOP_K_ROWS with the rig build's CPU code (the CPU against itself)
#   sampler  test-backend-sampler on MODEL, CPU only: every test with the switch unset, every test with it on for the
#            whole process, and BASE's every test; test_top_k_rows_switch compares off with on itself
#   graph    one test's graphs as the scheduler prints them (GGML_SCHED_DEBUG=2): BASE, the branch with the switch off
#            and with it on; the TOP_K, PAD and GET_ROWS nodes counted, and BASE's node lines against the branch's
#            with the switch off
#   record   scripts/record_checks.sh SUITE=top-k-rows into WORK/sets, then its tar
#   replay   the record replayed on the CPU (-b CPU --replay): the set reads back and passes before a rig sees it
# Env: WORK (a folder of its own), PUB (a checkout of this repository), MODEL (a small GGUF for test-backend-sampler),
#   BRANCH (default pascal-p106-v4-b12246-c691), BASE (default fork/pascal-p106-v4), STEPS (default all, in the order
#   above), LOCK (a flock file, optional), JOBS (default 8), DOCKER_ARGS (e.g. labels),
#   IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04)
# Output: WORK/chain.log (a line per step), WORK/build.log, WORK/tbo/, WORK/sampler/, WORK/graph/, WORK/sets/,
#   WORK/replay/, WORK/summary.txt; WORK/DONE when it ends
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
PUB=$(realpath "${PUB:?PUB=<this repository>}")
MODEL=$(realpath "${MODEL:?MODEL=<a small GGUF>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c691}
BASE=${BASE:-fork/pascal-p106-v4}
STEPS=${STEPS:-build cpu tbo sampler graph record replay}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$WORK/chain.log"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
# the recorded cases, as scripts/record_checks.sh's suite top-k-rows selects them
TKV='ne=\[248320,'
tarball() { ls -td "$WORK"/w/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
STUB='s=/usr/local/cuda/lib64/stubs/libcuda.so; if [ -e $s ]; then mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1;
      export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi'

if step build; then
    log "build fork/$BRANCH"
    # shellcheck disable=SC2086
    locked env TESTS=1 JOBS="$JOBS" WORK="$WORK/w" DOCKER_ARGS="$D" nice -n 19 bash "$PUB/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/build.log" 2>&1
    rc=$?
    T=$(tarball)
    log "build rc=$rc: $(basename "${T:-none}") md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/DONE"; exit 1; }
fi
T=$(tarball)
BIN=${T:+$T/bin}

if step cpu; then
    log "cpu builds of $(git -C "$WORK/w/src" rev-parse --short=9 HEAD) and $BASE"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK/w:/work" -e BASE="$BASE" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory "*"
        rm -rf /work/base && git -C /work/src worktree prune && git -C /work/src worktree add -q --detach /work/base "$BASE"
        rc=0
        for b in cpu:/work/src cpu-base:/work/base; do
            d=/work/${b%%:*}; s=${b#*:}
            mkdir -p $d
            cmake -S $s -B $d/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
                -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $d/cmake.log 2>&1
            r=0
            nice -n 19 cmake --build $d/build -j "$JOBS" --target test-backend-sampler test-backend-ops > $d/build.log 2>&1 || r=$?
            [ $r = 0 ] && mkdir -p $d/bin && cp $d/build/bin/test-backend-sampler $d/build/bin/test-backend-ops $d/bin/
            echo "${b%%:*} rc=$r commit $(git -C $s rev-parse --short=9 HEAD)"
            [ $r = 0 ] || rc=$r
        done
        chown -R "$HOST_UID:$HOST_GID" /work
        exit $rc' > "$WORK/cpu.out" 2>&1
    log "cpu rc=$?: $(grep -a ' rc=' "$WORK/cpu.out" | tr '\n' ' ')"
fi
CPU=$WORK/w/cpu/bin
CPU_BASE=$WORK/w/cpu-base/bin

# the image prints a banner first: a step's own status is its output's last line
if step tbo; then
    mkdir -p "$WORK/tbo"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$BIN:$BIN:ro" -v "$WORK/tbo:/k" "$IMAGE" sh -c "$STUB"'
        nice -n 19 '"$BIN"'/test-backend-ops -b CPU -o TOP_K,TOP_K_ROWS > /k/tbo.log 2>&1; echo "rc=$?"' \
        > "$WORK/tbo/tbo.rc" 2>&1
    log "tbo: $(tail -n 1 "$WORK/tbo/tbo.rc"); $(grep -a -c 'FAIL' "$WORK/tbo/tbo.log") FAIL lines; $(grep -a -m 1 'tests passed' "$WORK/tbo/tbo.log")"
fi

if step sampler; then
    S=$WORK/sampler
    mkdir -p "$S"
    # shellcheck disable=SC2086
    # each test in a process of its own (the build's own list), so that a failing test does not end the others
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK/w:$WORK/w:ro" -v "$MODEL:/m.gguf:ro" -v "$S:/k" "$IMAGE" bash -c '
        run() { # <name> <bin dir> [env...]
            local n=$1 b=$2 t; shift 2
            for t in $($b/test-backend-sampler --test list /m.gguf 2>&1 | sed -n "s/^  \(test_[a-z0-9_]*\)$/\1/p"); do
                env "$@" nice -n 19 $b/test-backend-sampler --device cpu --test $t /m.gguf > /k/$n-$t.log 2>&1
                printf "%s\t%s\t%s\n" "$n" "$t" "$?"
            done
        }
        run off '"$CPU"'
        run on '"$CPU"' LLAMA_SAMPLING_TOP_K_ROWS=1
        run base '"$CPU_BASE"'
        chown -R '"$(id -u):$(id -g)"' /k' > "$S/rc.out" 2>&1
    grep -a -P '^(off|on|base)\t' "$S/rc.out" > "$S/rc.tsv"
    for n in off on base; do
        log "sampler $n: $(awk -F'\t' -v n=$n '$1 == n' "$S/rc.tsv" | wc -l) tests, $(awk -F'\t' -v n=$n '$1 == n && $3 == 0' "$S/rc.tsv" | wc -l) pass; failing: $(awk -F'\t' -v n=$n '$1 == n && $3 != 0 { printf "%s (rc %s) ", $2, $3 }' "$S/rc.tsv")"
    done
    log "sampler switch test (off run): $(grep -a -E '^chains |top-k rows switch' "$S/off-test_top_k_rows_switch.log" | tr '\n' ' ')"
fi

if step graph; then
    # the graphs of test_multi_output_cpu_suffix (a top-k first, two outputs a sequence in its second context) as the
    # scheduler prints them; the node lines without their addresses, base against off
    G=$WORK/graph
    mkdir -p "$G"
    for n in base off on; do
        b=$CPU; e=LLAMA_SAMPLING_TOP_K_ROWS=0
        [ $n = base ] && b=$CPU_BASE
        [ $n = on ] && e=LLAMA_SAMPLING_TOP_K_ROWS=1
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$b:$b:ro" -v "$MODEL:/m.gguf:ro" -v "$G:/k" "$IMAGE" sh -c '
            env '"$e"' GGML_SCHED_DEBUG=2 nice -n 19 '"$b"'/test-backend-sampler --device cpu --test test_multi_output_cpu_suffix /m.gguf \
                > /k/'"$n"'.log 2>&1
            echo "rc=$?"' > "$G/$n.rc" 2>&1
        grep -a 'node #' "$G/$n.log" | sed -E 's/0x[0-9a-f]+//g' > "$G/$n.nodes"
        log "graph $n: $(tail -n 1 "$G/$n.rc"); node lines $(wc -l < "$G/$n.nodes"), TOP_K $(grep -a -c 'TOP_K)' "$G/$n.nodes"), PAD $(grep -a -c '(       PAD)' "$G/$n.nodes"), GET_ROWS $(grep -a -c 'GET_ROWS)' "$G/$n.nodes")"
        gzip -f -9 "$G/$n.log"
    done
    if cmp -s "$G/base.nodes" "$G/off.nodes"; then log "graph: base and off the same node lines"; else
        log "graph: base and off differ: $(diff "$G/base.nodes" "$G/off.nodes" | grep -c '^[<>]') lines"; fi
fi

if step record; then
    log "record top-k-rows"
    # shellcheck disable=SC2086
    locked env STEPS="record tar" OUT="$WORK/sets" SUITE=top-k-rows BIN="$BIN" JOBS=4 DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$IMAGE" \
        nice -n 19 bash "$PUB/scripts/record_checks.sh" > "$WORK/record.log" 2>&1
    log "record rc=$?: $(tail -n 4 "$WORK/sets/top-k-rows.log" 2> /dev/null | tr '\n' ' ')"
fi

if step replay; then
    S=$WORK/sets/top-k-rows
    mkdir -p "$WORK/replay"
    for f in "TOP_K|$TKV" "TOP_K_ROWS|."; do
        o=${f%%|*}
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$BIN:$BIN:ro" -v "$S:$S:ro" -v "$WORK/replay:/k" "$IMAGE" sh -c "$STUB"'
            nice -n 19 '"$BIN"'/test-backend-ops -b CPU -o '"$o"' -p "'"${f#*|}"'" --replay '"$S"' --replay-log /k/'"$o"'.cases.log \
                > /k/'"$o"'.log 2>&1; echo "rc=$?"' > "$WORK/replay/$o.rc" 2>&1
        log "replay $o on the CPU: $(tail -n 1 "$WORK/replay/$o.rc"); $(grep -a -c 'FAIL' "$WORK/replay/$o.log") FAIL lines; $(grep -a -m 1 'tests passed' "$WORK/replay/$o.log"); $(grep -a 'replay:' "$WORK/replay/$o.log" | tail -n 1)"
    done
fi

grep -a -v '^$' "$WORK/chain.log" > "$WORK/summary.txt"
touch "$WORK/DONE"
