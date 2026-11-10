#!/usr/bin/env bash
# C704 (Q1244) on a build machine, CPU only: the rig tarball of the branch with its tests, the CPU checks of the
# k-pool indexer reading the pooled keys in place (LLAMA_QSA_INDEXER_ROWS=1), and the recorded CUDA set a P106
# replays. Every step that builds or tests takes LOCK, so that one runs at a time on a shared machine.
# Steps, in order:
#   build   the rig tarball of fork/BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK/w
#   cpu     a static native CPU build of test-llama-archs and test-backend-ops from the same checkout (WORK/cpu/bin),
#           the flags of BASE_CPU's build
#   tbo     test-backend-ops -b CPU -o LIGHTNING_INDEXER with the rig build's CPU code: the CPU against itself, so
#           the "same" cases check that the in-place read gives the gathered copy's scores bit for bit
#   record  scripts/record_checks.sh SUITE=indexer-rows into WORK/sets, then its tar
#   replay  the record replayed on the CPU (-b CPU --replay): the set reads back and passes before a rig sees it
#   archs   test-llama-archs logits (LLAMA_ARCHS_LOGITS): BASE_CPU writes every model's, this build compares with the
#           switch off and on; then, with LLAMA_ARCHS_INDEXER_TOP_K=8 so that the selection follows the scores,
#           this build with the switch off writes qwen4exp's and glm5-next's and with it on compares; and, as the
#           control, top-k 8 against the every-cell logits, which must differ
#   graph   qwen4exp's graph with the switch off and on as the scheduler prints it: the gather's nodes and the
#           indexer's, counted
# Env: WORK (a folder of its own), PUB (a checkout of this repository), BASE_CPU (bin/ of a static native CPU build
#   of the base branch with test-llama-archs), BRANCH (default pascal-p106-v4-b12246-c704), STEPS (default all, in
#   the order above), LOCK (a flock file, optional), JOBS (default 8), DOCKER_ARGS (e.g. labels),
#   IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04)
# Output: WORK/chain.log (a line per step), WORK/build.log, WORK/tbo/, WORK/sets/, WORK/replay/, WORK/archs/,
#   WORK/summary.txt; WORK/DONE when it ends
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
PUB=$(realpath "${PUB:?PUB=<this repository>}")
: "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c704}
STEPS=${STEPS:-build cpu tbo record replay archs graph}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$WORK/chain.log"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
# the recorded cases, as scripts/record_checks.sh's suite indexer-rows selects them
LID='n_cells=|nh=4,|nb=(1|32),'
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
    log "cpu build of $(git -C "$WORK/w/src" rev-parse --short=9 HEAD)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK/w:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        mkdir -p /work/cpu
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/cpu/cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target test-llama-archs test-backend-ops > /work/cpu/build.log 2>&1 || rc=$?
        [ $rc = 0 ] && mkdir -p /work/cpu/bin && cp /work/cpu/build/bin/test-llama-archs /work/cpu/build/bin/test-backend-ops /work/cpu/bin/
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/ccache
        exit $rc' > "$WORK/cpu.out" 2>&1
    log "cpu rc=$?"
fi
CPU=$WORK/w/cpu/bin

# the image prints a banner first: a step's own status is its output's last line
if step tbo; then
    mkdir -p "$WORK/tbo"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$BIN:$BIN:ro" -v "$WORK/tbo:/k" "$IMAGE" sh -c "$STUB"'
        nice -n 19 '"$BIN"'/test-backend-ops -b CPU -o LIGHTNING_INDEXER > /k/lid.log 2>&1; echo "rc=$?"' \
        > "$WORK/tbo/lid.rc" 2>&1
    log "tbo lid: $(tail -n 1 "$WORK/tbo/lid.rc"); $(grep -a -c 'FAIL' "$WORK/tbo/lid.log") FAIL lines; $(grep -a -m 1 'tests passed' "$WORK/tbo/lid.log")"
fi

if step record; then
    log "record indexer-rows"
    # shellcheck disable=SC2086
    locked env STEPS="record tar" OUT="$WORK/sets" SUITE=indexer-rows BIN="$BIN" JOBS=4 DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$IMAGE" \
        nice -n 19 bash "$PUB/scripts/record_checks.sh" > "$WORK/record.log" 2>&1
    log "record rc=$?: $(tail -n 3 "$WORK/sets/indexer-rows.log" 2> /dev/null | tr '\n' ' ')"
fi

if step replay; then
    S=$WORK/sets/indexer-rows
    mkdir -p "$WORK/replay"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$BIN:$BIN:ro" -v "$S:$S:ro" -v "$WORK/replay:/k" "$IMAGE" sh -c "$STUB"'
        nice -n 19 '"$BIN"'/test-backend-ops -b CPU -o LIGHTNING_INDEXER -p "'"$LID"'" --replay '"$S"' --replay-log /k/cases.log \
            > /k/cpu.log 2>&1; echo "rc=$?"' > "$WORK/replay/rc.txt" 2>&1
    log "replay on the CPU: $(tail -n 1 "$WORK/replay/rc.txt"); $(grep -a -c 'FAIL' "$WORK/replay/cpu.log") FAIL lines; $(grep -a -m 1 'tests passed' "$WORK/replay/cpu.log"); $(grep -a 'replay:' "$WORK/replay/cpu.log" | tail -n 1)"
fi

if step archs; then
    K=$WORK/archs
    mkdir -p "$K/a" "$K/b"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" "$IMAGE" bash -c '
        run() { # <name> <bin dir> <logits prefix> [env...] [-- test-llama-archs args]
            local n=$1 b=$2 p=$3; shift 3
            local e=() a=()
            while [ $# -gt 0 ] && [ "$1" != -- ]; do e+=("$1"); shift; done
            [ "${1:-}" = -- ] && shift && a=("$@")
            env LLAMA_ARCHS_LOGITS=$p "${e[@]}" nice -n 19 $b/test-llama-archs -b CPU -s 1 "${a[@]}" 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/$n.log
            echo "$n rc=${PIPESTATUS[0]}"
        }
        run base '"$BASE_CPU"' '"$K"'/a/logits-
        ls '"$K"'/a/logits-*.f32 | wc -l > '"$K"'/n_base.txt
        run off '"$CPU"' '"$K"'/a/logits-
        run on  '"$CPU"' '"$K"'/a/logits- LLAMA_QSA_INDEXER_ROWS=1
        run top8-off '"$CPU"' '"$K"'/b/logits- LLAMA_ARCHS_INDEXER_TOP_K=8 -- -a "qwen4exp|glm5-next"
        ls '"$K"'/b/logits-*.f32 | wc -l > '"$K"'/n_top8.txt
        run top8-on  '"$CPU"' '"$K"'/b/logits- LLAMA_ARCHS_INDEXER_TOP_K=8 LLAMA_QSA_INDEXER_ROWS=1 -- -a "qwen4exp|glm5-next"
        # the control: at top-k 8 the logits must differ from the every-cell ones, or the selection would not matter
        run top8-vs-all '"$CPU"' '"$K"'/a/logits- LLAMA_ARCHS_INDEXER_TOP_K=8 -- -a "qwen4exp|glm5-next"
        rm -f '"$K"'/a/*.f32 '"$K"'/b/*.f32
        chown -R '"$(id -u):$(id -g)"' '"$K"'' > "$K/rc.txt" 2>&1
    for n in off on top8-on top8-vs-all; do
        grep -a 'logits against' "$K/$n.log" | grep -v ', 0 of ' > "$K/$n.differ.txt"
        log "archs $n: $(grep -a -c 'logits against' "$K/$n.log") compared, $(wc -l < "$K/$n.differ.txt") differ: $(cut -d: -f1 "$K/$n.differ.txt" | tr '\n' ' '); $(grep -a -m 1 '^Summary' "$K/$n.log")"
    done
    log "archs: base wrote $(cat "$K/n_base.txt"), top8-off wrote $(cat "$K/n_top8.txt"); $(grep ' rc=' "$K/rc.txt" | tr '\n' ' ')"
fi

if step graph; then
    # which path each setting builds: qwen4exp's graph as the scheduler prints it (GGML_SCHED_DEBUG=2, every graph the
    # test computes), its GET_ROWS nodes, its indexer nodes, and the indexer nodes whose sources hold the gathered copy
    # (indexer_k, the gather's reshape)
    G=$WORK/graph
    mkdir -p "$G"
    for n in off on; do
        e="LLAMA_QSA_INDEXER_ROWS=0"; [ $n = on ] && e="LLAMA_QSA_INDEXER_ROWS=1"
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$CPU:$CPU:ro" -v "$G:/k" "$IMAGE" sh -c '
            env '"$e"' GGML_SCHED_DEBUG=2 nice -n 19 '"$CPU"'/test-llama-archs -b CPU -s 1 -a qwen4exp -v 99 > /k/'"$n"'.log 2>&1
            echo "rc=$?"' > "$G/$n.rc" 2>&1
        log "graph $n: $(tail -n 1 "$G/$n.rc"); node lines $(grep -a -c 'node #' "$G/$n.log"), GET_ROWS $(grep -a 'node #' "$G/$n.log" | grep -a -c '(  GET_ROWS)'), LIGHTNING_INDEXER $(grep -a 'node #' "$G/$n.log" | grep -a -c '(LIGHTNING_)'), of them reading the gathered copy $(grep -a 'node #' "$G/$n.log" | grep -a '(LIGHTNING_)' | grep -a -c ' indexer_k-')"
        gzip -f -9 "$G/$n.log"
    done
fi

grep -a -v '^$' "$WORK/chain.log" > "$WORK/summary.txt"
touch "$WORK/DONE"
