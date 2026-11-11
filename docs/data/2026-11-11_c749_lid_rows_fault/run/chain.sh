#!/usr/bin/env bash
# C749 (Q1324) on a build machine, CPU only, no GPU: GGML_CUDA_LID_ROWS_FAULT=1 (checks only) scores rows i mod n_cells
# of a list, so the fault run of the in-place indexer cases reaches every case. The steps, in order:
#   build   the rig tarball of fork/BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK/w
#   code    the branch's GPU code against BASE's, kernel by kernel: cuobjdump -sass and -ptx of llama-server and
#           test-backend-ops, compared by C103's funcs.py (every kernel of the base the same; the new ones listed)
#   ctest   C565's cpu_tests.sh on the branch's commit, static as the rig builds it: tree, build, ctest
#   emu     a static CPU build of test-backend-ops from the same commit with run/cpu_fault.patch applied, the rig build's
#           CPU flags (SSE4.2, no AVX): the CPU then honours GGML_CUDA_LID_ROWS_FAULT=1 as the card does, rows i mod
#           n_cells of the list in the list op only (the gathered copy beside it reads the listed rows)
#   replay  the run lid-rows-fault of the set (-o LIGHTNING_INDEXER -p n_cells=, the 113 in-place cases) replayed on the
#           CPU three times: rig-fault, the rig build with the switch (the CPU backend ignores it: every case passes);
#           emu-off, the patched build without the switch (every case passes); emu-fault, the patched build with the
#           switch (every case must fail, and the run end with its summary)
#   tables  WORK/results.tsv (a line per replay) and WORK/cases.tsv (a line per case: its verdict in each replay, and
#           emu-fault's errors)
# Every step that builds or tests takes LOCK, so one runs at a time on a shared machine, at a low CPU share
# (--cpu-shares=256, nice 19).
# Env: WORK (a folder of its own), PUB (a checkout of this repository), BASE (the base build's rig tarball),
#   SET (the v4night set's folder: cases/, runs.tsv), BRANCH (default pascal-p106-v4-b12246-c749), STEPS (default all,
#   in the order above), LOCK (a flock file, optional), JOBS (default 8), DOCKER_ARGS (e.g. labels), IMAGE (default
#   nvidia/cuda:12.9.2-devel-ubuntu24.04), CPU_IMAGE (default pq-c565-cpu:24.04, which cpu_tests.sh makes),
#   SEED_SRC and SEED_CCACHE (optional: a llama.cpp clone with the fork as remote "fork", and the base build's ccache,
#   copied into WORK/w before the first build so that it fetches and compiles only what differs)
# Output: WORK/chain.log (a line per step), WORK/build.log, WORK/code/, WORK/ctest/, WORK/emu/, WORK/replay/,
#   WORK/results.tsv, WORK/cases.tsv; WORK/DONE when it ends
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
PUB=$(realpath "${PUB:?PUB=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c749}
STEPS=${STEPS:-build code ctest emu replay tables}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/w"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$WORK/chain.log"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
tarball() { ls -td "$WORK"/w/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
STUB='s=/usr/local/cuda/lib64/stubs/libcuda.so; if [ -e $s ]; then mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1;
      export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi'
# the run lid-rows-fault of the set's run list: its filter
OPS=LIGHTNING_INDEXER
PARAMS='n_cells='

if step build; then
    [ -d "$WORK/w/src/.git" ] || [ -z "${SEED_SRC:-}" ] || cp -a "$SEED_SRC" "$WORK/w/src"
    [ -d "$WORK/w/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -a "$SEED_CCACHE" "$WORK/w/ccache"
    log "build fork/$BRANCH"
    # shellcheck disable=SC2086
    locked env TESTS=1 JOBS="$JOBS" WORK="$WORK/w" DOCKER_ARGS="$D" nice -n 19 bash "$PUB/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/build.log" 2>&1
    rc=$?
    T=$(tarball)
    log "build rc=$rc: $(basename "${T:-none}").tar.gz $([ -n "$T" ] && stat -c %s "$T.tar.gz") B md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1)"
    [ $rc = 0 ] || { touch "$WORK/DONE"; exit 1; }
fi
T=$(tarball)
BIN=${T:+$T/bin}
SHA=$(git -C "$WORK/w/src" rev-parse "fork/$BRANCH")

if step code; then
    # every kernel of the base must be the same; the fault's list kernel is the one new
    : "${BASE:?BASE=<the base rig tarball>}"
    C=$WORK/code
    mkdir -p "$C/base"
    b=$(tar tzf "$BASE" | head -n 1 | cut -d/ -f1)
    [ -x "$C/base/$b/bin/test-backend-ops" ] || tar xzf "$BASE" -C "$C/base" "$b/bin/llama-server" "$b/bin/test-backend-ops"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:$WORK" "$IMAGE" bash -c '
        for x in llama-server test-backend-ops; do
            for w in sass ptx; do
                nice -n 19 cuobjdump -$w "$1/base/$2/bin/$x" > "$1/base.$x.$w"
                nice -n 19 cuobjdump -$w "$3/$x" > "$1/branch.$x.$w"
            done
        done
        chown -R "$4" "$1"' sh "$C" "$b" "$BIN" "$(id -u):$(id -g)" > "$C/code.log" 2>&1
    for x in llama-server test-backend-ops; do
        for w in sass ptx; do
            echo "== $x $w" >> "$C/code.log"
            nice -n 19 python3 "$PUB/docs/data/2026-11-11_c103_union_gather/run/funcs.py" "$w" "$C/base.$x.$w" "$C/branch.$x.$w" \
                >> "$C/code.log" 2>&1
        done
    done
    log "code: $(grep -c '^verdict: SAME' "$C/code.log") of 4 listings with every base kernel the same, $(grep -c '^verdict: DIFFERS' "$C/code.log") with one changed; new: $(grep '^  new: ' "$C/code.log" | sort -u | sed 's/^  new: //' | tr '\n' ' ')"
fi

if step ctest; then
    log "ctest of $SHA, static"
    # the whole script under LOCK (its own LOCK covers only its build, and a second flock on the file would wait on this one)
    # shellcheck disable=SC2086
    locked env WORK="$WORK/ctest" SRC="$WORK/w/src" TREES="c749=$SHA" STEPS='tree build ctest' SHARED=OFF JOBS="$JOBS" \
        LOCK= DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$CPU_IMAGE" nice -n 19 \
        bash "$PUB/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/ctest.out" 2>&1
    log "ctest rc=$?: $(grep -v '^tree	check' "$WORK/ctest/summary.tsv" | cut -f 2-4 | tr '\t\n' ' ;')"
fi

if step emu; then
    E=$WORK/emu
    rm -rf "$E/src" && mkdir -p "$E/src" "$E/ccache"
    git -C "$WORK/w/src" archive "$SHA" | tar -x -C "$E/src"
    # outside any repository, git apply patches the folder like patch -p1
    GIT_CEILING_DIRECTORIES=$E git -C "$E/src" apply "$HERE/cpu_fault.patch"
    rc=$?
    log "emu: $SHA + cpu_fault.patch apply rc=$rc"
    # shellcheck disable=SC2086
    [ $rc = 0 ] && locked docker run --rm $D -v "$E:$E" -e CCACHE_DIR="$E/ccache" "$CPU_IMAGE" bash -c '
        cmake -S '"$E/src"' -B '"$E/build"' -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF \
            -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$E/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$E/build"' -j '"$JOBS"' --target test-backend-ops > '"$E/build.log"' 2>&1
        rc=$?
        chown -R '"$(id -u):$(id -g)"' '"$E"'
        exit $rc'
    log "emu build rc=$?: $(ls "$E/build/bin/test-backend-ops" 2> /dev/null)"
fi
EMU=$WORK/emu/build/bin

if step replay; then
    : "${SET:?SET=<the v4night set folder>}"
    SET=$(realpath "$SET")
    R=$WORK/replay
    mkdir -p "$R"
    for spec in "rig-fault:$BIN:GGML_CUDA_LID_ROWS_FAULT=1" "emu-off:$EMU:" "emu-fault:$EMU:GGML_CUDA_LID_ROWS_FAULT=1"; do
        IFS=: read -r run b e <<< "$spec"
        rm -f "$R/$run.tsv"
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$b:$b:ro" -v "$SET:$SET:ro" -v "$R:/k" "$IMAGE" sh -c "$STUB"'
            env '"$e"' nice -n 19 '"$b"'/test-backend-ops -b CPU -o '"$OPS"' -p "'"$PARAMS"'" --replay '"$SET"' \
                --replay-log /k/'"$run"'.tsv > /k/'"$run"'.log 2>&1
            echo "exit $?" >> /k/'"$run"'.log
            chown '"$(id -u):$(id -g)"' /k/'"$run"'.log /k/'"$run"'.tsv' > "$R/$run.docker" 2>&1
        log "replay $run ($e): $(tail -n 1 "$R/$run.log"); cases $(wc -l < "$R/$run.tsv"), OK $(cut -f 3 "$R/$run.tsv" | grep -c '^OK$'), FAIL $(cut -f 3 "$R/$run.tsv" | grep -c '^FAIL$'); $(grep -a -m 1 'tests passed' "$R/$run.log" | sed 's/^ *//'); $(grep -a 'replay: .*cases replayed' "$R/$run.log" | sed 's/^ *replay: [^:]*: //')"
    done
fi

if step tables; then
    R=$WORK/replay
    {
        printf 'run\tbinary\tswitch\texpect\tcases\tok\tfail\texit\tsummary\n'
        for spec in "rig-fault:rig build:GGML_CUDA_LID_ROWS_FAULT=1:pass" "emu-off:CPU stand-in build:-:pass" \
                "emu-fault:CPU stand-in build:GGML_CUDA_LID_ROWS_FAULT=1:fail"; do
            IFS=: read -r run b e x <<< "$spec"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$b" "$e" "$x" "$(wc -l < "$R/$run.tsv")" \
                "$(cut -f 3 "$R/$run.tsv" | grep -c '^OK$')" "$(cut -f 3 "$R/$run.tsv" | grep -c '^FAIL$')" \
                "$(tail -n 1 "$R/$run.log" | sed 's/^exit //')" \
                "$(grep -a -m 1 'tests passed' "$R/$run.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')"
        done
    } > "$WORK/results.tsv"
    # a line per case, in emu-fault's order: the case, its verdict in each replay, emu-fault's errors
    printf 'id\tcase\trig-fault\temu-off\temu-fault\temu-fault_errors\n' > "$WORK/cases.tsv"
    awk -F'\t' 'FILENAME == ARGV[1] { v1[$1] = $3 } FILENAME == ARGV[2] { v2[$1] = $3 }
        FILENAME == ARGV[3] { printf "%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, ($1 in v1 ? v1[$1] : "-"), ($1 in v2 ? v2[$1] : "-"), $3, $4 }' \
        "$R/rig-fault.tsv" "$R/emu-off.tsv" "$R/emu-fault.tsv" >> "$WORK/cases.tsv"
    log "tables: $(tail -n +2 "$WORK/results.tsv" | cut -f 1,5-8 | tr '\t\n' ' ;')"
fi

touch "$WORK/DONE"
