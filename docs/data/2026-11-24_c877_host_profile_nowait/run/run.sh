#!/usr/bin/env bash
# C877 (Q1583, 2026-11-24): GGML_CUDA_PROFILE_HOST_NOWAIT=1, the host profile without waiting (fork branch
# pascal-p106-v5-host-profile-nowait, one commit on pascal-p106-v5 at 8a9629942), built and checked on a build machine
# with three newer NVIDIA cards, which run the rig build's compute 6.1 PTX through the driver's JIT. Correctness and
# counts only: nothing here is timed. Steps (STEPS, in this order):
#   src      a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the branch
#            fetched and checked against HEAD_SHA, its diff against the base listed; compiler caches hard-linked from
#            RIG_CCACHE and CPU_CCACHE
#   rig      the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and the base's two extra targets
#            (test-llama-archs, test-cuda-keepalive), the web page pinned (HF_UI_VERSION), under LOCK; its md5 and its
#            binaries' md5s; warnings against BASE_RIG_LOG
#   session  Flint Pascal Q8 GGUF (FLINT) with its v5 preset (--pipe-depth 4 kept), the split changed to the three cards
#            (GPUS, TS), one load per arm, each running C950's session ROUNDS times through run/session_marks.py:
#              off       the head, the profile unset
#              wait      the head, GGML_CUDA_PROFILE_HOST=<arm>/hp.csv (the profile's default: it waits)
#              nowait    the head, the same and GGML_CUDA_PROFILE_HOST_NOWAIT=1
#              basewait  the base's rig build (BASE_RIG), GGML_CUDA_PROFILE_HOST=<arm>/hp.csv (the profile as it was)
#   archs    test-llama-archs -b CPU -s 1 of the head's rig build against the logits the base's rig build (BASE_RIG)
#            writes in the same run: every file bit for bit
#   sass     the card code: cuobjdump -sass and -ptx of both rig builds' llama-server; one SHA-1 per kernel copy
#            (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/) and per PTX function
#            (ptx_functions.py, from docs/data/2026-11-24_c985_topk_order/run/)
#   cpu      cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the head: a CPU build of every target and
#            ctest without network; ctest test by test and the warnings against BASE_CTEST and BASE_CPU_LOG
#   tables   run/c877_tables.py over the session's output, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> BASE_RIG=<base rig build folder>
#   BASE_RIG_LOG=<its build log> BASE_CTEST=<base ctest table> BASE_CPU_LOG=<base CPU build log>
#   REPO=<a copy of this repository> FLINT=<Flint Pascal Q8 GGUF>
#   [MOUNTS='<folders the GPU containers mount read-only>'] [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>]
#   [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <ids>">] [STEPS=...] [JOBS=8] [GPUS=0,1,2] [TS=1,1,1]
#   [ARMS="off wait nowait basewait"] [ROUNDS=1] [NAME=c877]
#   bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the builds and CPU runs without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-host-profile-nowait}
HEAD_SHA=${HEAD_SHA:-7d182f9b0f53d29d4bdc94f2ef8598e9fbfce976}
BASE_SHA=${BASE_SHA:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig session archs sass cpu tables}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18877}
ARMS=${ARMS:-off wait nowait basewait}
ROUNDS=${ROUNDS:-1}
LV=${LV:-4}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c877}
D=$REPO/docs/data/2026-11-24_c877_host_profile_nowait/run
SESSION=$REPO/docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
VOL=(-v "$W:$W" -v "$REPO:$REPO:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
# the PTX JIT's cache, shared by every GPU container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
mkdir -p "$W/cpu" "$W/rig" "$W/logs" "$W/checks" "$W/session" "$W/jitcache" "$W/tables"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
export LC_ALL=C # sort and comm in one collation
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" fetch -q "$SEED" 'refs/tags/*:refs/tags/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    ls "$T/bin" | tr '\n' ' '; echo
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$BASE_RIG/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "--- only in base:"; comm -23 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step session; then
    T=$(head_rig)
    O=$W/session
    [ -n "$T" ] || { echo "no head build"; finish 1; }
    md5sum "$T/bin/llama-server" "$BASE_RIG/bin/llama-server" "$FLINT.sha256" "$SESSION" "$D/session_marks.py" \
        "$REPO/release/flint-v5/settings.env" > "$O/md5.txt" 2>&1
    unset PRESET_MODEL PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1091
    . "$REPO/release/flint-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--pipe-depth 4"* ]] || { say "no -ts or --pipe-depth 4 in the preset"; finish 1; }
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$O/preflight.txt" 2>&1; say "preflight on cards $GPUS rc=$?"; fi
    say "session: arms '$ARMS', rounds $ROUNDS, cards $GPUS"
    for a in $ARMS; do
        o=$O/$a c=$NAME-$a up=0 bin=$T/bin e=""
        case $a in
            off) e="" ;;
            wait) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv" ;;
            nowait) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv GGML_CUDA_PROFILE_HOST_NOWAIT=1" ;;
            basewait) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv"; bin=$BASE_RIG/bin ;;
            *) say "unknown arm $a"; continue ;;
        esac
        rm -rf "$o"; mkdir -p "$o"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\n' "$bin" "$PRESET_ENV" "$e" "$FLINT" "$ARGS" "$LV" > "$o/command.txt"
        say "$a: load (${e:-profile unset})"
        # shellcheck disable=SC2086
        docker run --rm -d $LBL --cpu-shares=256 "${VOL[@]}" -v "$BASE_RIG:$BASE_RIG:ro" "${JIT[@]}" --name "$c" \
            --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" bash -c "env $PRESET_ENV $e $bin/llama-server \
                -m $FLINT $ARGS -lv $LV --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        echo "up=$up" > "$o/load.txt"
        if [ $up = 1 ]; then
            for r in $(seq 1 "$ROUNDS"); do
                timeout 1800 python3 -I "$D/session_marks.py" "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" > "$o/session-r$r.out" 2>&1
                say "  $a round $r: rc=$? $(awk -F'\t' -v r="r$r" '$1 == r' "$o/session.tsv" | wc -l) requests,"\
                    "$(awk -F'\t' -v r="r$r" '$1 == r && $5 > 0' "$o/session.tsv" | wc -l) with tokens"
            done
        fi
        # SIGTERM to the server itself (the container's shell, its first process, would not pass docker stop's on):
        # the server frees its context, so the profile writes what it still holds and its count lines
        docker exec "$c" bash -c 'for p in /proc/[0-9]*; do [ "$(cat $p/comm 2>/dev/null)" = llama-server ] && kill -TERM ${p#/proc/}; done' \
            > /dev/null 2>&1
        t0=$(date +%s)
        while [ ! -f "$o/server.rc" ] && [ $(( $(date +%s) - t0 )) -lt 120 ]; do sleep 2; done
        docker stop -t 30 "$c" > /dev/null 2>&1
        docker rm -f "$c" > /dev/null 2>&1
        say "$a: $(cat "$o/load.txt"), $(cat "$o/server.rc" 2>/dev/null); profile lines $( [ -f "$o/hp.csv" ] && echo $(( $(wc -l < "$o/hp.csv") - 1 )) || echo none);"\
            "count lines $(grep -ac 'ggml_cuda_profile_host_flush:' "$o/server.log"); pipelined lines $(grep -ac 'pipelined = ' "$o/server.log")"
        grep -a 'ggml_cuda_profile_host\(_flush\)\?:' "$o/server.log" | sed 's/^/    /'
    done
    echo "=== session done $(date -Is)"
fi

if step archs; then
    T=$(head_rig)
    K=$W/checks/archs
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        run() { env $3 nice -n 19 timeout 3600 $2/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}" >> /k/rc.txt; }
        run head-plain /head/bin ""
        run base-write /base/bin LLAMA_ARCHS_LOGITS=/k/logits-
        ls /k/logits-*.f32 | wc -l > /k/n_files.txt
        run head-compare /head/bin LLAMA_ARCHS_LOGITS=/k/logits-
        rm -f /k/logits-*.f32
        chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/rc.txt"
    echo "files written by base: $(cat "$K/n_files.txt"); compared: $(grep -ac 'logits against' "$K/head-compare.log");"\
        "differ: $(grep -a 'logits against' "$K/head-compare.log" | grep -vc ', 0 of ')"
    for f in head-plain base-write head-compare; do echo "$f: $(grep -a -m 1 '^Summary' "$K/$f.log")"; done
    echo "=== archs done $(date -Is)"
fi

if step sass; then
    T=$(head_rig)
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S"
    cuo() { local d; d=$(dirname "$2"); docker run --rm $LBL --cpu-shares=256 --network none -v "$d:$d:ro" "$IMAGE" cuobjdump "$1" "$2"; }
    for x in B N; do b=$BASE_RIG/bin/llama-server; [ $x = N ] && b=$T/bin/llama-server
        md5sum "$b" >> "$S/md5.txt"
        cuo -sass "$b" | nice -n 19 c++filt > "$S/dump-$x.txt"
        cuo -ptx "$b" > "$S/ptx-$x.txt"
        nice -n 19 python3 -I "$REPO/docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py" hash < "$S/dump-$x.txt" > "$S/h-$x.tsv"
    done
    {
        echo "SASS dump: base $(wc -l < "$S/dump-B.txt") lines, head $(wc -l < "$S/dump-N.txt")"
        echo "PTX dump: base $(wc -l < "$S/ptx-B.txt") lines, head $(wc -l < "$S/ptx-N.txt")"
        echo "kernel copies hashed: base $(($(wc -l < "$S/h-B.tsv") - 1)), head $(($(wc -l < "$S/h-N.tsv") - 1))"
        echo "kernel copies only in base (name, hash):"; comm -23 <(tail -n +2 "$S/h-B.tsv" | sort) <(tail -n +2 "$S/h-N.tsv" | sort) | cut -c1-200
        echo "kernel copies only in head (name, hash):"; comm -13 <(tail -n +2 "$S/h-B.tsv" | sort) <(tail -n +2 "$S/h-N.tsv" | sort) | cut -c1-200
    } | tee "$S/summary.txt"
    # the PTX function by function (the compute 6.1 code the newer cards compile)
    nice -n 19 python3 -I "$REPO/docs/data/2026-11-24_c985_topk_order/run/ptx_functions.py" "$S/ptx-B.txt" "$S/ptx-N.txt" \
        | tee -a "$S/summary.txt"
    rm -f "$S/dump-B.txt" "$S/dump-N.txt" "$S/ptx-B.txt" "$S/ptx-N.txt"
    echo "=== sass done $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$W/rig/src" TREES="hostprof=$HEAD_SHA" STEPS="tree build ctest" JOBS="$JOBS" CTEST_JOBS=6 \
        THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$W/logs/cpu-head.log" 2>&1
    echo "head cpu tests rc=$?"
    grep -P '^hostprof\t' "$W/cpu/summary.tsv"
    K=$W/checks
    warnings "$BASE_CPU_LOG" > "$K/cpu-warnings-base.txt"
    warnings "$W/cpu/hostprof/build.log" > "$K/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$K/cpu-warnings-base.txt") lines, head $(wc -l < "$K/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt"
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/hostprof/ctest.log" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$K/ctest-head.tsv"
    echo "ctest: head $(wc -l < "$K/ctest-head.tsv") tests ($(grep -c 'Passed$' "$K/ctest-head.tsv") passed)"
    echo "ctest, test by test, head against $BASE_CTEST: $(diff -q "$BASE_CTEST" "$K/ctest-head.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$BASE_CTEST" "$K/ctest-head.tsv"
    echo "=== cpu done $(date -Is)"
fi

if step tables; then
    nice -n 19 python3 -I "$D/c877_tables.py" "$W/session" "$W/tables" $ARMS | tee "$W/tables/summary.txt"
    echo "=== tables done $(date -Is)"
fi
finish 0
