#!/usr/bin/env bash
# C985 (Q1718, 2026-11-24): GGML_CUDA_TOP_K_SORT, the card's top-k candidates in a fixed order (fork branch
# pascal-p106-v5-topk-order, one commit on pascal-p106-v5 at 8a9629942), built and checked on a build machine with
# three newer NVIDIA cards, which run the rig build's compute 6.1 PTX through the driver's JIT. Correctness only:
# nothing here is timed. Steps (STEPS, in this order):
#   src      a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the branch
#            fetched and checked against HEAD_SHA, its diff against the base listed; compiler caches hard-linked from
#            RIG_CCACHE and CPU_CCACHE
#   rig      the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, the base's two extra targets
#            (test-llama-archs, test-cuda-keepalive) and two more (test-top-k-sort, test-backend-sampler), the web page
#            pinned (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s; warnings against BASE_RIG_LOG
#   units    on one card (GPU_UNIT), each in a container of its own, the switch unset and GGML_CUDA_TOP_K_SORT=1:
#            test-top-k-sort on C962's three probe inputs and its two tie rows (2,000 calls each, as one row and four,
#            plain and with the input written before every call; with the switch: --expect-sorted);
#            test-backend-ops -o TOP_K; test-backend-sampler --device gpu on MODEL_SMALL, every test in a process of
#            its own
#   session  C962's R14 in the session's own context under the pipe (docs/data/2026-11-23_c962_sampled_reply_gpu/run/
#            session_r14.py, C950's run/session.py): Flint Pascal Q8 GGUF (FLINT) with its v5 preset, the split changed
#            to the three cards (GPUS, TS), one load per arm, the arms alternating on (GGML_CUDA_TOP_K_SORT=1) and off;
#            each load runs the session once per entry of VARIANTS (R1 to R13 greedy with the prompt cache, then R14)
#   archs    test-llama-archs -b CPU -s 1 of the head's rig build against the logits the base's rig build (BASE_RIG)
#            writes in the same run: every file bit for bit
#   sass     the card code: cuobjdump -sass and -ptx of both rig builds' llama-server; one SHA-1 per kernel copy
#            (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/), the kernels only in one build listed
#   cpu      cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the head: a CPU build of every target and
#            ctest without network; ctest test by test and the warnings against BASE_CTEST and BASE_CPU_LOG (the base
#            engine's CPU build log and ctest table)
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> BASE_RIG=<base rig build folder>
#   BASE_RIG_LOG=<its build log> BASE_CTEST=<base ctest table> BASE_CPU_LOG=<base CPU build log>
#   REPO=<a copy of this repository> FLINT=<Flint Pascal Q8 GGUF> MODEL_SMALL=<a small GGUF with Flint's vocabulary>
#   [MOUNTS='<folders the GPU containers mount read-only>'] [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>]
#   [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <ids>">] [STEPS=...] [JOBS=8] [GPU_UNIT=2] [GPUS=0,1,2] [TS=1,1,1]
#   [ARMS="on-a off-a on-b off-b on-c off-c"] [VARIANTS="pipe pipe pipe pipe pipe p95 greedy"] [CALLS=2000] [NAME=c985]
#   bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1 (default c985; a second run beside the first, in another W, takes
# another NAME), --cpu-shares=256; the builds and CPU runs without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-topk-order}
HEAD_SHA=${HEAD_SHA:-931d777f58b5179a768179b5309c4f12e7d941d5}
BASE_SHA=${BASE_SHA:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig units session archs sass cpu}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
GPU_UNIT=${GPU_UNIT:-2}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18985}
ARMS=${ARMS:-on-a off-a on-b off-b on-c off-c}
VARIANTS=${VARIANTS:-pipe pipe pipe pipe pipe p95 greedy}
CALLS=${CALLS:-2000}
LV=${LV:-4}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c985}
SW=GGML_CUDA_TOP_K_SORT
D962=$REPO/docs/data/2026-11-23_c962_sampled_reply_gpu/run
SESSION=$REPO/docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
VOL=(-v "$W:$W" -v "$REPO:$REPO:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
[ -n "${MODEL_SMALL:-}" ] && VOL+=(-v "$MODEL_SMALL:$MODEL_SMALL:ro")
# the PTX JIT's cache, shared by every GPU container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
mkdir -p "$W/cpu" "$W/rig" "$W/logs" "$W/checks" "$W/units" "$W/session" "$W/jitcache"
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
# shellcheck disable=SC2086
gbox() { local g=$1; shift; docker run --rm $LBL --cpu-shares=256 --gpus "\"device=$g\"" "${VOL[@]}" "${JIT[@]}" "$IMAGE" "$@"; }
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
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" \
        EXTRA_TARGETS="test-llama-archs test-cuda-keepalive test-top-k-sort test-backend-sampler" \
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

if step units; then
    T=$(head_rig)
    U=$W/units
    [ -n "$T" ] || { echo "no head build"; finish 1; }
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPU_UNIT" > "$U/preflight.txt" 2>&1; say "preflight on card $GPU_UNIT rc=$?"; fi
    IN=$(ls "$D962"/probe/*.in | tr '\n' ' ')
    for s in off on; do
        e=""; x=""; [ $s = on ] && { e="$SW=1"; x="--expect-sorted"; }
        for mode in plain rewrite; do
            r=""; [ $mode = rewrite ] && r="-r"
            # shellcheck disable=SC2086
            gbox "$GPU_UNIT" bash -c "env $e timeout 1800 $T/bin/test-top-k-sort -n $CALLS $r $x $IN" > "$U/topk-$s-$mode.txt" 2>&1
            say "test-top-k-sort $s $mode: rc=$? $(grep -a '^result:' "$U/topk-$s-$mode.txt")"
        done
        gbox "$GPU_UNIT" bash -c "env $e timeout 1800 $T/bin/test-backend-ops -o TOP_K -b CUDA0" > "$U/ops-$s.txt" 2>&1
        say "test-backend-ops TOP_K $s: rc=$? $(grep -aE 'tests passed' "$U/ops-$s.txt" | tr '\n' ' ')"
        : > "$U/sampler-$s.tsv"
        for t in $(git -C "$W/rig/src" show "$HEAD_SHA:tests/test-backend-sampler.cpp" | grep -oE '^ *\{ "test_[a-z0-9_]+"' | grep -oE 'test_[a-z0-9_]+'); do
            gbox "$GPU_UNIT" bash -c "env $e timeout 900 $T/bin/test-backend-sampler --device gpu --test $t $MODEL_SMALL" \
                > "$U/sampler-$s-$t.txt" 2>&1
            printf '%s\t%s\n' "$t" "$?" >> "$U/sampler-$s.tsv"
        done
        say "test-backend-sampler $s: $(awk -F'\t' '$2 == 0' "$U/sampler-$s.tsv" | wc -l) of $(wc -l < "$U/sampler-$s.tsv") exit 0;"\
            "not 0: $(awk -F'\t' '$2 != 0 {print $1 "=" $2}' "$U/sampler-$s.tsv" | tr '\n' ' ')"
    done
    echo "=== units done $(date -Is)"
fi

if step session; then
    T=$(head_rig)
    O=$W/session
    md5sum "$T/bin/llama-server" "$FLINT.sha256" "$SESSION" "$D962/session_r14.py" "$REPO/release/flint-v5/settings.env" > "$O/md5.txt" 2>&1
    unset PRESET_MODEL PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1091
    . "$REPO/release/flint-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--pipe-depth 4"* ]] || { say "no -ts or --pipe-depth 4 in the preset"; finish 1; }
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$O/preflight.txt" 2>&1; say "preflight on cards $GPUS rc=$?"; fi
    say "session: arms '$ARMS', variants '$VARIANTS', cards $GPUS"
    for a in $ARMS; do
        o=$O/$a c=$NAME-$a up=0 e=""
        case $a in on-*) e="$SW=1" ;; off-*) e="" ;; *) say "unknown arm $a"; continue ;; esac
        rm -rf "$o"; mkdir -p "$o"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\n' "$T/bin" "$PRESET_ENV" "$e" "$FLINT" "$ARGS" "$LV" > "$o/command.txt"
        say "$a: load (${e:-switch unset})"
        # shellcheck disable=SC2086
        docker run --rm -d $LBL --cpu-shares=256 "${VOL[@]}" "${JIT[@]}" --name "$c" --gpus "\"device=$GPUS\"" \
            -p "127.0.0.1:$PORT:$PORT" "$IMAGE" bash -c "env $PRESET_ENV $e $T/bin/llama-server -m $FLINT $ARGS -lv $LV \
                --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        echo "up=$up load_s=$(( $(date +%s) - t0 ))" > "$o/load.txt"
        if [ $up = 1 ]; then
            i=0
            for v in $VARIANTS; do
                i=$((i + 1))
                timeout 1800 python3 -I "$D962/session_r14.py" "http://127.0.0.1:$PORT" "$SESSION" "$o" "v$i" "$v" >> "$o/session_r14.out" 2>&1
                say "  $a v$i $v: R14 $(awk -F'\t' -v t="v$i" '$1 == t && $3 == "R14" {print $7}' "$o/session_r14.tsv")"
            done
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
        fi
        docker stop -t 60 "$c" > /dev/null 2>&1
        docker rm -f "$c" > /dev/null 2>&1
        say "$a: $(cat "$o/load.txt"); switch line: $(grep -ac "$SW=1" "$o/server.log"), top-k rows line: $(grep -ac 'one top-k over the output rows' "$o/server.log")"
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
    WORK="$W/cpu" SRC="$W/rig/src" TREES="topk=$HEAD_SHA" STEPS="tree build ctest" JOBS="$JOBS" CTEST_JOBS=6 \
        THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$W/logs/cpu-head.log" 2>&1
    echo "head cpu tests rc=$?"
    grep -P '^topk\t' "$W/cpu/summary.tsv"
    K=$W/checks
    warnings "$BASE_CPU_LOG" > "$K/cpu-warnings-base.txt"
    warnings "$W/cpu/topk/build.log" > "$K/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$K/cpu-warnings-base.txt") lines, head $(wc -l < "$K/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt"
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/topk/ctest.log" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$K/ctest-head.tsv"
    echo "ctest: head $(wc -l < "$K/ctest-head.tsv") tests ($(grep -c 'Passed$' "$K/ctest-head.tsv") passed)"
    echo "ctest, test by test, head against $BASE_CTEST: $(diff -q "$BASE_CTEST" "$K/ctest-head.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$BASE_CTEST" "$K/ctest-head.tsv"
    echo "=== cpu done $(date -Is)"
fi
finish 0
