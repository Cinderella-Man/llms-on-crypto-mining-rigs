#!/usr/bin/env bash
# C1004 (Q1737, 2026-11-24): GGML_CUDA_PROFILE_HOST_ANCHORS, each evaluation's start and end on the card on the host
# profile's clock (fork branch pascal-p106-v5-profile-anchors, two commits on pascal-p106-v5 at af00bb13b), built and
# checked on a build machine with newer NVIDIA cards, which run the rig build's compute 6.1 PTX through the driver's JIT.
# Correctness only: nothing here is timed. Steps (STEPS, in this order):
#   src      a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the branch
#            fetched and checked against HEAD_SHA, its diff against the base listed; the compiler cache hard-linked
#            from RIG_CCACHE
#   rig      the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and the base's extra targets
#            (EXTRA_TARGETS), the web page pinned (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s; warnings
#            against BASE_RIG_LOG
#   sass     the card code: cuobjdump -sass and -ptx of both rig builds' llama-server; one SHA-1 per kernel copy
#            (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/) and per PTX function
#            (ptx_functions.py, from docs/data/2026-11-24_c985_topk_order/run/)
#   session  MODEL (Flint Pascal Q5 GGUF) with Flint's v5 preset (--pipe-depth 4 kept), the split changed to the cards
#            given (GPUS, TS), one load per arm, each running C950's session ROUNDS times through C877's
#            session_marks.py:
#              off       the profile unset
#              wait      GGML_CUDA_PROFILE_HOST=<arm>/hp.csv: the profile's default, which waits (C877's six columns)
#              waitanc   the same and GGML_CUDA_PROFILE_HOST_ANCHORS=1: the host waits for each card's previous
#                        evaluation, so its entry reading after the wait checks the placed end
#              nowait    GGML_CUDA_PROFILE_HOST_NOWAIT=1, whose anchors are on by default: the map
#              nowait0   the same and GGML_CUDA_PROFILE_HOST_ANCHORS=0: C877's six columns again
#            EXTRA_ENV is added to every load (the third pass: GGML_CUDA_TOP_K_SORT=1, so the sampled request R14 gives
#            one reply on every load). Before the first load it waits until no other container holds one of the cards
#            (up to GPU_WAIT seconds) and runs PREFLIGHT on them.
#   tables   run/c1004_tables.py over the session's output, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> BASE_RIG=<base rig build folder>
#   BASE_RIG_LOG=<its build log> REPO=<a copy of this repository> MODEL=<Flint Pascal Q5 GGUF>
#   [MOUNTS='<folders the GPU containers mount read-only>'] [RIG_CCACHE=<dir>] [LOCK=<flock file>]
#   [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <ids>">] [STEPS=...] [JOBS=8] [GPUS=0,1] [TS=1,1]
#   [ARMS="off wait waitanc nowait nowait0"] [ROUNDS=1] [EXTRA_ENV='<VAR=value ...>'] [NAME=c1004]
#   bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the build without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
MODEL=${MODEL:?MODEL=<Flint Pascal Q5 GGUF>}
REF=${REF:-fork/pascal-p106-v5-profile-anchors}
HEAD_SHA=${HEAD_SHA:-cad303eb92f71387586d54f25f566d36e57f265c}
BASE_SHA=${BASE_SHA:-af00bb13b0fd59612bcab545964b61978f04b8d6}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-top-k-sort}
STEPS=${STEPS:-src rig sass session tables}
JOBS=${JOBS:-8}
GPUS=${GPUS:-0,1}
TS=${TS:-1,1}
PORT=${PORT:-19004}
ARMS=${ARMS:-off wait waitanc nowait nowait0}
ROUNDS=${ROUNDS:-1}
LV=${LV:-4}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
GPU_WAIT=${GPU_WAIT:-7200}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c1004}
D=$REPO/docs/data/2026-11-24_c1004_profile_anchors/run
MARKS=$REPO/docs/data/2026-11-24_c877_host_profile_nowait/run/session_marks.py
SESSION=$REPO/docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
VOL=(-v "$W:$W" -v "$REPO:$REPO:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
# the PTX JIT's cache, shared by every GPU container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
mkdir -p "$W/rig" "$W/logs" "$W/checks" "$W/session" "$W/jitcache" "$W/tables"
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
# running containers, not ours, that hold one of the cards in GPUS (--gpus by index or UUID, or every card)
holders() {
    local want=",$GPUS," i
    for i in ${GPUS//,/ }; do want+="$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader 2>/dev/null),"; done
    docker ps -q | while read -r c; do
        docker inspect -f '{{.Name}} L{{index .Config.Labels "'"$NAME"'"}} {{range .HostConfig.DeviceRequests}}{{.Count}}:{{join .DeviceIDs ","}} {{end}}' "$c"
    done | awk -v want="$want" '$2 != "L1" { for (i = 3; i <= NF; i++) { split($i, p, ":");
        if (p[1] == "-1") { print $1; next } n = split(p[2], ids, ",");
        for (j = 1; j <= n; j++) if (index(want, "," ids[j] ",")) { print $1; next } } }'
}
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
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="$EXTRA_TARGETS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    ls "$T/bin" | tr '\n' ' '; echo
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$BASE_RIG/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), base $(wc -l < "$W/rig/bin-md5-base.txt");"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-base.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep -c '^[<>]')"
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "--- only in base:"; comm -23 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
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

if step session; then
    T=$(head_rig)
    O=$W/session
    [ -n "$T" ] || { echo "no head build"; finish 1; }
    md5sum "$T/bin/llama-server" "$SESSION" "$MARKS" "$REPO/release/flint-v5/settings.env" > "$O/md5.txt" 2>&1
    [ -f "$MODEL.sha256" ] && cat "$MODEL.sha256" >> "$O/md5.txt"
    unset PRESET_MODEL PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1091
    . "$REPO/release/flint-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--pipe-depth 4"* ]] || { say "no -ts or --pipe-depth 4 in the preset"; finish 1; }
    t0=$(date +%s)
    while h=$(holders) && [ -n "$h" ]; do
        [ $(( $(date +%s) - t0 )) -lt "$GPU_WAIT" ] || { say "cards $GPUS still held by: $h"; finish 1; }
        sleep 30
    done
    say "cards $GPUS free of other containers (waited $(( $(date +%s) - t0 )) s)"
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$O/preflight.txt" 2>&1; rc=$?; say "preflight on cards $GPUS rc=$rc";
        [ $rc = 0 ] || finish 1; fi
    say "session: arms '$ARMS', rounds $ROUNDS, cards $GPUS"
    for a in $ARMS; do
        o=$O/$a c=$NAME-$a up=0 e=""
        case $a in
            off) e="" ;;
            wait) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv" ;;
            waitanc) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv GGML_CUDA_PROFILE_HOST_ANCHORS=1" ;;
            nowait) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv GGML_CUDA_PROFILE_HOST_NOWAIT=1" ;;
            nowait0) e="GGML_CUDA_PROFILE_HOST=$o/hp.csv GGML_CUDA_PROFILE_HOST_NOWAIT=1 GGML_CUDA_PROFILE_HOST_ANCHORS=0" ;;
            *) say "unknown arm $a"; continue ;;
        esac
        e="${EXTRA_ENV:+$EXTRA_ENV }$e"
        rm -rf "$o"; mkdir -p "$o"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\n' "$T/bin" "$PRESET_ENV" "$e" "$MODEL" "$ARGS" "$LV" > "$o/command.txt"
        say "$a: load (${e:-profile unset})"
        # shellcheck disable=SC2086
        docker run --rm -d $LBL --cpu-shares=256 "${VOL[@]}" "${JIT[@]}" --name "$c" \
            --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" bash -c "env $PRESET_ENV $e $T/bin/llama-server \
                -m $MODEL $ARGS -lv $LV --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        echo "up=$up" > "$o/load.txt"
        if [ $up = 1 ]; then
            for r in $(seq 1 "$ROUNDS"); do
                timeout 1800 python3 -I "$MARKS" "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" > "$o/session-r$r.out" 2>&1
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
            "header $( [ -f "$o/hp.csv" ] && head -n 1 "$o/hp.csv" || echo none); count lines $(grep -ac 'ggml_cuda_profile_host_flush:' "$o/server.log")"
        grep -a 'ggml_cuda_profile_host\(_flush\)\?:' "$o/server.log" | sed 's/^/    /'
    done
    echo "=== session done $(date -Is)"
fi

if step tables; then
    nice -n 19 python3 -I "$D/c1004_tables.py" "$W/session" "$W/tables" $ARMS | tee "$W/tables/summary.txt"
    echo "=== tables done $(date -Is)"
fi
finish 0
