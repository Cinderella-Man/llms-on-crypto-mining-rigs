#!/usr/bin/env bash
# C1025 (Q1758, 2026-11-24): upstream b12674 (#30672) opens NVML at every CUDA start (dlopen of libnvidia-ml.so.1,
# nvmlInit, a brand query per card with fast BF16) to pick cuBLAS's accumulation for quantized products on Quadro- and
# RTX-branded cards. Fork branch pascal-p106-v5-nvml-probe (one commit on the v5 candidate pascal-p106-v5-cand-b12674,
# eee615cdb) skips the probe when no card reads its answer (no card with fast BF16, compute 8.0 and newer), and
# GGML_CUDA_NVML_PROBE=1 / =0 probes always / never. Built and checked on a build machine with newer NVIDIA cards
# (compute 8.6, which do have fast BF16): they run the rig build's compute 6.1 PTX through the driver's JIT, and with no
# newer code in the build their quantized prompt products of 64 or more tokens go to cuBLAS, where the choice is made.
# Correctness only: nothing here is timed (the probe's own line prints times; they are not results on this machine).
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the candidate), the fork as remote "fork", the branch
#           fetched and checked against HEAD_SHA, its diff against the candidate listed; the compiler cache hard-linked
#           from RIG_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and EXTRA_TARGETS, the web page pinned
#           (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s; warnings against CAND_RIG_LOG
#   sass    cuobjdump -sass of the candidate's llama-server (CAND_RIG) and the head's: one SHA-1 per kernel copy
#           (sass_compare.py, from docs/data/2026-11-17_c861a_fp32_scale/run/), the names whose copies differ, and for
#           those whether only constants differ (sass_classify.py, from docs/data/2026-11-24_c912_upstream_b12674/run/)
#   check   on one card (GPU, after PREFLIGHT): (1) llama-server -lv 4 --list-devices under LD_DEBUG=files, the
#           candidate and the head with GGML_CUDA_NVML_PROBE unset, 0, 1 and 2 (invalid): which loaded libnvidia-ml
#           and the head's probe line; (2) one load of MODEL each on the candidate and the head unset, =0 and =1, with
#           cuBLAS's API log on (CUBLAS_LOGINFO_DBG): two greedy requests (a prompt of over 64 tokens, whose quantized
#           products go to cuBLAS on these cards, and a short one), 64 tokens each, token ids compared with the
#           candidate's, and every cuBLAS call's function and compute type counted
#   tables  run/c1025_tables.py over the outputs, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the candidate> CAND_RIG=<the candidate's rig build folder>
#   CAND_RIG_LOG=<its build log> REPO=<a copy of this repository> MODEL=<a small dense-attention GGUF>
#   [RIG_CCACHE=<dir>] [LOCK=<flock file>] [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <id>">]
#   [MOUNTS='<folders the GPU containers mount read-only>'] [STEPS=...] [JOBS=8] [GPU=2] [PORT=18925]
#   [JIT_SEED=<a PTX JIT cache to copy>] [NAME=c1025] bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the cuobjdump containers without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the candidate>}
CAND_RIG=${CAND_RIG:?CAND_RIG=<the candidate rig build folder>}
CAND_RIG_LOG=${CAND_RIG_LOG:?CAND_RIG_LOG=<its build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-nvml-probe}
HEAD_SHA=${HEAD_SHA:-88b0db4032e755a52225b8269361de62f01ff4d9}
CAND_REF=${CAND_REF:-fork/pascal-p106-v5-cand-b12674}
CAND_SHA=${CAND_SHA:-eee615cdb9d26c917de01e3c49f794c41de31684}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12674}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-top-k-sort}
STEPS=${STEPS:-src rig sass check tables}
JOBS=${JOBS:-8}
GPU=${GPU:-2}
PORT=${PORT:-18925}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c1025}
D=$REPO/docs/data/2026-11-24_c1025_nvml_probe/run
SASS_PY=$REPO/docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py
CLASSIFY_PY=$REPO/docs/data/2026-11-24_c912_upstream_b12674/run/sass_classify.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
VOL=(-v "$W:$W" -v "$CAND_RIG:$CAND_RIG:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
mkdir -p "$W/rig" "$W/logs" "$W/checks" "$W/jitcache" "$W/tables"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
export LC_ALL=C
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
build_dir() { ls -d "$1"/llama.cpp-*-"${2:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# a clone of SEED at $1 with upstream as origin and the fork as "fork" (build_llamacpp_docker.sh fetches both)
clone_seed() {
    local S=$1
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || return 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" fetch -q "$SEED" 'refs/tags/*:refs/tags/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
}
echo "=== start $(date -Is): $REF ($HEAD_SHA) on $CAND_REF ($CAND_SHA); steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    clone_seed "$S" || finish 1
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" "${CAND_REF#fork/}:refs/remotes/$CAND_REF" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF"); cand=$(git -C "$S" rev-parse "refs/remotes/$CAND_REF")
    echo "$REF = $got; $CAND_REF = $cand"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    [ "$cand" = "$CAND_SHA" ] || { echo "$CAND_REF moved: stop"; finish 1; }
    { git -C "$S" log --oneline "$CAND_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$CAND_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
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
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$CAND_RIG/bin" && md5sum *) > "$W/rig/bin-md5-cand.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), candidate $(wc -l < "$W/rig/bin-md5-cand.txt");"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-cand.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep -c '^[<>]')"
    grep -aE '^(name|commit|web page|avx|highest glibc|check tools):' "$T/BUILDINFO.txt"
    warnings "$CAND_RIG_LOG" > "$W/checks/rig-warnings-cand.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: candidate $(wc -l < "$W/checks/rig-warnings-cand.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-cand.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-cand.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step sass; then
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S/bin"
    H=$(build_dir "$W/rig" "$HEAD_SHA")
    for x in C H; do
        case $x in C) b=$CAND_RIG/bin/llama-server ;; H) b=$H/bin/llama-server ;; esac
        ln -f "$b" "$S/bin/$x-llama-server" 2> /dev/null || cp "$b" "$S/bin/$x-llama-server"
        echo "$x $(md5sum < "$S/bin/$x-llama-server" | cut -c1-32) $b" >> "$S/md5.txt"
        docker run --rm $LBL --cpu-shares=256 --network none -v "$S/bin:$S/bin:ro" "$IMAGE" cuobjdump -sass "$S/bin/$x-llama-server" \
            | nice -n 19 c++filt > "$S/dump-$x.txt"
        nice -n 19 python3 -I "$SASS_PY" hash < "$S/dump-$x.txt" > "$S/h-$x.tsv"
        say "sass $x: $(($(wc -l < "$S/h-$x.tsv") - 1)) kernel copies"
    done
    # every kernel of the candidate against the head's, as C912's sass_check.sh compares two builds
    O=$S/CH; mkdir -p "$O"
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-C.tsv" > "$O/h-B-off.tsv"
    nice -n 19 python3 -I "$SASS_PY" compare "$O/h-B-off.tsv" "$S/h-H.tsv" "$O/cmp" > "$O/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$O/cmp.identity.tsv" > "$O/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$O/cmp.identity.tsv" > "$O/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
        "$S/h-C.tsv" "$S/h-H.tsv" | sort -u > "$O/only_new.list"
    nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-C.txt" > "$O/ext-B.txt"
    nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-H.txt" > "$O/ext-N.txt"
    nice -n 19 python3 -I "$CLASSIFY_PY" "$O/h-B-off.tsv" "$S/h-H.tsv" "$O/ext-B.txt" "$O/ext-N.txt" \
        > "$O/classes.tsv" 2> "$O/classes_summary.txt"
    {
        echo "--- candidate against head: $(head -n 1 "$O/summary.txt")"
        echo "    names whose copies differ $(wc -l < "$O/differ.list"), missing $(wc -l < "$O/missing.list"), only in head $(wc -l < "$O/only_new.list")"
        sed 's/^/    /' "$O/classes_summary.txt" | head -n 12
    } | tee "$S/pairs.txt"
    rm -f "$O/ext-B.txt" "$O/ext-N.txt" "$S"/dump-?.txt "$S"/bin/?-llama-server
    echo "=== sass done $(date -Is)"
fi

if step check; then
    : "${MODEL:?MODEL=<a small GGUF>}"
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] || { echo "no head build"; finish 1; }
    o=$W/check; mkdir -p "$o"
    if [ -n "${JIT_SEED:-}" ] && [ -d "$JIT_SEED" ] && [ -z "$(ls -A "$W/jitcache")" ]; then cp -a "$JIT_SEED/." "$W/jitcache/"; fi
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPU" > "$o/preflight.txt" 2>&1; rc=$?; say "preflight on card $GPU rc=$rc"
        [ $rc = 0 ] || finish 1; fi
    md5sum "$MODEL" "$CAND_RIG/bin/llama-server" "$T/bin/llama-server" > "$o/md5.txt"
    DR=(docker run --rm $LBL --cpu-shares=256 "${VOL[@]}" -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
    bin_of() { if [ "$1" = C ]; then echo "$CAND_RIG/bin"; else echo "$T/bin"; fi; }
    # (1) the device list: which builds load libnvidia-ml, and the head's probe line
    for arm in C-unset H-unset H-0 H-1 H-2; do
        b=$(bin_of "${arm%%-*}"); v=${arm#*-}; e=""; [ "$v" = unset ] || e="GGML_CUDA_NVML_PROBE=$v"
        rm -f "$o/ld-list-$arm".*
        "${DR[@]}" --name "$NAME-list-$arm" --gpus "\"device=$GPU\"" "$IMAGE" \
            bash -c "env $e LD_DEBUG=files LD_DEBUG_OUTPUT=$o/ld-list-$arm $b/llama-server -lv 4 --list-devices > $o/list-$arm.log 2>&1;
                echo rc=\$? > $o/list-$arm.rc" > /dev/null 2>&1
        say "list $arm: $(cat "$o/list-$arm.rc"); libnvidia-ml in the loader's trace: $(cat "$o/ld-list-$arm".* 2>/dev/null | grep -c 'file=libnvidia-ml')"
        grep -a 'NVML\|GGML_CUDA_NVML_PROBE' "$o/list-$arm.log" | sed 's/^/    /'
    done
    # (2) one load each: two greedy requests, cuBLAS's API log
    P_LONG='Below is a short report on a small bakery. The bakery opens at six in the morning and closes at two in the afternoon. It sells bread, rolls, two kinds of cake and coffee. On weekdays about one hundred and twenty customers come in; on Saturdays about two hundred. The owner wants to know whether opening one hour later would lose customers. Write a numbered list of three things the owner should measure before deciding, and explain each in one sentence.'
    P_SHORT='The capital of France is'
    for arm in C-unset H-unset H-0 H-1; do
        b=$(bin_of "${arm%%-*}"); v=${arm#*-}; e=""; [ "$v" = unset ] || e="GGML_CUDA_NVML_PROBE=$v"
        l=$o/load-$arm c=$NAME-load-$arm up=0
        rm -rf "$l"; mkdir -p "$l"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'env: %s\nbin: %s\nmodel: %s\n' "$e" "$b" "$MODEL" > "$l/command.txt"
        # shellcheck disable=SC2086
        "${DR[@]}" -d --name "$c" --gpus "\"device=$GPU\"" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" \
            bash -c "env $e CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=$l/cublas.log LD_DEBUG=files LD_DEBUG_OUTPUT=$l/ld \
                $b/llama-server -lv 4 -m $MODEL -ngl 99 -c 4096 -np 1 --host 0.0.0.0 --port $PORT --no-webui \
                > $l/server.log 2>&1; echo rc=\$? > $l/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 2
        done
        echo "up=$up" > "$l/load.txt"
        if [ $up = 1 ]; then
            for p in long short; do
                if [ $p = long ]; then body=$P_LONG; else body=$P_SHORT; fi
                printf '{"prompt": "%s", "n_predict": 64, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false, "return_tokens": true}' "$body" |
                    curl -s -m 300 -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/completion" > "$l/reply-$p.json" 2> "$l/reply-$p.err"
            done
        fi
        docker stop -t 30 "$c" > /dev/null 2>&1
        docker rm -f "$c" > /dev/null 2>&1
        say "load $arm: $(cat "$l/load.txt"); libnvidia-ml in the loader's trace: $(cat "$l"/ld.* 2>/dev/null | grep -c 'file=libnvidia-ml');"\
            "cuBLAS log $(wc -l < "$l/cublas.log" 2>/dev/null || echo 0) lines"
        grep -a 'NVML' "$l/server.log" | sed 's/^/    /'
        gzip -f "$l/cublas.log" 2> /dev/null
    done
    echo "=== check done $(date -Is)"
fi

if step tables; then
    nice -n 19 python3 -I "$D/c1025_tables.py" "$W" "$W/tables" | tee "$W/tables/summary.txt"
    echo "=== tables done $(date -Is)"
fi
finish 0
