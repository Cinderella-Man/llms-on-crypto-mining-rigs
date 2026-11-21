#!/usr/bin/env bash
# C910 (2026-11-21): Miner Llama's --mtp-device fix (fork branch pascal-p106-v5-mtp-device-mem) built and checked on a
# build machine's CPU against its base, pascal-p106-v5. No GPU work: the CUDA build's binaries run their CPU backend
# against the driver stub, the GPU code is read with cuobjdump, and two local RPC servers stand in for two cards.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs and
#           test-cuda-keepalive added (as the base's tarball), the web UI archive pinned (HF_UI_VERSION), under LOCK;
#           warnings against BASE_RIG_LOG (the base tarball's build log)
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on base and head: a CPU build of every target,
#           ctest without network, test-llama-archs (plain, write, compare), the head's logits against the base's, the
#           hand-run tests; then the two builds' warnings compared
#   gen     the test models, with a real tokenizer (VOCAB): test-llama-archs of the base tree with its --mtp filter
#           widened to qwen35moe (the one change, printed) writes a dense llama model and a qwen35moe MTP model;
#           strip.py drops that model's own MTP head and head norm (".nextn.shared_head_"), so that its MTP graph reads
#           the main output head and norm, and with --mtp-device their copies
#   archs   test-llama-archs -b CPU of the head's rig build against the logits of the base's rig build (BASE_RIG)
#   server  llama-server of both rig builds, nothing set, five requests each (client.py): the dense model plain and
#           pipelined, the MTP model with its MTP draft (drafts dumped with LLAMA_SPEC_DUMP)
#   rpc     a CPU build with GGML_RPC=ON of base and head (llama, ggml-rpc-server, llama-server) and mtp_device_check.cpp
#           against each; per build, two ggml-rpc-server processes on 127.0.0.1 serve their CPU backend as two
#           devices, and on the MTP model: mtp-device-check (the memory fit's view, two loads and frees, the largest
#           host allocation of each load) with all layers on the devices and with none, then llama-server with the MTP
#           draft and -fit off -ngl 999 -ts 1,1 -lv 4 (so the log holds the loader's lines), --mtp-device unset and 0,
#           drafts dumped; compare.py writes server.tsv
#   sass    cuobjdump -sass of both rig llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/), every differing kernel extracted and classified
#           (sass_classify.py, from docs/data/2026-11-17_c854b_upstream_newest/run/)
# Usage: W=<work dir> SEED=<llama.cpp clone> BASE_RIG=<base rig build folder> BASE_RIG_LOG=<its build log>
#   VOCAB=<gguf with a tokenizer> REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py
#   sass_classify.py client.py compare.py strip.py mtp_device_check.cpp> [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>]
#   [LOCK=<flock file>] [STEPS=...] [JOBS=8] [SRV_IMAGE=<image with python3 and curl on the CUDA image>]
#   [PY_IMAGE=<python image with numpy>] bash run.sh
# Every container: labels pascal-quants=1 and c910=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
VOCAB=${VOCAB:?VOCAB=<gguf with a tokenizer>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-mtp-device-mem}
HEAD_SHA=${HEAD_SHA:-f94aa552b44fb6cda6bc918c15ad33e67c34b8c7}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig cpu gen archs server rpc sass}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-pr7-srv:24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
PY_IMAGE=${PY_IMAGE:-pq-pr7-py:3.12}
LBL="--label pascal-quants=1 --label c910=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
UIDGID="$(id -u):$(id -g)"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c910=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
MD=$W/gen/models
M_DENSE=$MD/llama-dense.gguf
M_MTP=$MD/qwen35moe-moe-mtp-mainhead.gguf
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS"

if step src; then
    S=$W/rig/src
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA" > "$W/commits.txt"
    git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA" >> "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum llama-server test-backend-ops test-llama-archs test-cuda-keepalive llama-bench llama-completion) | tee "$W/rig/bin-md5.txt"
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt"), head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step cpu; then
    rm -f "$W/cpu/summary.tsv"
    WORK="$W/cpu" SRC="$W/rig/src" TREES="base=$BASE_SHA c910=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    warnings "$W/cpu/base/build.log" > "$W/checks/cpu-warnings-base.txt"
    warnings "$W/cpu/c910/build.log" > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$W/checks/cpu-warnings-base.txt"), head $(wc -l < "$W/checks/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt"
    for t in base c910; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-$t.tsv"
    done
    echo "ctest, test by test: $(diff -q "$W/checks/ctest-base.tsv" "$W/checks/ctest-c910.tsv" > /dev/null && echo same || echo DIFFERENT)"
    cat "$W/checks/ctest-failed-c910.txt"
    echo "=== cpu done $(date -Is)"
fi

if step gen; then
    GT=$W/gen/tree
    rm -rf "$GT" "$MD" && mkdir -p "$GT/src" "$MD"
    git -C "$W/rig/src" archive "$BASE_SHA" | tar -x -C "$GT/src"
    cp "$GT/src/tests/test-llama-archs.cpp" "$W/gen/test-llama-archs.cpp.orig"
    perl -pi -e 's/if \(mtp && \(arch != LLM_ARCH_QWEN4EXP \|\| !moe\)\) \{/if (mtp && ((arch != LLM_ARCH_QWEN4EXP && arch != LLM_ARCH_QWEN35MOE) || !moe)) {/' \
        "$GT/src/tests/test-llama-archs.cpp"
    echo "--- the one change to the generator:"
    diff "$W/gen/test-llama-archs.cpp.orig" "$GT/src/tests/test-llama-archs.cpp"
    locked docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$W/cpu/ccache" \
        -e CCACHE_BASEDIR="$GT" "$CPU_IMAGE" bash -c "
        cmake -S $GT/src -B $GT/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
            -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $W/gen/cmake.log 2>&1 &&
        nice -n 19 cmake --build $GT/build -j $JOBS --target test-llama-archs > $W/gen/build.log 2>&1"
    echo "generator build rc=$?"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$VOCAB:/v/vocab.gguf:ro" "$CPU_IMAGE" bash -c "
        export LLAMA_ARCHS_VOCAB=/v/vocab.gguf LLAMA_ARCHS_N_CTX=4096
        $GT/build/bin/test-llama-archs -s 1 -o $MD -a '^llama\$' > $MD/gen-llama.log 2>&1; echo \"llama rc=\$?\"
        $GT/build/bin/test-llama-archs -s 1 -o $MD -a '^qwen35moe\$' --mtp > $MD/gen-q35.log 2>&1; echo \"qwen35moe mtp rc=\$?\"
        chown -R $UIDGID $MD"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$GT/src/gguf-py:/gguf-py:ro" -v "$REPO:/r:ro" \
        "$PY_IMAGE" bash -c "python3 -I /r/strip.py /gguf-py $MD/qwen35moe-moe-mtp.gguf $M_MTP .nextn.shared_head_; chown $UIDGID $M_MTP"
    ls -la "$MD"
    rm -rf "$GT"
    [ -f "$M_DENSE" ] && [ -f "$M_MTP" ] || { echo "test models missing"; finish 1; }
    echo "=== gen done $(date -Is)"
fi

if step archs; then
    T=$(head_rig)
    K=$W/checks/archs
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        run() { env $3 nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}" >> /k/rc.txt; }
        run head-plain /head/bin ""
        run base-write /base/bin LLAMA_ARCHS_LOGITS=/k/logits-
        ls /k/logits-*.f32 | wc -l > /k/n_files.txt
        run head-compare /head/bin LLAMA_ARCHS_LOGITS=/k/logits-
        rm -f /k/logits-*.f32
        chown -R '"$UIDGID"' /k'
    cat "$K/rc.txt"
    echo "files written by base: $(cat "$K/n_files.txt"); compared: $(grep -ac 'logits against' "$K/head-compare.log");"\
        "differ: $(grep -a 'logits against' "$K/head-compare.log" | grep -vc ', 0 of ')"
    for f in head-plain base-write head-compare; do echo "$f: $(grep -a -m 1 '^Summary' "$K/$f.log")"; done
    echo "=== archs done $(date -Is)"
fi

# the server function both server steps use, as a bash snippet run inside a container:
# srv <name> <bin dir> <model> <env: NAME=VALUE[,NAME=VALUE], or -> <server args...>
SRV_FN='
P=8197
up() { python3 -I -c "import sys, urllib.request; urllib.request.urlopen(\"http://127.0.0.1:$P/health\", timeout=2)" 2> /dev/null; }
srv() {
    local name=$1 bin=$2 model=$3 e=$4; shift 4
    [ "$e" = - ] && e=
    env ${e//,/ } LLAMA_SPEC_DUMP=/o/$name.dump.jsonl timeout 900 $bin/llama-server -m $model -c 4096 -np 1 -t 6 \
        --cache-ram 0 --no-webui --host 127.0.0.1 --port $P "$@" > /o/$name.log 2>&1 &
    local pid=$! ok=0
    for _ in $(seq 1 300); do
        up && { ok=1; break; }
        kill -0 $pid 2> /dev/null || break
        sleep 1
    done
    [ $ok = 1 ] && timeout 1200 python3 -I /r/client.py $P /o/$name.jsonl
    kill $pid 2> /dev/null; wait $pid; local rc=$?
    [ -s /o/$name.dump.jsonl ] || rm -f /o/$name.dump.jsonl
    echo "$name up=$ok rc=$rc"
}
'

if step server; then
    T=$(head_rig)
    K=$W/checks/server
    mkdir -p "$K"
    rm -f "$K"/base-* "$K"/head-*
    if ! docker image inspect "$SRV_IMAGE" > /dev/null 2>&1; then
        printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 curl > /dev/null && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
            | docker build -q --label pascal-quants=1 -t "$SRV_IMAGE" - || { echo "image build failed"; finish 1; }
    fi
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$REPO:/r:ro" -v "$MD:/m:ro" -v "$K:/o" "$SRV_IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        '"$SRV_FN"'
        D=/m/llama-dense.gguf
        Q=/m/qwen35moe-moe-mtp-mainhead.gguf
        srv base-plain /base/bin $D - -bs
        srv head-plain /head/bin $D - -bs
        srv base-pipe  /base/bin $D - -bs --pipe-depth 3
        srv head-pipe  /head/bin $D - -bs --pipe-depth 3
        srv base-mtp   /base/bin $Q - -bs --spec-type draft-mtp
        srv head-mtp   /head/bin $Q - -bs --spec-type draft-mtp
        chown -R '"$UIDGID"' /o'
    echo "server container rc=$?"
    docker run --rm $LBL --cpu-shares=256 --network none -v "$REPO:/r:ro" -v "$K:/o" "$PY_IMAGE" \
        bash -c "python3 -I /r/compare.py /o > /o/server.tsv 2>&1; chown $UIDGID /o/server.tsv"
    cat "$K/server.tsv"
    echo "=== server done $(date -Is)"
fi

if step rpc; then
    K=$W/checks/server
    mkdir -p "$K"
    rm -f "$K"/rpc-* "$K"/harness-*
    for tag in base head; do
        sha=$BASE_SHA; [ $tag = head ] && sha=$HEAD_SHA
        R=$W/rpc/$tag
        rm -rf "$R" && mkdir -p "$R/src"
        git -C "$W/rig/src" archive "$sha" | tar -x -C "$R/src"
        locked docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:/r:ro" \
            -e CCACHE_DIR="$W/cpu/ccache" -e CCACHE_BASEDIR="$R" "$CPU_IMAGE" bash -c "
            cmake -S $R/src -B $R/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
                -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DGGML_RPC=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $R/cmake.log 2>&1 &&
            nice -n 19 cmake --build $R/build -j $JOBS --target llama ggml-rpc-server llama-server > $R/build.log 2>&1 &&
            nice -n 19 g++ -std=c++17 -O1 -Wall -rdynamic /r/mtp_device_check.cpp -I$R/src/include -I$R/src/ggml/include \
                -I$R/src/src -L$R/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$R/build/bin \
                -o $R/build/bin/mtp-device-check > $R/harness-build.log 2>&1"
        rc=$?
        echo "$tag ($sha) RPC CPU build and harness rc=$rc"
        [ $rc = 0 ] || { tail -n 20 "$R/build.log" "$R/harness-build.log"; finish 1; }
        docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:/r:ro" -v "$MD:/m:ro" -v "$K:/o" \
            "$CPU_IMAGE" bash -c '
            '"$SRV_FN"'
            B='"$R"'/build/bin
            t='"$tag"'
            Q=/m/qwen35moe-moe-mtp-mainhead.gguf
            $B/ggml-rpc-server -H 127.0.0.1 -p 50052 -t 4 > /o/rpc-$t-server0.log 2>&1 & P0=$!
            $B/ggml-rpc-server -H 127.0.0.1 -p 50053 -t 4 > /o/rpc-$t-server1.log 2>&1 & P1=$!
            sleep 3
            EP=127.0.0.1:50052,127.0.0.1:50053
            timeout 900 $B/mtp-device-check $Q $EP $P0,$P1 999 > /o/harness-$t.tsv 2> /o/harness-$t.log; echo "harness $t ngl 999 rc=$?"
            timeout 900 $B/mtp-device-check $Q $EP $P0,$P1 0 > /o/harness-$t-ngl0.tsv 2> /o/harness-$t-ngl0.log; echo "harness $t ngl 0 rc=$?"
            srv rpc-$t-nd-1 $B $Q - --rpc $EP -fit off -ngl 999 -ts 1,1 --spec-type draft-mtp -lv 4
            srv rpc-$t-nd0  $B $Q - --rpc $EP -fit off -ngl 999 -ts 1,1 --spec-type draft-mtp --mtp-device 0 -lv 4
            kill $P0 $P1; wait
            chown -R '"$UIDGID"' /o'
        echo "$tag rpc container rc=$?"
        for f in "$K/harness-$tag.tsv" "$K/harness-$tag-ngl0.tsv"; do echo "--- $(basename "$f")"; cat "$f"; done
        for f in harness-$tag.log rpc-$tag-nd-1.log rpc-$tag-nd0.log; do
            echo "--- $f"; grep -a "MTP output head\|model buffer size\|nextn layer" "$K/$f" | sed -E "s/^[0-9.]+ //" | sort | uniq -c
        done
    done
    docker run --rm $LBL --cpu-shares=256 --network none -v "$REPO:/r:ro" -v "$K:/o" "$PY_IMAGE" \
        bash -c "python3 -I /r/compare.py /o > /o/server.tsv 2>&1; chown $UIDGID /o/server.tsv"
    cat "$K/server.tsv"
    echo "=== rpc done $(date -Is)"
fi

if step sass; then
    T=$(head_rig)
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S"
    B=$BASE_RIG/bin/llama-server
    N=$T/bin/llama-server
    dump() { local d; d=$(dirname "$1"); docker run --rm $LBL --cpu-shares=256 --network none -v "$d:$d:ro" "$IMAGE" \
        cuobjdump -sass "$1" | nice -n 19 c++filt; }
    md5sum "$B" "$N" > "$S/md5.txt"
    dump "$B" > "$S/dump-B.txt"
    dump "$N" > "$S/dump-N.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-B.txt" > "$S/h-B.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-N.txt" > "$S/h-N.tsv"
    # as C908: the base side's kq_fp32-off kernels against the head's, the kq_fp32=true ones compared apart (kq.txt)
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$S/h-B-off.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp.identity.tsv" > "$S/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
        "$S/h-B.tsv" "$S/h-N.tsv" | sort -u > "$S/only_new.list"
    echo "kq_fp32=true kernels' SHA-1, base against head: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-B.tsv" | sort) \
        <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)" > "$S/kq.txt"
    if [ -s "$S/differ.list" ]; then
        nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-B.txt" > "$S/ext-B.txt"
        nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-N.txt" > "$S/ext-N.txt"
        nice -n 19 python3 -I "$REPO/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
        nice -n 19 python3 -I "$REPO/sass_classify.py" "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/ext-B.txt" "$S/ext-N.txt" \
            > "$S/classes.tsv" 2> "$S/classes_summary.txt"
    fi
    rm -f "$S/dump-B.txt" "$S/dump-N.txt"
    cat "$S/md5.txt" "$S/summary.txt" "$S/kq.txt"
    [ -f "$S/linediff.txt" ] && cat "$S/linediff.txt" "$S/classes_summary.txt"
    echo "differing kernels: $(wc -l < "$S/differ.list"); only in head: $(wc -l < "$S/only_new.list"); missing in head: $(wc -l < "$S/missing.list")"
    cat "$S/differ.list" "$S/only_new.list"
    echo "=== sass done $(date -Is)"
fi

finish 0
