#!/usr/bin/env bash
# C907 (2026-11-21): LLAMA_SERVER_CKPT_NOWAIT (fork branch pascal-p106-v5-ckpt-stall) built and checked on a build
# machine's CPU against its base, pascal-p106-v5. No GPU work: the CUDA build's binaries run their CPU backend against
# the driver stub, and the GPU code is read with cuobjdump.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   quick   fail-fast: cpu_tests.sh builds only test-llama-context and test-llama-archs of the head (at the cpu step's
#           paths and compiler cache); random models of four recurrent and hybrid architectures (qwen35moe, Ornith's;
#           qwen35; qwen4exp; mamba) from test-llama-archs -o; test-llama-context's store cases on each (store, store_p,
#           store_at, store_at_p and the new store_nw, store_nw_p); a failure stops here
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs and
#           test-cuda-keepalive added (as the base's tarball), the web UI archive pinned (HF_UI_VERSION), under LOCK;
#           warnings against BASE_RIG_LOG (the base tarball's build log)
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on base and head, the head's build kept: a CPU
#           build of every target, ctest without network, test-llama-archs (plain, write, compare), the head's logits
#           against the base's, the hand-run tests; then the two builds' warnings compared
#   context every test-llama-context case of the head's CPU build on the quick step's four models
#   gen     a random qwen35moe model with Ornith's tokenizer (VOCAB: a vocabulary-only GGUF of the Qwen3.5 tokenizer,
#           248,320 tokens) and a 65,536-token training context, from the head's CPU test-llama-archs
#   server  llama-server of both rig builds on that model with Ornith's v5 preset's prompt line (-c 49152 -np 1 -b 4096
#           -ub 512, 4 checkpoints in device stores taken inside the calls, -bs, the preset's server switches) and
#           Ornith's own chat template (TPL); runs base-off, head-off, head-on (LLAMA_SERVER_CKPT_NOWAIT=1); client.py:
#           R183's four recall prompts (PROMPTS) and a restore pair; then head-cms: the switch unset and
#           --checkpoint-min-step 7680 (C945); compare.py writes server.tsv, calls.tsv, checkpoints.tsv
#   archs   test-llama-archs -b CPU of the head's rig build against the logits of the base's rig build (BASE_RIG)
#   sass    cuobjdump -sass of both rig llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/)
# Usage: W=<work dir> SEED=<llama.cpp clone> BASE_RIG=<base rig build folder> BASE_RIG_LOG=<its build log>
#   VOCAB=<vocabulary GGUF> TPL=<chat template file> PROMPTS=<R183's prompts folder>
#   REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py client.py compare.py>
#   [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS=...] [JOBS=8]
#   [SRV_IMAGE=<image with python3 and curl on the CUDA image>] [PY_IMAGE=<python image>] bash run.sh
# Every container: labels pascal-quants=1 and c907=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
VOCAB=${VOCAB:?VOCAB=<vocabulary GGUF>}
TPL=${TPL:?TPL=<chat template file>}
PROMPTS=${PROMPTS:?PROMPTS=<R183 prompts folder>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-ckpt-stall}
HEAD_SHA=${HEAD_SHA:-0aebc4a5778879898c1a591e706b00b9dccce63b}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src quick rig cpu context gen server archs sass}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-pr7-srv:24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
PY_IMAGE=${PY_IMAGE:-pq-pr7-py:3.12}
LBL="--label pascal-quants=1 --label c907=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
UIDGID="$(id -u):$(id -g)"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c907=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
CB=$W/cpu/tree/build/bin
CM=$W/ctxmodels
MD=$W/models
# a container on the head's CPU build (its shared libraries beside the binaries)
cbox() { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" "$@"; }
# one verdict line per case and run of test-llama-context's log: "<case>  OK|SKIP|FAIL  [reason]"
ctx_rows() { sed 's/\x1b\[[0-9;]*m//g; s/^[0-9][0-9]*\.[0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9][0-9] //' "$2" | \
    awk -v m="$1" '$2 ~ /^(OK|SKIP|FAIL)$/ { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; print m "\t" $1 "\t" $2 "\t" r }'; }
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

if step quick; then
    WORK="$W/cpu" SRC="$W/rig/src" TREES="c907=$HEAD_SHA:test-llama-context,test-llama-archs" STEPS="tree build" JOBS="$JOBS" \
        LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$W/logs/quick.log" 2>&1
    echo "quick build rc=$?"
    cat "$W/cpu/summary.tsv"
    [ -x "$CB/test-llama-context" ] && [ -x "$CB/test-llama-archs" ] || { grep -a "error:" "$W/cpu/c907/build.log" | head -n 20; finish 1; }
    K=$W/checks/quick
    rm -rf "$K" "$CM" && mkdir -p "$K" "$CM"
    cbox bash -c "for a in qwen35moe qwen35 qwen4exp mamba; do
            nice -n 19 $CB/test-llama-archs -a \"^\$a\\\$\" -s 1 -o $CM > $CM/save-\$a.log 2>&1; echo \"save \$a rc=\$?\"
        done
        cd $K && for m in $CM/*.gguf; do
            n=\$(basename \$m .gguf)
            nice -n 19 $CB/test-llama-context -m \$m --test 'store.*' -t $THREADS > $K/context-\$n.log 2>&1; echo \"\$n store cases rc=\$?\"
        done; chown -R $UIDGID $K $CM"
    printf 'model\tcase\tverdict\treason\n' > "$K/store.tsv"
    for f in "$K"/context-*.log; do n=$(basename "$f" .log); ctx_rows "${n#context-}" "$f" >> "$K/store.tsv"; done
    cat "$K/store.tsv"
    nf=$(awk -F'\t' '$3 == "FAIL"' "$K/store.tsv" | wc -l)
    nw=$(awk -F'\t' '$2 ~ /^store_nw/ && $3 == "OK"' "$K/store.tsv" | wc -l)
    echo "store cases: $(($(wc -l < "$K/store.tsv") - 1)) runs, $nf FAIL, store_nw/store_nw_p OK in $nw"
    [ "$nf" = 0 ] && [ "$nw" -gt 0 ] || { echo "quick check failed"; finish 1; }
    echo "=== quick done $(date -Is)"
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
    WORK="$W/cpu" SRC="$W/rig/src" TREES="base=$BASE_SHA c907=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    warnings "$W/cpu/base/build.log" > "$W/checks/cpu-warnings-base.txt"
    warnings "$W/cpu/c907/build.log" > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$W/checks/cpu-warnings-base.txt"), head $(wc -l < "$W/checks/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt"
    for t in base c907; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-$t.tsv"
    done
    echo "ctest, test by test: $(diff -q "$W/checks/ctest-base.tsv" "$W/checks/ctest-c907.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$W/checks/ctest-base.tsv" "$W/checks/ctest-c907.tsv"
    cat "$W/checks/ctest-failed-c907.txt"
    echo "=== cpu done $(date -Is)"
fi

if step context; then
    K=$W/checks/context
    rm -rf "$K" && mkdir -p "$K"
    cbox bash -c "cd $K && for m in $CM/*.gguf; do
            n=\$(basename \$m .gguf)
            nice -n 19 $CB/test-llama-context -m \$m -t $THREADS > $K/context-\$n.log 2>&1; echo \"\$n rc=\$?\"
        done; chown -R $UIDGID $K"
    printf 'model\tcase\tverdict\treason\n' > "$K/context.tsv"
    for f in "$K"/context-*.log; do n=$(basename "$f" .log); ctx_rows "${n#context-}" "$f" >> "$K/context.tsv"; done
    echo "context: $(ls "$CM"/*.gguf | wc -l) models, $(($(wc -l < "$K/context.tsv") - 1)) case runs:" \
        "$(awk -F'\t' 'NR > 1 { n[$3]++ } END { for (v in n) printf "%s %d; ", v, n[v] }' "$K/context.tsv")"
    awk -F'\t' '$3 == "FAIL"' "$K/context.tsv"
    echo "=== context done $(date -Is)"
fi

if step gen; then
    rm -rf "$MD" && mkdir -p "$MD"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$VOCAB:/v/vocab.gguf:ro" "$CPU_IMAGE" bash -c "
        export LLAMA_ARCHS_VOCAB=/v/vocab.gguf LLAMA_ARCHS_N_CTX=65536
        nice -n 19 $CB/test-llama-archs -s 1 -o $MD -a '^qwen35moe\$' > $MD/gen.log 2>&1; echo \"qwen35moe rc=\$?\"
        chown -R $UIDGID $MD"
    ls -la "$MD"
    M=$(ls "$MD"/qwen35moe-moe.gguf 2>/dev/null || ls "$MD"/qwen35moe*.gguf 2>/dev/null | head -n 1)
    [ -n "$M" ] || { echo "test model missing"; tail -n 20 "$MD/gen.log"; finish 1; }
    echo "model: $M"
    echo "=== gen done $(date -Is)"
fi

if step server; then
    T=$(head_rig)
    K=$W/checks/server
    M=$(ls "$MD"/qwen35moe-moe.gguf 2>/dev/null || ls "$MD"/qwen35moe*.gguf 2>/dev/null | head -n 1)
    rm -rf "$K" && mkdir -p "$K"
    if ! docker image inspect "$SRV_IMAGE" > /dev/null 2>&1; then
        printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 curl > /dev/null && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
            | docker build -q --label pascal-quants=1 -t "$SRV_IMAGE" - || { echo "image build failed"; finish 1; }
    fi
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$REPO:/r:ro" -v "$PROMPTS:/p:ro" -v "$M:/m/model.gguf:ro" \
        -v "$TPL:/t/template.jinja:ro" -v "$K:/o" "$SRV_IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        P=8197
        up() { python3 -I -c "import sys, urllib.request; urllib.request.urlopen(\"http://127.0.0.1:$P/health\", timeout=2)" 2> /dev/null; }
        # Ornith v5 preset prompt line (release/ornith-v5/settings.env) on the CPU: its server switches, -c 49152 -np 1
        # -b 4096 -ub 512, 4 checkpoints in device stores (CPU buffers here) taken inside the calls; the random model
        # trains on 65,536 tokens; Ornith own chat template
        export LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_TEXT_REUSE=1
        LINE="-m /m/model.gguf -t '"$THREADS"' -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja \
            --chat-template-file /t/template.jinja --reasoning-format deepseek -b 4096 -ub 512 -fit off \
            --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --no-webui -lv 4 --host 127.0.0.1 --port $P"
        srv() { # <name> <bin dir> <env...>
            local name=$1 bin=$2; shift 2
            env "$@" timeout 3600 $bin/llama-server $LINE > /o/$name.log 2>&1 &
            local pid=$! ok=0
            for _ in $(seq 1 300); do
                up && { ok=1; break; }
                kill -0 $pid 2> /dev/null || break
                sleep 1
            done
            [ $ok = 1 ] && timeout 3500 python3 -I /r/client.py $P /p /o/$name.jsonl > /o/$name.client.log 2>&1
            kill $pid 2> /dev/null; wait $pid; local rc=$?
            echo "$name up=$ok rc=$rc"
        }
        srv base-off /base/bin LLAMA_SERVER_CKPT_NOWAIT=0
        srv head-off /head/bin LLAMA_SERVER_CKPT_NOWAIT=0
        srv head-on  /head/bin LLAMA_SERVER_CKPT_NOWAIT=1
        # C945: the switch unset, checkpoints at least 7,680 tokens apart instead of 8,192 (two 4,096-token batches)
        LINE="$LINE -cms 7680" srv head-cms /head/bin LLAMA_SERVER_CKPT_NOWAIT=0
        chown -R '"$UIDGID"' /o'
    echo "server container rc=$?"
    docker run --rm $LBL --cpu-shares=256 --network none -v "$REPO:/r:ro" -v "$K:/o" "$PY_IMAGE" \
        bash -c "python3 -I /r/compare.py /o > /o/server.tsv 2>&1; chown $UIDGID /o/server.tsv /o/calls.tsv /o/checkpoints.tsv"
    cat "$K/server.tsv"
    cat "$K/calls.tsv"
    for n in base-off head-off head-on head-cms; do
        echo "$n: $(grep -ciE 'error|abort|failed to restore' "$K/$n.log") error lines, $(grep -c 'goes inside the call' "$K/$n.log") checkpoints planned inside a call, $(grep -c 'restored context checkpoint' "$K/$n.log") restored, $(grep -c ', no wait,' "$K/$n.log") filled without a wait"
    done
    echo "=== server done $(date -Is)"
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
    # as C905: the kq_fp32=true variants (a second build of 176 kernels under the same names) compared apart (kq.txt),
    # every other kernel by name (summary-off.txt)
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$S/h-B-off.tsv"
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-N.tsv" > "$S/h-N-off.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B-off.tsv" "$S/h-N-off.tsv" "$S/cmp-off" > "$S/summary-off.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp-off.identity.tsv" > "$S/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp-off.identity.tsv" > "$S/missing.list"
    echo "kq_fp32=true kernels' SHA-1, base against head: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-B.tsv" | sort) \
        <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)" \
        "($(awk -F'\t' '$2 == "true"' "$S/h-B.tsv" | wc -l) and $(awk -F'\t' '$2 == "true"' "$S/h-N.tsv" | wc -l))" > "$S/kq.txt"
    rm -f "$S/dump-B.txt" "$S/dump-N.txt"
    cat "$S/md5.txt" "$S/summary-off.txt" "$S/kq.txt"
    echo "differing kernels: $(wc -l < "$S/differ.list"); missing in head: $(wc -l < "$S/missing.list")"
    echo "=== sass done $(date -Is)"
fi

finish 0
