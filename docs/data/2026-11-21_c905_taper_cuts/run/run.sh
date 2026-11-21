#!/usr/bin/env bash
# C905 (2026-11-21): the prompt taper's cuts=1 (fork branch pascal-p106-v5-taper-cuts) built and checked on a build
# machine's CPU against its base, pascal-p106-v5. No GPU work: the CUDA build's binaries run their CPU backend against
# the driver stub, and the GPU code is read with cuobjdump.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   quick   fail-fast: cpu_tests.sh builds only test-ubatch-taper of the head (at the cpu step's paths and compiler
#           cache, so that step reuses it), and its unit part runs; a failure stops here
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs, test-cuda-keepalive
#           and test-ubatch-taper added, the web UI archive pinned (HF_UI_VERSION), under LOCK; warnings against
#           BASE_RIG_LOG (the base tarball's build log)
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on base and head, the head's build kept: a CPU
#           build of every target, ctest without network, test-llama-archs (plain, write, compare), the head's logits
#           against the base's, the hand-run tests; then the two builds' warnings compared
#   models  random qwen4exp test models from the head's CPU test-llama-archs (4,096-token vocabulary): the MoE model,
#           and the MoE model with one MTP block plus that block alone as a draft file (--mtp-draft)
#   taper   test-ubatch-taper of the head's CPU build on the MoE model: the unit part, and every arm against the taper
#           off on prompts of LENS tokens, the in-call arms (cuts inside each call, checkpoints filled there) among them
#   plan    test-ubatch-taper --plan-check: per spec of PLAN_SPECS and call size of PLAN_LENS (the call sizes of
#           C222's and R225's replays), one call as llama-server's with in-call checkpoints, and the pieces logged;
#           n_rs_seq 3 (Flash-Next's MTP draft of three) and 0
#   archs   test-llama-archs -b CPU of the head's rig build against the logits of the base's rig build (BASE_RIG)
#   server  llama-server of both rig builds on the MTP model and its draft, with the v5 preset's prompt line (-b 4096
#           -ub 512, 4 checkpoints in device stores, taken inside the calls, the MTP draft of three); runs: base-off,
#           head-off (no taper), base-v5, head-v5 (short=1400:64), head-v5c (short=1400:64,cuts=1), head-v5c-req
#           (the server at short=1400:64, every request sending short=1400:64,cuts=1); client.py's 16 requests each;
#           compare.py writes server.tsv and calls.tsv
#   sass    cuobjdump -sass of both rig llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/)
# Usage: W=<work dir> SEED=<llama.cpp clone> BASE_RIG=<base rig build folder> BASE_RIG_LOG=<its build log>
#   REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py client.py compare.py>
#   [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS=...] [JOBS=8]
#   [SRV_IMAGE=<image with python3 and curl on the CUDA image>] [PY_IMAGE=<python image>] bash run.sh
# Every container: labels pascal-quants=1 and c905=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-taper-cuts}
HEAD_SHA=${HEAD_SHA:-1720f26d3ce7cde6f7de1780fdd499a320971513}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src quick rig cpu models taper plan archs server sass}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
LENS=${LENS:-100,577,583,600,1534,3000,9000}
PLAN_LENS=${PLAN_LENS:-29,30,31,32,33,34,37,71,92,97,102,110,126,144,176,258,292,302,306,352,362,403,430,438,441,449,461,516,520,569,577,583,599,610,775,803,815,896,936,1028,1086,1130,1300,1359,1400,1534,1549,1567,1578,1690,1747,1985,2007,2308,3313}
PLAN_SPECS=${PLAN_SPECS:-short=600:64;short=600:64,cuts=1;short=2048:128;short=2048:128,cuts=1;short=2048:64;short=2048:64,cuts=1;short=1400:64;short=1400:64,cuts=1;short=1023:64;short=1023:64,cuts=1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-pr7-srv:24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
PY_IMAGE=${PY_IMAGE:-pq-pr7-py:3.12}
LBL="--label pascal-quants=1 --label c905=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
UIDGID="$(id -u):$(id -g)"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c905=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
CB=$W/cpu/tree/build/bin
MD=$W/models
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
    WORK="$W/cpu" SRC="$W/rig/src" TREES="c905=$HEAD_SHA:test-ubatch-taper" STEPS="tree build" JOBS="$JOBS" \
        LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$W/logs/quick.log" 2>&1
    echo "quick build rc=$?"
    cat "$W/cpu/summary.tsv"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" \
        "$W/cpu/tree/build/bin/test-ubatch-taper" --unit-only > "$W/logs/quick-unit.log" 2>&1
    rc=$?
    cat "$W/logs/quick-unit.log"
    [ $rc = 0 ] || { grep -a "error:" "$W/cpu/c905/build.log" | head -n 20; echo "quick check failed"; finish 1; }
    echo "=== quick done $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive test-ubatch-taper" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum llama-server test-backend-ops test-llama-archs test-cuda-keepalive test-ubatch-taper llama-bench llama-completion) | tee "$W/rig/bin-md5.txt"
    warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt"), head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step cpu; then
    rm -f "$W/cpu/summary.tsv"
    WORK="$W/cpu" SRC="$W/rig/src" TREES="base=$BASE_SHA c905=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    warnings "$W/cpu/base/build.log" > "$W/checks/cpu-warnings-base.txt"
    warnings "$W/cpu/c905/build.log" > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$W/checks/cpu-warnings-base.txt"), head $(wc -l < "$W/checks/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt"
    for t in base c905; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-$t.tsv"
    done
    echo "ctest, test by test: $(diff -q "$W/checks/ctest-base.tsv" "$W/checks/ctest-c905.tsv" > /dev/null && echo same || echo DIFFERENT)"
    diff "$W/checks/ctest-base.tsv" "$W/checks/ctest-c905.tsv"
    cat "$W/checks/ctest-failed-c905.txt"
    grep -a "test-ubatch-taper" "$W/cpu/c905/ctest.log" | head -n 3
    echo "=== cpu done $(date -Is)"
fi

# a container on the head's CPU build (its shared libraries beside the binaries)
cbox() { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" "$@"; }

if step models; then
    rm -rf "$MD" && mkdir -p "$MD"
    cbox bash -c "export LLAMA_TEST_ARCHS_N_VOCAB=4096
        nice -n 19 $CB/test-llama-archs -a '^qwen4exp\$' -s 1 -o $MD > $MD/gen-plain.log 2>&1; echo \"plain rc=\$?\"
        nice -n 19 $CB/test-llama-archs -a '^qwen4exp\$' -s 1 -o $MD --mtp-draft > $MD/gen-mtp.log 2>&1; echo \"mtp rc=\$?\"
        chown -R $UIDGID $MD"
    ls -la "$MD"
    [ -f "$MD/qwen4exp-moe.gguf" ] && [ -f "$MD/qwen4exp-moe-mtp.gguf" ] && [ -f "$MD/qwen4exp-moe-mtp-draft.gguf" ] \
        || { echo "test models missing"; finish 1; }
    echo "=== models done $(date -Is)"
fi

if step taper; then
    K=$W/checks/taper
    rm -rf "$K" && mkdir -p "$K"
    cbox bash -c "nice -n 19 $CB/test-ubatch-taper -m $MD/qwen4exp-moe.gguf -t $THREADS --lens $LENS --tsv $K/taper.tsv \
        > $K/taper.log 2> $K/taper.err; echo \"test-ubatch-taper rc=\$?\" | tee $K/rc.txt; chown -R $UIDGID $K"
    grep -aE '^unit|^base:|^model:|^PASS|^FAIL' "$K/taper.log"
    grep -aE ' -in' "$K/taper.log" | head -n 200
    grep -a FAIL "$K/taper.err" | head -n 40
    echo "=== taper done $(date -Is)"
fi

if step plan; then
    K=$W/checks/plan
    rm -rf "$K" && mkdir -p "$K"
    for rs in 3 0; do
        cbox bash -c "nice -n 19 $CB/test-ubatch-taper -m $MD/qwen4exp-moe.gguf -t $THREADS --plan-check '$PLAN_LENS' \
            --plan-specs '$PLAN_SPECS' --plan-rs $rs --tsv $K/plans-rs$rs.tsv > $K/plan-rs$rs.log 2>&1; \
            echo \"plan check rs $rs rc=\$?\"; chown -R $UIDGID $K"
        tail -n 2 "$K/plan-rs$rs.log"
        awk -F'\t' 'NR > 1 { n[$1]++; if ($10 != "holds") b[$1]++ } END { for (s in n) printf "  %s: %d calls, %d with a piece under min that is not a whole segment\n", s, n[s], b[s] + 0 }' \
            "$K/plans-rs$rs.tsv" | sort
    done
    echo "=== plan done $(date -Is)"
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

if step server; then
    T=$(head_rig)
    K=$W/checks/server
    rm -rf "$K" && mkdir -p "$K"
    if ! docker image inspect "$SRV_IMAGE" > /dev/null 2>&1; then
        printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 curl > /dev/null && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
            | docker build -q --label pascal-quants=1 -t "$SRV_IMAGE" - || { echo "image build failed"; finish 1; }
    fi
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$REPO:/r:ro" -v "$MD:/m:ro" -v "$K:/o" "$SRV_IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        P=8197
        up() { python3 -I -c "import sys, urllib.request; urllib.request.urlopen(\"http://127.0.0.1:$P/health\", timeout=2)" 2> /dev/null; }
        # the v5 preset prompt line (release/flash-next-v5/settings.env) on the CPU: -b 4096 -ub 512, 4 checkpoints in
        # device stores (CPU buffers here) taken inside the calls, the MTP draft of three; a random model trains on 256
        # tokens, so its context length is raised
        LINE="-m /m/qwen4exp-moe-mtp.gguf -md /m/qwen4exp-moe-mtp-draft.gguf -c 8192 -np 1 -t '"$THREADS"' -b 4096 -ub 512 \
            --ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --cache-ram 0 -bs \
            --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-n-ubatch 512 \
            --override-kv qwen4exp.context_length=int:16384 --no-webui -lv 4 --host 127.0.0.1 --port $P"
        srv() { # <name> <bin dir> <request taper or -> <server args...>
            local name=$1 bin=$2 rt=$3; shift 3
            [ "$rt" = - ] && rt=
            timeout 1800 $bin/llama-server $LINE "$@" > /o/$name.log 2>&1 &
            local pid=$! ok=0
            for _ in $(seq 1 300); do
                up && { ok=1; break; }
                kill -0 $pid 2> /dev/null || break
                sleep 1
            done
            [ $ok = 1 ] && timeout 1700 python3 -I /r/client.py $P /o/$name.jsonl $rt
            kill $pid 2> /dev/null; wait $pid; local rc=$?
            echo "$name up=$ok rc=$rc"
        }
        srv base-off     /base/bin -
        srv head-off     /head/bin -
        srv base-v5      /base/bin - --ubatch-taper short=1400:64
        srv head-v5      /head/bin - --ubatch-taper short=1400:64
        srv head-v5c     /head/bin - --ubatch-taper short=1400:64,cuts=1
        srv head-v5c-req /head/bin short=1400:64,cuts=1 --ubatch-taper short=1400:64
        chown -R '"$UIDGID"' /o'
    echo "server container rc=$?"
    docker run --rm $LBL --cpu-shares=256 --network none -v "$REPO:/r:ro" -v "$K:/o" "$PY_IMAGE" \
        bash -c "python3 -I /r/compare.py /o > /o/server.tsv 2>&1; chown $UIDGID /o/server.tsv /o/calls.tsv"
    cat "$K/server.tsv"
    for n in base-v5 head-v5 head-v5c; do
        echo "--- $n: calls of 577, 583, 1534 tokens"; awk -F'\t' -v r=$n '$1 == r && ($3 == 577 || $3 == 583 || $3 == 1534)' "$K/calls.tsv"
    done
    for n in base-off head-off base-v5 head-v5 head-v5c head-v5c-req; do
        echo "$n: $(grep -ciE 'error|abort|failed to restore' "$K/$n.log") error lines, $(grep -c 'goes inside the call' "$K/$n.log") checkpoints inside a call, $(grep -c 'restored context checkpoint' "$K/$n.log") restored"
    done
    echo "=== server done $(date -Is)"
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
    # as C908 and C910: the kq_fp32=true variants (a second build of 176 kernels under the same names) compared apart
    # (kq.txt), every other kernel by name (summary-off.txt)
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
