#!/usr/bin/env bash
# Pull requests 9 and 10 (2026-11-23): Miner Llama's cumulative branch with both merged (pascal-p106-v5-contrib2), built
# and checked on a build machine against the base 3dfad876f (C908's build). CPU checks first; the last step runs
# test-backend-ops' FLASH_ATTN_EXT on one card through the sm_61 build's compute_61 PTX.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base commit), the fork as remote "fork", the
#           branch fetched and checked against HEAD_SHA; compiler caches hard-linked from RIG_CCACHE and CPU_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, test-llama-archs and
#           test-cuda-keepalive added (as the base's tarball), the web UI archive pinned (HF_UI_VERSION), under LOCK;
#           warnings against BASE_RIG_LOG (the base tarball's build log)
#   cpu     cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on base and head: a CPU build of every target,
#           ctest without network, test-llama-archs (plain, write, compare), the head's logits against the base's, the
#           hand-run tests; then the two builds' warnings compared, and ctest test by test against BASE_CTEST if given
#   archs   test-llama-archs -b CPU of the head's rig build against the logits of the base's rig build (BASE_RIG)
#   server  llama-server of both rig builds on a random llama model with a real tokenizer (VOCAB), five requests each
#           (client.py), no projector: plain, pipelined, and pipelined with LLAMA_PIPE_MEDIA unset, 0, 1 and 2;
#           compare.py writes server.tsv
#   sass    cuobjdump -sass of both llama-servers, one SHA-1 per kernel (sass_compare.py, from
#           docs/data/2026-11-17_c861a_fp32_scale/run/). The head's Pascal tile kernel has two more template flags (lds,
#           kdp); its name is cut back to the old form where both are false (in both builds, so a base that already
#           has the flags compares too), and the default kernels are compared with the base's. Every differing kernel is extracted, compared line by line and classified (sass_classify.py,
#           from docs/data/2026-11-17_c854b_upstream_newest/run/); then each build's register, stack and shared memory
#           use of the Pascal attention kernels (cuobjdump -res-usage), and the decode kernel's global loads in the
#           SASS and in the compute_61 PTX (the PTX is what a newer card compiles)
#   fa      test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT on card GPU through the compute_61 PTX, one run per arm of
#           FA_ARMS (below). GGML_CUDA_FATTN_TILE_PASCAL=2 makes the Pascal kernels take their cases on a card that is
#           not compute 6.1 (on compute 6.1 they do so by default); _FAULT=1 scales their output by 1.05, so every case
#           they take must fail (a check that the cases reach them)
# The fix of the decode kernel's mask read (C973) ran the same script with REF=fork/pascal-p106-v5-dec-mask,
# HEAD_SHA and BASE_SHA set, BASE_RIG the review branch's build and STEPS="src rig sass fa".
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit> BASE_RIG=<base rig build folder>
#   BASE_RIG_LOG=<its build log> VOCAB=<gguf with a tokenizer> REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh
#   sass_compare.py sass_classify.py client.py compare.py> [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>] [BASE_CTEST=<ctest
#   table of the base>] [LOCK=<flock file>] [STEPS=...] [JOBS=8] [SRV_IMAGE=<image with python3 and curl on the CUDA
#   image>] [GPU=2] [FA_ARMS='<names>'] [PREFLIGHT=<script taking -g <gpu>>] bash run.sh
# Every container: labels pascal-quants=1 and pr910=1, --cpu-shares=256; the CPU ones --memory=16g and no network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base commit>}
BASE_RIG=${BASE_RIG:?BASE_RIG=<base rig build folder>}
BASE_RIG_LOG=${BASE_RIG_LOG:?BASE_RIG_LOG=<base rig build log>}
VOCAB=${VOCAB:?VOCAB=<gguf with a tokenizer>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-contrib2}
HEAD_SHA=${HEAD_SHA:-82d9eb5ef243c6166c5095ead1d7d8e43eb8a9cf}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig cpu archs server sass fa}
JOBS=${JOBS:-8}
GPU=${GPU:-2}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-pr910-srv:24.04}
LBL="--label pascal-quants=1 --label pr910=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
mkdir -p "$W/rig" "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=pr910=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
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
    ls "$T/bin" | tr '\n' ' '; echo
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
    WORK="$W/cpu" SRC="$W/rig/src" TREES="base=$BASE_SHA pr910=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu rc=$?"
    cat "$W/cpu/summary.tsv"
    warnings "$W/cpu/base/build.log" > "$W/checks/cpu-warnings-base.txt"
    warnings "$W/cpu/pr910/build.log" > "$W/checks/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$W/checks/cpu-warnings-base.txt"), head $(wc -l < "$W/checks/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/cpu-warnings-base.txt" "$W/checks/cpu-warnings-head.txt"
    for t in base pr910; do
        sed -n '/The following tests FAILED:/,$p' "$W/cpu/$t/ctest.log" | grep -aE '^\s+[0-9]+ - ' \
            | sed -E 's/^\s+[0-9]+ - //' > "$W/checks/ctest-failed-$t.txt"
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$W/checks/ctest-$t.tsv"
    done
    echo "ctest, test by test, head against base: $(diff -q "$W/checks/ctest-base.tsv" "$W/checks/ctest-pr910.tsv" > /dev/null && echo same || echo DIFFERENT)"
    if [ -n "${BASE_CTEST:-}" ]; then
        echo "ctest, test by test, head against $BASE_CTEST: $(diff -q "$BASE_CTEST" "$W/checks/ctest-pr910.tsv" > /dev/null && echo same || echo DIFFERENT)"
        diff "$BASE_CTEST" "$W/checks/ctest-pr910.tsv"
    fi
    echo "=== cpu done $(date -Is)"
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
        chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/rc.txt"
    echo "files written by base: $(cat "$K/n_files.txt"); compared: $(grep -ac 'logits against' "$K/head-compare.log");"\
        "differ: $(grep -a 'logits against' "$K/head-compare.log" | grep -vc ', 0 of ')"
    for f in head-plain base-write head-compare; do echo "$f: $(grep -a -m 1 '^Summary' "$K/$f.log")"; done
    echo "=== archs done $(date -Is)"
fi

if step server; then
    T=$(head_rig)
    K=$W/checks/server
    rm -rf "$K" && mkdir -p "$K/models"
    if ! docker image inspect "$SRV_IMAGE" > /dev/null 2>&1; then
        printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 curl > /dev/null && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
            | docker build -q --label pascal-quants=1 -t "$SRV_IMAGE" - || { echo "image build failed"; finish 1; }
    fi
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$REPO:/r:ro" -v "$VOCAB:/v/vocab.gguf:ro" -v "$K:/o" "$SRV_IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        env LLAMA_ARCHS_VOCAB=/v/vocab.gguf LLAMA_ARCHS_N_CTX=4096 /base/bin/test-llama-archs -s 1 -o /o/models -a "^llama\$" > /o/models/gen.log 2>&1
        M=/o/models/llama-dense.gguf
        ls -la /o/models
        P=8197
        # srv <name> <bin dir> <env: NAME=VALUE, or - for none> <server args...>
        srv() {
            local name=$1 bin=$2 e=$3; shift 3
            [ "$e" = - ] && e=
            env $e timeout 900 $bin/llama-server -m $M -c 4096 -np 1 -t 6 --cache-ram 0 --no-webui \
                --host 127.0.0.1 --port $P "$@" > /o/$name.log 2>&1 &
            local pid=$! ok=0
            for _ in $(seq 1 300); do
                curl -sf 127.0.0.1:$P/health > /dev/null && { ok=1; break; }
                kill -0 $pid 2> /dev/null || break
                sleep 1
            done
            [ $ok = 1 ] && timeout 1200 python3 -I /r/client.py $P /o/$name.jsonl
            kill $pid 2> /dev/null; wait $pid; local rc=$?
            echo "$name up=$ok rc=$rc"
        }
        srv base-plain       /base/bin - -bs
        srv head-plain       /head/bin - -bs
        srv base-pipe        /base/bin - -bs --pipe-depth 3
        srv head-pipe        /head/bin - -bs --pipe-depth 3
        srv head-pipe-media0 /head/bin LLAMA_PIPE_MEDIA=0 -bs --pipe-depth 3
        srv head-pipe-media1 /head/bin LLAMA_PIPE_MEDIA=1 -bs --pipe-depth 3
        srv head-pipe-media2 /head/bin LLAMA_PIPE_MEDIA=2 -bs --pipe-depth 3
        rm -f /o/models/*.gguf
        python3 -I /r/compare.py /o > /o/server.tsv 2>&1
        chown -R '"$(id -u):$(id -g)"' /o'
    echo "server container rc=$?"
    cat "$K/server.tsv"
    echo "=== server done $(date -Is)"
fi

if step sass; then
    T=$(head_rig)
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S"
    B=$BASE_RIG/bin/llama-server
    N=$T/bin/llama-server
    cuo() { local d; d=$(dirname "$2"); docker run --rm $LBL --cpu-shares=256 --network none -v "$d:$d:ro" "$IMAGE" cuobjdump "$1" "$2"; }
    # the tile kernel with lds = kdp = false gets its old name, flash_attn_tile_pascal<D, K, V, split>, in both builds
    cut_flags() { sed -E 's/(flash_attn_tile_pascal<[0-9]+, \(ggml_type\)[0-9]+, \(ggml_type\)[0-9]+, (true|false)), false, false>/\1>/'; }
    md5sum "$B" "$N" > "$S/md5.txt"
    cuo -sass "$B" | nice -n 19 c++filt | cut_flags > "$S/dump-B.txt"
    cuo -sass "$N" | nice -n 19 c++filt | cut_flags > "$S/dump-N.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-B.txt" > "$S/h-B.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" hash < "$S/dump-N.txt" > "$S/h-N.tsv"
    # compare matches each base kernel against the new build's kq_fp32-off kernel of that name, so the base side takes
    # its kq_fp32-off kernels only and the kq_fp32=true ones are compared apart (kq.txt), as C854 step two did
    awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$S/h-B-off.tsv"
    nice -n 19 python3 -I "$REPO/sass_compare.py" compare "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
    awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
    awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp.identity.tsv" > "$S/missing.list"
    awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
        "$S/h-B.tsv" "$S/h-N.tsv" | sort -u > "$S/only_new.list"
    nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-B.txt" > "$S/ext-B.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" extract "$S/differ.list" < "$S/dump-N.txt" > "$S/ext-N.txt"
    nice -n 19 python3 -I "$REPO/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
    echo "kq_fp32=true kernels' SHA-1, base against head: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-B.tsv" | sort) \
        <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)" > "$S/kq.txt"
    nice -n 19 python3 -I "$REPO/sass_classify.py" "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/ext-B.txt" "$S/ext-N.txt" \
        > "$S/classes.tsv" 2> "$S/classes_summary.txt"
    # the Pascal attention kernels: their rows of the hash tables, their SASS, and the decode kernel's global loads
    grep -a 'flash_attn_tile_pascal\|flash_attn_dec_pascal' "$S/h-B.tsv" | cut -f1,3- | sed -E 's/\(char const\*.*\)\t/\t/' > "$S/pascal-B.tsv"
    grep -a 'flash_attn_tile_pascal\|flash_attn_dec_pascal' "$S/h-N.tsv" | cut -f1,3- | sed -E 's/\(char const\*.*\)\t/\t/' > "$S/pascal-N.tsv"
    awk '/Function : /{p = ($0 ~ /flash_attn_dec_pascal/)} p' "$S/dump-N.txt" > "$S/dec.sass"
    grep -aoE '(LDG|LD|LDS|STS|STG)\.[A-Z0-9.]*' "$S/dec.sass" | sort | uniq -c > "$S/dec-sass-mem.txt"
    rm -f "$S/dump-B.txt" "$S/dump-N.txt"
    for x in B N; do b=$B; [ $x = N ] && b=$N
        cuo -res-usage "$b" | nice -n 19 c++filt | awk '/^[ \t]*Function /{f =($0 ~ /flash_attn_tile_pascal|flash_attn_dec_pascal/); if (f) print; next} f {print; f = 0}' > "$S/res-$x.txt"
    done
    # the compute_61 PTX of the decode kernel (what the driver compiles for a newer card): its global loads
    cuo -ptx "$N" | awk '/^[ \t]*(\.visible |\.weak )?\.entry /{p = ($0 ~ /flash_attn_dec_pascal/)} p' > "$S/dec.ptx"
    grep -oE 'ld\.global[.a-z0-9]*' "$S/dec.ptx" | sort | uniq -c > "$S/dec-ptx-loads.txt"
    cat "$S/md5.txt" "$S/summary.txt" "$S/kq.txt" "$S/linediff.txt" "$S/classes_summary.txt"
    echo "differing kernels: $(wc -l < "$S/differ.list"); only in head: $(wc -l < "$S/only_new.list"); missing in head: $(wc -l < "$S/missing.list")"
    cat "$S/differ.list" "$S/only_new.list"
    echo "--- Pascal kernels, base"; cat "$S/pascal-B.tsv"
    echo "--- Pascal kernels, head"; cat "$S/pascal-N.tsv"
    echo "--- resources, base"; cat "$S/res-B.txt"
    echo "--- resources, head"; cat "$S/res-N.txt"
    echo "--- decode kernel, SASS memory instructions"; cat "$S/dec-sass-mem.txt"
    echo "--- decode kernel, PTX global loads ($(wc -l < "$S/dec.ptx") lines)"; cat "$S/dec-ptx-loads.txt"
    echo "=== sass done $(date -Is)"
fi

if step fa; then
    T=$(head_rig)
    K=$W/checks/fa
    J=$W/checks/jitcache
    mkdir -p "$K" "$J"
    [ -n "${PREFLIGHT:-}" ] && { "$PREFLIGHT" -g "$GPU" > "$K/preflight.txt" 2>&1; echo "preflight rc=$?"; }
    P=GGML_CUDA_FATTN_TILE_PASCAL
    ALL="${P}_LDS=1 ${P}_KDP=1 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
    arm_env() { # <arm> -> "<build> <environment>"
        case $1 in
            base-plain)        echo "base -" ;;
            head-plain)        echo "head -" ;;
            base-forced)       echo "base $P=2" ;;
            head-forced)       echo "head $P=2" ;;
            head-lds)          echo "head $P=2 ${P}_LDS=1" ;;
            head-kdp)          echo "head $P=2 ${P}_KDP=1" ;;
            head-dec)          echo "head $P=2 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4" ;;
            head-all)          echo "head $P=2 $ALL" ;;
            base-forced-fault) echo "base $P=2 ${P}_FAULT=1" ;;
            head-forced-fault) echo "head $P=2 ${P}_FAULT=1" ;;
            head-all-fault)    echo "head $P=2 $ALL ${P}_FAULT=1" ;;
            *) return 1 ;;
        esac
    }
    FA_ARMS=${FA_ARMS:-base-plain head-plain base-forced head-forced head-lds head-kdp head-dec head-all base-forced-fault head-forced-fault head-all-fault}
    printf 'arm\tbuild\tenvironment\trc\tpassed\tFAIL lines\tnot supported\tPascal kernel log lines\n' > "$K/fa.tsv"
    for arm in $FA_ARMS; do
        spec=$(arm_env "$arm") || { echo "unknown arm $arm"; continue; }
        set -- $spec
        build=$1; shift; e="$*"; [ "$e" = - ] && e=
        echo "--- $arm ($build: ${e:-nothing set}) $(date -Is)"
        docker run --rm $LBL --cpu-shares=256 --gpus "\"device=$GPU\"" \
            -v "$J:/jit" -e CUDA_CACHE_PATH=/jit -e CUDA_CACHE_MAXSIZE=4294967296 \
            -v "$BASE_RIG:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c "
            env PR910_ARM=$arm $e nice -n 19 timeout 3600 /$build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT \
                2>&1 | sed 's/\x1b\[[0-9;]*m//g' > /k/$arm.log
            echo \${PIPESTATUS[0]} > /k/$arm.rc
            chown $(id -u):$(id -g) /k/$arm.log /k/$arm.rc"
        rc=$(cat "$K/$arm.rc" 2>/dev/null || echo none)
        passed=$(grep -aoE '[0-9]+/[0-9]+ tests passed' "$K/$arm.log" | tail -n 1)
        nfail=$(grep -ac ': FAIL' "$K/$arm.log")
        nsup=$(grep -ac 'not supported' "$K/$arm.log")
        klog=$(grep -aoE 'compute-6\.1 [a-z ]+attention[^(]*' "$K/$arm.log" | sort -u | tr '\n' ';')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$build" "${e:--}" "$rc" "${passed:-none}" "$nfail" "$nsup" "${klog:--}" | tee -a "$K/fa.tsv"
    done
    echo "=== fa done $(date -Is)"
fi

finish 0
