#!/usr/bin/env bash
# C701 (Q1238): the CPU check of LLAMA_UBATCH_TRACE on a build machine, CPU only, every run in a container without
# network. Two trees, built one after the other at the same paths ($WORK/cpu/tree/{src,build}), so ccache hits:
#   (ccache: $WORK/ccache, the one scripts/build_llamacpp_docker.sh fills)
#   build   a static CPU build (GGML_CUDA=OFF, GGML_RPC=ON) of TIP (the branch with the switch) and BASE (the branch
#           it was cut from); the binaries are kept in $WORK/cpu/<tag>/bin
#   model   a random-weight qwen4exp model (Flash-Next's architecture: hybrid memory, MoE) written by TIP's
#           test-llama-archs, with the tokenizer of the tree's models/ggml-vocab-qwen35.gguf
#   cpu     llama-completion on the CPU backend, one 4-call prompt (-b 256 -ub 64) and 16 greedy tokens: BASE, TIP with
#           the switch unset, TIP with LLAMA_UBATCH_TRACE=1. The printed text must be the same bytes in the three runs,
#           the unset run must log no trace line and the traced run must log every kind the CPU can show
#   rpc     the same three runs with the model's layers on a ggml-rpc-server on the same container's loopback (a second
#           backend on the CPU): several splits per evaluation, input copies between backends, a card0, re-plans
#   server  llama-server of TIP, the switch unset and at 1: one /completion request, the same text in both, the phase
#           averages only when on
#   table   trace_table.py (beside this script) on the traced rpc run: $WORK/cpu/table_rpc/
# Usage: WORK=<dir> SRC=<llama.cpp clone holding both commits> TIP=<commit> BASE=<commit> [JOBS=8] [THREADS=4]
#   [PORT=18301] [LOCK=<flock file>] [DOCKER_ARGS=...] [BASE_IMAGE=ubuntu:24.04] [IMAGE=pq-c565-cpu:24.04] bash cpu_check.sh
# Output: $WORK/cpu/<tag>/ (tree.txt, cmake.log, build.log, bin/), $WORK/cpu/runs/<arm>.{out,err}, $WORK/cpu/counts.tsv
#   (arm, trace kind, lines), $WORK/cpu/summary.tsv (stage, check, verdict, detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${TIP:?TIP=<commit>}" "${BASE:?BASE=<commit>}"
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
PORT=${PORT:-18301}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
HERE=$(dirname "$(realpath "$0")")
C=$WORK/cpu
T=$C/tree
B=$T/build
R=$C/runs
D="--rm --label pascal-quants=1 --cpu-shares=256 --network none ${DOCKER_ARGS:-}"
KINDS="decode-trace ubatch-trace sched-trace sched-replan galloc-trigger galloc-grow"
mkdir -p "$C" "$T" "$R" "$C/models" "$WORK/ccache"
rm -f "$C/DONE"
printf 'stage\tcheck\tverdict\tdetail\n' > "$C/summary.tsv"
printf 'arm\tkind\tlines\n' > "$C/counts.tsv"
verdict() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$C/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
# shellcheck disable=SC2086
box()     { docker run $D -v "$WORK:$WORK" "$IMAGE" "$@"; }

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        "$BASE_IMAGE" | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$C/image.log" 2>&1 || { echo "image build failed: $C/image.log"; exit 1; }
fi
echo "=== c701 cpu check started $(date -Is), TIP ${TIP:0:9}, BASE ${BASE:0:9}"

# <tag> <commit> <targets>: the tree, its build, its binaries kept
build() {
    local tag=$1 commit=$2 targets=$3 O=$C/$1 rc=0
    mkdir -p "$O"
    rm -rf "$T/src" "$B" "$O/bin" && mkdir -p "$T/src" "$O/bin"
    git -C "$SRC" archive "$commit" | tar -x -C "$T/src" || rc=$?
    echo "commit $(git -C "$SRC" rev-parse "$commit") $(git -C "$SRC" log -1 --format=%s "$commit")" > "$O/tree.txt"
    verdict "$tag" tree "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(cut -c1-120 "$O/tree.txt")"
    [ $rc = 0 ] || return 1
    # shellcheck disable=SC2086
    locked docker run $D -v "$WORK:$WORK" -e CCACHE_DIR="$WORK/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$B"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_RPC=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$O/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$B"' -j '"$JOBS"' --target '"$targets"' > '"$O/build.log"' 2>&1'
    rc=$?
    verdict "$tag" build "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(grep -ac 'warning:' "$O/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$O/build.log") error lines"
    for t in $targets; do [ -x "$B/bin/$t" ] && cp "$B/bin/$t" "$O/bin/"; done
    [ "$tag" = tip ] && cp "$T/src/models/ggml-vocab-qwen35.gguf" "$C/models/vocab.gguf"
    return $rc
}

build tip "$TIP" "llama-completion llama-server ggml-rpc-server test-llama-archs" || { date -Is > "$C/DONE"; exit 1; }
build base "$BASE" "llama-completion ggml-rpc-server"

# the model: qwen4exp's MoE variant if the test writes one, else its dense one
box bash -c "LLAMA_ARCHS_VOCAB=$C/models/vocab.gguf LLAMA_ARCHS_N_CTX=4096 nice -n 19 $C/tip/bin/test-llama-archs \
    -a '^qwen4exp\$' -s 1 -d 0.1 -o $C/models" > "$C/models/archs.log" 2>&1
M=$(ls "$C"/models/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/qwen4exp-*.gguf 2>/dev/null | head -n 1)
verdict model write "$([ -n "$M" ] && echo OK || echo FAIL)" "${M:-none}; $(ls -l "$C"/models/*.gguf 2>/dev/null | awk '{print $NF": "$5" bytes"}' | tr '\n' ' ')"
[ -n "$M" ] || { date -Is > "$C/DONE"; exit 1; }

# a prompt of about 300 tokens: with -b 256 -ub 64 two decode calls of four and one micro-batches
P=$(for i in $(seq 1 30); do printf 'The cards on a mining rig wait for the host, then compute the layers they hold. '; done)
COMMON="-m $M -n 16 --temp 0 -s 1 -t $THREADS -c 1024 -b 256 -ub 64 -no-cnv --no-warmup --no-display-prompt"

# <arm> <tag> <trace: - or level> [rpc]: one llama-completion run, its text and its log
run() {
    local arm=$1 tag=$2 lvl=$3 rpc=${4:-} env="" pre="" post="; exit \$rc"
    [ "$lvl" != - ] && env="LLAMA_UBATCH_TRACE=$lvl"
    if [ -n "$rpc" ]; then
        pre="$C/$tag/bin/ggml-rpc-server -H 127.0.0.1 -p $PORT -t $THREADS > $R/$arm.rpc.log 2>&1 & S=\$!; sleep 3;"
        post="; kill \$S; wait \$S 2> /dev/null; exit \$rc"
        rpc="--rpc 127.0.0.1:$PORT -ngl 99"
    fi
    box bash -c "$pre env $env nice -n 19 $C/$tag/bin/llama-completion $COMMON $rpc -p \"$P\" > $R/$arm.out 2> $R/$arm.err; rc=\$?$post"
    local rc=$?
    for k in $KINDS; do printf '%s\t%s\t%s\n' "$arm" "$k" "$(grep -ac "$k: " "$R/$arm.err")" >> "$C/counts.tsv"; done
    printf '%s\t%s\t%s\n' "$arm" "rc" "$rc" >> "$C/counts.tsv"
    return $rc
}
n()   { awk -F'\t' -v a="$1" -v k="$2" '$1 == a && $2 == k {print $3}' "$C/counts.tsv"; }
all() { local s=0; for k in $KINDS; do s=$((s + $(n "$1" "$k"))); done; echo $s; }

for mode in cpu rpc; do
    flag=""; [ $mode = rpc ] && flag=rpc
    run $mode-base base - $flag; run $mode-off tip - $flag; run $mode-on tip 1 $flag
    same=OK
    cmp -s "$R/$mode-base.out" "$R/$mode-off.out" || same=FAIL
    cmp -s "$R/$mode-off.out" "$R/$mode-on.out" || same=FAIL
    [ -s "$R/$mode-base.out" ] || same=FAIL
    verdict $mode text "$same" "base, off, on: $(md5sum < "$R/$mode-base.out" | cut -c1-8) $(md5sum < "$R/$mode-off.out" | cut -c1-8) $(md5sum < "$R/$mode-on.out" | cut -c1-8), $(wc -c < "$R/$mode-on.out") bytes; rc $(n $mode-base rc) $(n $mode-off rc) $(n $mode-on rc)"
    verdict $mode off-silent "$([ "$(all $mode-off)" = 0 ] && [ "$(all $mode-base)" = 0 ] && echo OK || echo FAIL)" "trace lines: off $(all $mode-off), base $(all $mode-base)"
    detail=""; for k in $KINDS; do detail="$detail$k $(n $mode-on "$k"), "; done
    ok=OK
    for k in decode-trace ubatch-trace sched-trace; do [ "$(n $mode-on $k)" -gt 0 ] || ok=FAIL; done
    verdict $mode on-lines "$ok" "${detail%, }"
done

# the rpc arm: the first prompt micro-batch's lines, and how many evaluations had several splits and a card0
grep -a -m 1 -B 12 'ubatch-trace: dec n_tokens = 64' "$R/rpc-on.err" | grep -aE 'trace|replan|galloc' > "$C/rpc_first_ubatch.txt"
verdict rpc splits "$(grep -a 'sched-trace: ' "$R/rpc-on.err" | grep -avc 'splits = 1 ' | awk '{print ($1 > 0) ? "OK" : "FAIL"}')" \
    "$(grep -a 'sched-trace: ' "$R/rpc-on.err" | grep -avc 'splits = 1 ') of $(grep -ac 'sched-trace: ' "$R/rpc-on.err") evaluations with 2+ splits; $(grep -a 'ubatch-trace: .* card0 = ' "$R/rpc-on.err" | grep -avc 'card0 = -1 ') of $(grep -ac 'ubatch-trace: .* card0 = ' "$R/rpc-on.err") micro-batches with a card0"

# the server: one request with the switch unset and at 1
srv() {
    local arm=$1 lvl=$2 env=""
    [ "$lvl" != - ] && env="LLAMA_UBATCH_TRACE=$lvl"
    box bash -c "env $env nice -n 19 $C/tip/bin/llama-server -m $M -t $THREADS -c 1024 -b 256 -ub 64 --host 127.0.0.1 --port $PORT \
            --no-warmup > $R/$arm.err 2>&1 & S=\$!
        python3 - <<'PY' > $R/$arm.out
import json, time, urllib.request
base = 'http://127.0.0.1:$PORT'
for _ in range(300):
    try:
        if urllib.request.urlopen(base + '/health', timeout=2).status == 200: break
    except Exception: time.sleep(1)
req = {'prompt': '''$P''', 'n_predict': 16, 'temperature': 0, 'seed': 1, 'cache_prompt': False}
r = urllib.request.urlopen(urllib.request.Request(base + '/completion', json.dumps(req).encode(), {'Content-Type': 'application/json'}), timeout=600)
print(json.loads(r.read())['content'])
PY
        rc=\$?; sleep 6; kill \$S; wait \$S 2> /dev/null; exit \$rc"
    for k in $KINDS "avg t_decode"; do printf '%s\t%s\t%s\n' "$arm" "$k" "$(grep -ac "$k" "$R/$arm.err")" >> "$C/counts.tsv"; done
}
srv server-off -; srv server-on 1
verdict server text "$(cmp -s "$R/server-off.out" "$R/server-on.out" && [ -s "$R/server-off.out" ] && echo OK || echo FAIL)" "off, on: $(md5sum < "$R/server-off.out" | cut -c1-8) $(md5sum < "$R/server-on.out" | cut -c1-8)"
verdict server lines "$([ "$(all server-off)" = 0 ] && [ "$(n server-off 'avg t_decode')" = 0 ] && [ "$(n server-on ubatch-trace)" -gt 0 ] && [ "$(n server-on 'avg t_decode')" -gt 0 ] && echo OK || echo FAIL)" \
    "off: trace $(all server-off), phase $(n server-off 'avg t_decode'); on: ubatch $(n server-on ubatch-trace), sched $(n server-on sched-trace), phase $(n server-on 'avg t_decode')"

# shellcheck disable=SC2086
mkdir -p "$C/table_rpc" && docker run $D -v "$WORK:$WORK" -v "$HERE:$HERE:ro" "$IMAGE" \
    python3 -I "$HERE/trace_table.py" "$R/rpc-on.err" "$C/table_rpc" > "$C/table_rpc/console.txt" 2>&1
verdict table rpc "$([ $? = 0 ] && [ -s "$C/table_rpc/summary.txt" ] && echo OK || echo FAIL)" "$(head -n 1 "$C/table_rpc/summary.txt" 2>/dev/null | cut -c1-200)"

echo "=== c701 cpu check done $(date -Is)"
date -Is > "$C/DONE"
