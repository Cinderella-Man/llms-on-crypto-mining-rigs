#!/usr/bin/env bash
# C151 (Q701) on a build machine, CPU only: the rig build of pascal-p106-v3-b12076-vgraphs (upstream llama.cpp #31156,
# per-size graph arenas and schedulers, and #31274, scheduler scratch sized per graph and calloc'd graph metadata, on
# -ladder2) with the extra test binaries from the same tree, a CPU-only build of the same tree, and the CPU checks, in
# containers without a GPU:
#   1. build      the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1) and test-llama-archs, test-llama-context and
#                 test-draft-mem from its build tree (they go to the rig in a second tarball)
#   2. cpubuild   the same tree without CUDA: llama-server and the tests below
#   3. ops        test-backend-ops -b CPU (every op of the CPU backend against itself: the build and the ops run)
#   4. sched      test-opt and test-alloc, which drive ggml's scheduler (#31274 changes how it sizes its scratch and
#                 where it keeps the optimizer's allocation dependencies)
#   5. models     random-weight qwen4exp (Flash-Next's architecture), qwen35 and mamba models from test-llama-archs
#   6. ctx        test-llama-context on each model, every case, at every switch setting: default; LLAMA_GRAPH_PER_SIZE=0;
#                 LLAMA_SCHED_PER_SIZE=1; LLAMA_GRAPH_META_CALLOC=0
#   7. archs      test-llama-archs -b CPU, every architecture, at the default and with LLAMA_SCHED_PER_SIZE=1
#   8. spec       spec_check.sh: llama-server's speculative decoding on the random qwen4exp, its verify widths 1-4, with
#                 every switch: the same tokens and top probabilities in every arm
#   9. ckpt       C22's llama-server checkpoint arms (docs/data/2026-11-04_ckpt_cards/run/server_check.sh) at the defaults
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-vgraphs] [JOBS=6] [THREADS=8] [LOCK=<file>] \
#   [DOCKER_ARGS=...] [STEPS='build cpubuild sched models ctx archs spec ckpt ops'] bash cpu_checks.sh
#   WORK   the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs the fork as remote
#          "fork", with an https fetch URL)
#   LOCK   a file to flock around each build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra/ (+ extra-<sha>.tar.gz),
#   $WORK/cpu/bin/, $WORK/checks/<check>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when
#   $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-vgraphs}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
STEPS=${STEPS:-build cpubuild sched models ctx archs spec ckpt ops}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks" "$WORK/cpu"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
SETTINGS="default LLAMA_GRAPH_PER_SIZE=0 LLAMA_SCHED_PER_SIZE=1 LLAMA_GRAPH_META_CALLOC=0"
# a container without a GPU: the CPU build's binaries, the scripts here and the outputs
cpu_run() { # <check dir> <script>
    docker run --rm $D -v "$WORK/cpu/bin:/x:ro" -v "$HERE:/run:ro" -v "$REPO/docs/data/2026-11-04_ckpt_cards/run:/ckrun:ro" \
        -v "$1:/k" -e THREADS="$THREADS" -e SETTINGS="$SETTINGS" "$IMAGE" bash -c "$2"'
        rc=$?; chown -R '"$(id -u):$(id -g)"' /k; exit $rc'
}
echo "=== chain started $(date -Is), steps: $STEPS"

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
    # the test binaries the rig build leaves out, from the same build tree
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs test-llama-context test-draft-mem \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-llama-archs build/bin/test-llama-context build/bin/test-draft-mem /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
cpubuild() {
    # the same checkout as the rig build (it leaves $WORK/src at the branch), built outside the tree without CUDA
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        git -C /work/src log -1 --format="cpu build of %H (%ci)"
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target llama-server test-backend-ops test-opt test-alloc \
            test-llama-archs test-llama-context test-draft-mem > /work/logs/cpubuild.log 2>&1 \
            || { tail -40 /work/logs/cpubuild.log; exit 1; }
        mkdir -p /work/cpu/bin
        cp /work/cpu/build/bin/llama-server /work/cpu/build/bin/test-* /work/cpu/bin/
        git -C /work/src rev-parse HEAD > /work/cpu/bin/COMMIT
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build cpubuild
export WORK BRANCH JOBS IMAGE D REPO

if step build; then
    echo "=== 1. build $(date -Is)"
    locked bash -c build
    rc=$?
    echo "=== build ended $(date -Is) rc=$rc"
    if [ $rc != 0 ]; then verdict build FAIL "see logs/build.log, logs/extra.log"; date -Is > "$WORK/DONE"; exit 1; fi
    t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
    T=$WORK/$(basename "$t" .tar.gz)
    sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
    (cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
    (cd "$WORK" && tar czf "extra-$sha.tar.gz" extra && md5sum "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
    cp "$WORK"/extra/test-* "$T/bin/"
    verdict build OK "$(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"
fi

if step cpubuild; then
    echo "=== 2. cpubuild $(date -Is)"
    locked bash -c cpubuild > "$WORK/logs/cpubuild.console" 2>&1
    rc=$?
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -m 1 "cpu build of" "$WORK/logs/cpubuild.console")"
    [ $rc = 0 ] || { date -Is > "$WORK/DONE"; exit 1; }
fi

if step ops; then
    echo "=== 3. ops $(date -Is)"
    mkdir -p "$WORK/checks/ops"
    cpu_run "$WORK/checks/ops" 'nice -n 19 /x/test-backend-ops -b CPU > /k/test-backend-ops.log 2>&1'
    rc=$?
    verdict ops "$([ $rc = 0 ] && echo OK || echo FAIL)" \
        "rc $rc; $(grep -E 'tests passed|backends passed' "$WORK/checks/ops/test-backend-ops.log" | tr -s ' ' | tr '\n' ';')"
fi

if step sched; then
    echo "=== 4. sched $(date -Is)"
    mkdir -p "$WORK/checks/sched"
    cpu_run "$WORK/checks/sched" 'cd /k && for t in test-opt test-alloc; do nice -n 19 /x/$t > /k/$t.log 2>&1; echo "$t rc=$?"; done > /k/rc.txt'
    nok=$(grep -c 'rc=0' "$WORK/checks/sched/rc.txt")
    verdict sched "$([ "$nok" = 2 ] && echo OK || echo FAIL)" "$(tr '\n' ' ' < "$WORK/checks/sched/rc.txt"); test-opt: $(grep -E 'tests passed|passed' "$WORK/checks/sched/test-opt.log" | tail -n 1)"
fi

if step models; then
    echo "=== 5. models $(date -Is)"
    mkdir -p "$WORK/checks/models"
    cpu_run "$WORK/checks/models" 'for a in qwen4exp qwen35 mamba; do /x/test-llama-archs -a "^$a\$" -s 1 -o /k > /k/save-$a.log 2>&1; echo "$a: exit $?"; done; ls /k/*.gguf'
    verdict models "$(ls "$WORK"/checks/models/*.gguf > /dev/null 2>&1 && echo OK || echo FAIL)" "$(cd "$WORK/checks/models" && ls *.gguf | tr '\n' ' ')"
fi

if step ctx; then
    echo "=== 6. ctx $(date -Is)"
    mkdir -p "$WORK/checks/ctx"
    cp "$WORK"/checks/models/*.gguf "$WORK/checks/ctx/"
    cpu_run "$WORK/checks/ctx" '
        cd /k
        fail=0
        for s in $SETTINGS; do
            for m in /k/*.gguf; do
                n=$(basename "$m" .gguf)
                if [ "$s" = default ]; then e=""; else e="$s"; fi
                env $e nice -n 19 /x/test-llama-context -m "$m" -t "$THREADS" > "/k/context-$s-$n.log" 2>&1
                rc=$?
                echo "$s $n: exit $rc"
                [ $rc = 0 ] || fail=1
                grep -q "failed   : 0" "/k/context-$s-$n.log" || fail=1
            done
        done
        rm -f /k/*.gguf /k/dump_state.*
        exit $fail' > "$WORK/checks/ctx/console.txt" 2>&1
    rc=$?
    # per case: the runner prints "<case>  OK|SKIP|FAIL  [reason]" (the [rs] cases twice, the second time against a
    # pre-filled cache), and another thread's log line can end up in front of it, so the case is found anywhere in the
    # line; per run: the runner's own totals
    printf 'setting\tmodel\tcase\tverdict\treason\n' > "$WORK/checks/ctx/context.tsv"
    printf 'setting\tmodel\tpassed\tfailed\tskipped\n' > "$WORK/checks/ctx/context_runs.tsv"
    for s in $SETTINGS; do
        for f in "$WORK"/checks/ctx/context-"$s"-*.log; do
            n=$(basename "$f" .log); n=${n#context-"$s"-}
            sed 's/\x1b\[[0-9;]*m//g' "$f" | awk -v s="$s" -v m="$n" '
                { for (i = 2; i <= NF; i++) if ($i ~ /^(OK|SKIP|FAIL)$/ && $(i - 1) ~ /^[a-z][a-z0-9_]*$/) {
                    r = ""; for (j = i + 1; j <= NF; j++) r = r (j > i + 1 ? " " : "") $j
                    print s "\t" m "\t" $(i - 1) "\t" $i "\t" r; break } }' >> "$WORK/checks/ctx/context.tsv"
            printf '%s\t%s\t%s\t%s\t%s\n' "$s" "$n" "$(awk '/passed   :/ { print $3; exit }' "$f")" \
                "$(awk '/failed   :/ { print $3; exit }' "$f")" "$(awk '/skipped  :/ { print $3; exit }' "$f")" \
                >> "$WORK/checks/ctx/context_runs.tsv"
        done
    done
    tally=$(awk -F'\t' 'NR > 1 { p[$1] += $3; f[$1] += $4; k[$1] += $5 } END { for (s in p) printf "%s: %d passed, %d failed, %d skipped; ", s, p[s], f[s], k[s] }' "$WORK/checks/ctx/context_runs.tsv")
    verdict ctx "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $tally"
fi

if step archs; then
    echo "=== 7. archs $(date -Is)"
    mkdir -p "$WORK/checks/archs"
    cpu_run "$WORK/checks/archs" '
        nice -n 19 /x/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/default.log; echo "default rc=${PIPESTATUS[0]}"
        LLAMA_SCHED_PER_SIZE=1 nice -n 19 /x/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/sps1.log; echo "sps1 rc=${PIPESTATUS[0]}"' \
        > "$WORK/checks/archs/rc.txt" 2>&1
    d=""
    for s in default sps1; do
        d="$d$s: $(grep "^$s rc" "$WORK/checks/archs/rc.txt"), $(grep -m 1 '^Summary' "$WORK/checks/archs/$s.log"), rows with a FAIL: $(grep -c '^|.*FAIL' "$WORK/checks/archs/$s.log"); "
    done
    ok=$(grep -c ' rc=0' "$WORK/checks/archs/rc.txt")
    nf=$(cat "$WORK"/checks/archs/{default,sps1}.log | grep -c '^|.*FAIL')
    verdict archs "$([ "$ok" = 2 ] && [ "$nf" = 0 ] && echo OK || echo REVIEW)" "$d"
fi

if step spec; then
    echo "=== 8. spec $(date -Is)"
    mkdir -p "$WORK/checks/spec"
    q=$(cd "$WORK/checks/models" && ls *qwen4exp*.gguf | head -1)
    cp "$WORK/checks/models/$q" "$WORK/checks/spec/"
    cpu_run "$WORK/checks/spec" '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        BIN=/x MODEL=/k/'"$q"' OUT=/k/out THREADS=$THREADS nice -n 19 bash /run/spec_check.sh
        rc=$?; rm -f /k/*.gguf; exit $rc' > "$WORK/checks/spec/console.txt" 2>&1
    rc=$?
    tail -n 12 "$WORK/checks/spec/console.txt"
    verdict spec "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tr '\n' ' ' < "$WORK/checks/spec/out/verdict.txt" 2>/dev/null)"
fi

if step ckpt; then
    echo "=== 9. ckpt $(date -Is)"
    mkdir -p "$WORK/checks/ckpt"
    q=$(cd "$WORK/checks/models" && ls *qwen4exp*.gguf | head -1)
    cp "$WORK/checks/models/$q" "$WORK/checks/ckpt/"
    cpu_run "$WORK/checks/ckpt" '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        BIN=/x MODEL=/k/'"$q"' OUT=/k/server THREADS=$THREADS SARGS="--override-kv qwen4exp.context_length=int:16384" \
            nice -n 19 bash /ckrun/server_check.sh
        rc=$?; rm -f /k/*.gguf; exit $rc' > "$WORK/checks/ckpt/console.txt" 2>&1
    rc=$?
    verdict ckpt "$(grep -qx PASS "$WORK/checks/ckpt/server/verdict.txt" 2>/dev/null && echo OK || echo FAIL)" \
        "rc $rc; $(tr '\n' ' ' < "$WORK/checks/ckpt/server/verdict.txt" 2>/dev/null)"
fi

cat "$WORK/md5.txt" 2>/dev/null
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
