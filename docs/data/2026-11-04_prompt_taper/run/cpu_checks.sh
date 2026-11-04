#!/usr/bin/env bash
# C218 (Q771, Q363) on a build machine, CPU only: the rig build of pascal-p106-v3-b12076-taper (a taper of prompt
# micro-batches: smaller micro-batches where a call fills or drains the layer-split pipeline, LLAMA_UBATCH_TAPER /
# --ubatch-taper / a request's "ubatch_taper", off by default, on -ladder2), CPU-only builds of it and of its base, and
# the checks that need no GPU, in containers without a GPU:
#   1. cpubuild   the branch without CUDA: llama-server, test-ubatch-taper, test-llama-archs, test-llama-context,
#                 test-batch-alloc (first, so that a compile error stops the chain in minutes)
#   2. build      the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1); test-ubatch-taper, test-llama-archs and
#                 test-llama-context from its tree (a second tarball, for a check on the rig)
#   3. basebuild  the base commit (BASE, -ladder2) without CUDA: llama-server, test-llama-archs
#   4. unit       test-ubatch-taper --unit-only: the spec parser and the plans of a call
#   5. batchalloc test-batch-alloc: the batch splitter's own tests, with the plan code in it
#   6. models     random-weight qwen4exp (Flash-Next's architecture) from test-llama-archs
#   7. taper      test-ubatch-taper on the random qwen4exp: prompts of LENS tokens in llama-server's calls, taper arms
#                 against off (logits of every position and of the last one, checkpoints, state, greedy tokens, the
#                 logged splits against the plan), with and without recurrent rollback snapshots (n_rs_seq 3)
#   8. identity   test-llama-archs -b CPU on every architecture with LLAMA_ARCHS_LOGITS: the base build writes each
#                 architecture's logits (one batch of 128 tokens, then steps of 1-8), the new build compares with the
#                 taper off (pass: bit for bit), with LLAMA_UBATCH_TAPER=short=4096:9 (every call in micro-batches of 9
#                 tokens, the smallest a plan makes; pass: each architecture within nmse 1e-8 or within the base's own
#                 nmse between its one batch and its steps) and, for information, short=4096:1,min=1 (one token at a time)
#   9. ctx        test-llama-context on the random qwen4exp (state save, load, copy, rollback), off and with short=4096:9
#  10. server     server_check.sh: llama-server, base against the new build's arms (off, env, flag, per request)
#  11. replay     a dry run of the rig's replay (taper_replay.py) against the CPU llama-server on the random qwen4exp:
#                 two of the agent check's tasks, passes off and on, one fresh prompt; that it runs and reports (the
#                 random model's speeds mean nothing)
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-taper] [REMOTE=fork] [BASE=a330672c1] \
#   [JOBS=6] [THREADS=8] [LENS=100,600,3000,9000] [LOCK=<file>] [DOCKER_ARGS=...] [STEPS='cpubuild build ...'] bash cpu_checks.sh
#   WORK    the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs REMOTE as a
#           remote with a fetch URL that resolves inside the container)
#   SRC_URL where the host fetches BRANCH from for the CPU build (default: the fork on GitHub)
#   LOCK    a file to flock around each build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra-<sha>.tar.gz, $WORK/cpu/bin/,
#   $WORK/cpu-base/bin/, $WORK/checks/<check>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when
#   $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-taper}
REMOTE=${REMOTE:-fork}
SRC_URL=${SRC_URL:-https://github.com/Cinderella-Man/llama.cpp}
BASE=${BASE:-a330672c1}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
LENS=${LENS:-100,600,3000,9000}
STEPS=${STEPS:-cpubuild build basebuild unit batchalloc models taper identity ctx server replay}
TURNS=${TURNS:-$REPO/docs/data/2026-11-03_retype_agent/agent/turns.csv}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks" "$WORK/cpu" "$WORK/cpu-base"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
# a container without a GPU: both CPU builds' binaries, the scripts here and the outputs
cpu_run() { # <check dir> <script>
    docker run --rm $D -v "$WORK/cpu/bin:/x:ro" -v "$WORK/cpu-base/bin:/xb:ro" -v "$HERE:/run:ro" -v "$1:/k" \
        -e THREADS="$THREADS" -e LENS="$LENS" "$IMAGE" bash -c "$2"'
        rc=$?; chown -R '"$(id -u):$(id -g)"' /k; exit $rc'
}
echo "=== chain started $(date -Is), steps: $STEPS"

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "$REMOTE/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-ubatch-taper test-llama-archs test-llama-context \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-ubatch-taper build/bin/test-llama-archs build/bin/test-llama-context /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
cpubuild() { # <source dir under /work> <output dir under /work> <targets>
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e S="$1" -e O="$2" -e T="$3" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S "/work/$S" -B "/work/$O/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build "/work/$O/build" -j "$JOBS" --target $T > "/work/logs/$O-build.log" 2>&1 \
            || { grep -B2 -A6 -E "error" "/work/logs/$O-build.log" | head -60; exit 1; }
        mkdir -p "/work/$O/bin"
        for t in $T; do cp "/work/$O/build/bin/$t" "/work/$O/bin/"; done
        chown -R "$HOST_UID:$HOST_GID" "/work/$O" /work/logs'
}
export -f build cpubuild
export WORK BRANCH REMOTE JOBS IMAGE D REPO

if step cpubuild; then
    echo "=== 1. cpubuild $(date -Is)"
    # the branch checked out in $WORK/src (the rig build checks out the same commit there; its build dir stays outside)
    git -C "$WORK/src" fetch -q "$SRC_URL" "$BRANCH" && git -C "$WORK/src" checkout -q --force FETCH_HEAD \
        && git -C "$WORK/src" rev-parse HEAD > "$WORK/cpu/COMMIT" || { verdict cpubuild FAIL "fetch"; date -Is > "$WORK/DONE"; exit 1; }
    locked bash -c 'cpubuild src cpu "llama-server test-ubatch-taper test-llama-archs test-llama-context test-batch-alloc"' \
        > "$WORK/logs/cpubuild.console" 2>&1
    rc=$?
    cp "$WORK/cpu/COMMIT" "$WORK/cpu/bin/COMMIT" 2>/dev/null
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(cat "$WORK/cpu/COMMIT")"
    [ $rc = 0 ] || { cat "$WORK/logs/cpubuild.console"; date -Is > "$WORK/DONE"; exit 1; }
fi

if step build; then
    echo "=== 2. build $(date -Is)"
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
    git -C "$WORK/src" rev-parse HEAD > "$WORK/COMMIT"
    verdict build OK "$(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"
fi

if step basebuild; then
    echo "=== 3. basebuild $(date -Is)"
    rm -rf "$WORK/base-src" && mkdir -p "$WORK/base-src"
    git -C "$WORK/src" archive "$BASE" | tar -x -C "$WORK/base-src"
    locked bash -c 'cpubuild base-src cpu-base "llama-server test-llama-archs"' > "$WORK/logs/basebuild.console" 2>&1
    rc=$?
    git -C "$WORK/src" rev-parse "$BASE" > "$WORK/cpu-base/bin/COMMIT"
    verdict basebuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(cat "$WORK/cpu-base/bin/COMMIT")"
    [ $rc = 0 ] || { date -Is > "$WORK/DONE"; exit 1; }
fi

if step unit; then
    echo "=== 4. unit $(date -Is)"
    mkdir -p "$WORK/checks/unit"
    cpu_run "$WORK/checks/unit" 'nice -n 19 /x/test-ubatch-taper --unit-only > /k/unit.log 2>&1'
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^unit:' "$WORK/checks/unit/unit.log")"
fi

if step batchalloc; then
    echo "=== 5. batchalloc $(date -Is)"
    mkdir -p "$WORK/checks/batchalloc"
    cpu_run "$WORK/checks/batchalloc" 'nice -n 19 /x/test-batch-alloc > /k/batchalloc.log 2>&1'
    rc=$?
    verdict batchalloc "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$WORK/checks/batchalloc/batchalloc.log")"
fi

if step models; then
    echo "=== 6. models $(date -Is)"
    mkdir -p "$WORK/checks/models"
    cpu_run "$WORK/checks/models" '/x/test-llama-archs -a "^qwen4exp\$" -s 1 -o /k > /k/save.log 2>&1; ls /k/*.gguf'
    verdict models "$(ls "$WORK"/checks/models/*qwen4exp*.gguf > /dev/null 2>&1 && echo OK || echo FAIL)" \
        "$(cd "$WORK/checks/models" && ls *.gguf 2>/dev/null | tr '\n' ' ')"
fi
Q=$(cd "$WORK/checks/models" 2>/dev/null && ls *qwen4exp*.gguf 2>/dev/null | head -1)

if step taper; then
    echo "=== 7. taper $(date -Is)"
    mkdir -p "$WORK/checks/taper"
    cp "$WORK/checks/models/$Q" "$WORK/checks/taper/"
    cpu_run "$WORK/checks/taper" 'nice -n 19 /x/test-ubatch-taper -m /k/'"$Q"' -t $THREADS --lens $LENS --tsv /k/taper.tsv \
        > /k/taper.log 2>&1; rc=$?; rm -f /k/*.gguf; exit $rc'
    rc=$?
    verdict taper "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^(unit|model|base):' "$WORK/checks/taper/taper.log" | tr '\n' ' ')"
fi

if step identity; then
    echo "=== 8. identity $(date -Is)"
    mkdir -p "$WORK/checks/identity"
    rm -f "$WORK"/checks/identity/*.f32
    cpu_run "$WORK/checks/identity" '
        run() { # <name> <bin dir> <env...>
            n=$1 b=$2; shift 2
            env "$@" LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $b/test-llama-archs -b CPU -s 1 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > /k/$n.log
            echo "$n rc=${PIPESTATUS[0]}"
        }
        run base  /xb
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run off   /x
        run taper  /x LLAMA_UBATCH_TAPER=short=4096:9
        run taper1 /x LLAMA_UBATCH_TAPER=short=4096:1,min=1
        rm -f /k/*.f32' > "$WORK/checks/identity/rc.txt" 2>&1
    # per architecture and run: "<arch>: logits against <file>: nmse X, max abs diff Y, N of M differ". test-llama-archs
    # builds a second model for some architectures (llama, granite, ...), which the base run already compares against the
    # first one's file (a large nmse) and every run repeats: a line is that second model's when its architecture has such
    # a line in the base run and its nmse is the same to the printed digits (base_self = 1)
    printf 'run\tarch\tnmse\tmax_abs_diff\tdiffer\tof\tbase_self\n' > "$WORK/checks/identity/identity.tsv"
    grep 'logits against' "$WORK/checks/identity/base.log" > "$WORK/checks/identity/base_self.txt"
    for n in base off taper taper1; do
        grep 'logits against' "$WORK/checks/identity/$n.log" | awk -v r=$n '
            function parse(line, f) {
                if (!match(line, /^[a-z0-9_.-]+: logits against [^:]*: nmse [^,]*, max abs diff [^,]*, [0-9]+ of [0-9]+ differ/)) {
                    return 0
                }
                split(line, f, /: logits against [^:]*: nmse |, max abs diff |, | of | differ/)
                return 1
            }
            NR == FNR { if (parse($0, g)) s[g[1] " " g[2]] = 1; next }
            { if (parse($0, f)) printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\n", r, f[1], f[2], f[3], f[4], f[5], ((f[1] " " f[2]) in s) }' \
            "$WORK/checks/identity/base_self.txt" - >> "$WORK/checks/identity/identity.tsv"
    done
    # the base's own split sensitivity per architecture: "<arch>: one batch against steps of 1-8 tokens: nmse X" (the line
    # follows the table's row prefix, so the architecture is the text after the last "|")
    grep 'one batch against steps' "$WORK/checks/identity/base.log" \
        | sed -e 's/: one batch against steps of 1-8 tokens: nmse /\t/' -e 's/^.*|//' > "$WORK/checks/identity/base_steps.tsv"
    nb=$(cat "$WORK/checks/identity/n_base.txt" 2>/dev/null || echo 0)
    ns=$(wc -l < "$WORK/checks/identity/base_self.txt")
    # per run: architectures compared, second-model lines, any value different, nmse over 1e-8, nmse above the base's own
    # one-batch-against-steps nmse (and above 1e-8)
    awk -F'\t' 'NR == FNR { st[$1] = $2 + 0; next }
        FNR > 1 && $1 != "base" {
            r = $1; if ($7) { self[r]++; next }
            n[r]++; if ($5 > 0) d[r]++; if ($3 + 0 > 1e-8) o[r]++
            if ($3 + 0 > 1e-8 && (!($2 in st) || $3 + 0 > st[$2])) { x[r]++; xa[r] = xa[r] " " $2 ":" $3 "/" ($2 in st ? st[$2] : "none") }
            if ($3 + 0 > m[r]) { m[r] = $3 + 0; ma[r] = $2 }
        }
        END { for (r in n) printf "%s\t%d\t%d\t%d\t%d\t%d\t%.3e\t%s\t%s\n", r, n[r], self[r], d[r], o[r], x[r], m[r], ma[r], xa[r] }' \
        "$WORK/checks/identity/base_steps.tsv" "$WORK/checks/identity/identity.tsv" | sort > "$WORK/checks/identity/runs.tsv"
    ok=1
    # off: every architecture the base wrote, bit for bit
    awk -F'\t' -v nb="$nb" -v ns="$ns" '$1 == "off" && $2 == nb && nb > 0 && $3 == ns && $4 == 0 { f = 1 } END { exit !f }' \
        "$WORK/checks/identity/runs.tsv" || ok=0
    # short=4096:9: every architecture within the base's own split tolerance (its one batch against steps of 1-8 tokens)
    awk -F'\t' -v nb="$nb" -v ns="$ns" '$1 == "taper" && $2 == nb && $3 == ns && $6 == 0 { f = 1 } END { exit !f }' \
        "$WORK/checks/identity/runs.tsv" || ok=0
    d=$(awk -F'\t' '{ printf "%s: %d archs (+%d second-model lines), %d with values different, %d over nmse 1e-8, %d above the base'"'"'s own steps nmse%s, largest %s (%s); ", $1, $2, $3, $4, $5, $6, ($6 ? " (" substr($9, 2) ")" : ""), $7, $8 }' \
        "$WORK/checks/identity/runs.tsv")
    verdict identity "$([ $ok = 1 ] && echo OK || echo FAIL)" "base wrote $nb archs; $d$(grep ' rc=' "$WORK/checks/identity/rc.txt" | tr '\n' ' ')"
fi


if step ctx; then
    echo "=== 9. ctx $(date -Is)"
    mkdir -p "$WORK/checks/ctx"
    cp "$WORK/checks/models/$Q" "$WORK/checks/ctx/"
    cpu_run "$WORK/checks/ctx" '
        fail=0
        for s in default on; do
            if [ "$s" = default ]; then e=""; else e="LLAMA_UBATCH_TAPER=short=4096:9"; fi
            env $e nice -n 19 /x/test-llama-context -m /k/'"$Q"' -t "$THREADS" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > "/k/context-$s.log"
            rc=${PIPESTATUS[0]}
            echo "$s: exit $rc, $(grep -E "passed   :|failed   :|skipped  :" /k/context-$s.log | tr -s " " | tr "\n" " ")"
            [ $rc = 0 ] || fail=1
            grep -q "failed   : 0" "/k/context-$s.log" || fail=1
        done
        rm -f /k/*.gguf /k/dump_state.*
        exit $fail' > "$WORK/checks/ctx/console.txt" 2>&1
    rc=$?
    verdict ctx "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tr '\n' ' ' < "$WORK/checks/ctx/console.txt")"
fi

if step server; then
    echo "=== 10. server $(date -Is)"
    mkdir -p "$WORK/checks/server"
    cp "$WORK/checks/models/$Q" "$WORK/checks/server/"
    cpu_run "$WORK/checks/server" '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        BASE_BIN=/xb BIN=/x MODEL=/k/'"$Q"' OUT=/k/out THREADS=$THREADS LENS="${LENS//,/ }" nice -n 19 bash /run/server_check.sh
        rc=$?; rm -f /k/*.gguf; exit $rc' > "$WORK/checks/server/console.txt" 2>&1
    rc=$?
    tail -n 12 "$WORK/checks/server/console.txt"
    verdict server "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tr '\n' ' ' < "$WORK/checks/server/out/verdict.txt" 2>/dev/null)"
fi

if step replay; then
    echo "=== 11. replay $(date -Is)"
    mkdir -p "$WORK/checks/replay/out"
    cp "$WORK/checks/models/$Q" "$TURNS" "$WORK/checks/replay/"
    cpu_run "$WORK/checks/replay" '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl python3 > /dev/null
        /x/llama-server -m /k/'"$Q"' -ngl 0 -t "$THREADS" -c 16384 -b 2048 -ub 512 -np 1 --ctx-checkpoints 4 --cache-ram 0 \
            --no-webui -lv 5 --override-kv qwen4exp.context_length=int:16384 --host 127.0.0.1 --port 18094 > /k/server.log 2>&1 &
        pid=$!
        for _ in $(seq 1 120); do curl -s http://127.0.0.1:18094/health | grep -q "\"ok\"" && break; sleep 1; done
        nice -n 19 python3 -I /run/taper_replay.py --port 18094 --turns /k/turns.csv --passes "off on" --fresh 1 --fresh-n 600 \
            --tasks bugfix-javascript-grade-school,bugfix-python-affine-cipher --label dry --out /k/out > /k/replay.out 2>&1
        rc=$?
        kill $pid; wait $pid 2>/dev/null
        grep -o "ubatch taper: .*" /k/server.log > /k/taper_lines.txt
        rm -f /k/*.gguf; exit $rc' > "$WORK/checks/replay/console.txt" 2>&1
    rc=$?
    nr=$(($(wc -l < "$WORK/checks/replay/out/requests.tsv" 2>/dev/null || echo 1) - 1))
    nt=$(wc -l < "$WORK/checks/replay/taper_lines.txt" 2>/dev/null || echo 0)
    verdict replay "$([ $rc = 0 ] && [ "$nr" -gt 0 ] && [ "$nt" -gt 0 ] && echo OK || echo FAIL)" \
        "rc $rc; $nr requests, $nt taper lines; $(tail -n 4 "$WORK/checks/replay/out/summary.txt" 2>/dev/null | tr '\n' ' ')"
fi

cat "$WORK/md5.txt" 2>/dev/null
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
