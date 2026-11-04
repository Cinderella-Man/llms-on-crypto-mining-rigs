#!/usr/bin/env bash
# C200/C201 (Q750, Q231, Q751, Q755) on a build machine, CPU only: the rig build of pascal-p106-v3-b12076-hops (the
# residual in F16, BF16 or Q8_0 across card boundaries, LLAMA_HOP_CAST; a staged copy through pinned host slots,
# GGML_CUDA_HOP_STAGED; both off by default, on -ladder2), the extra test binaries from its tree, CPU-only builds of it
# and of its base, the extended hop microbenchmark, and the checks that need no GPU, in containers without a GPU:
#   1. build      the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1); test-hop-cast, test-llama-archs and
#                 test-llama-context from its build tree (a second tarball, for the GPU checks)
#   2. cpubuild   the same tree without CUDA: llama-server, test-backend-ops, test-hop-cast, test-llama-archs,
#                 test-llama-context
#   3. basebuild  the base commit (BASE, -ladder2) without CUDA: llama-server, test-llama-archs
#   4. bench      bench/megakernel_p106/build.sh (the hop microbenchmark with the manual staged hop), and the rig bundle:
#                 the binary, rig_hops.sh, fn_line.sh and fn_requests.py (docs/data/2026-11-03_flash_next_baseline/run)
#   5. ops        test-backend-ops -b CPU -o CPY: every copy case, the new ones at the hop shapes among them
#   6. models     random-weight qwen4exp (Flash-Next's architecture) from test-llama-archs
#   7. unit       test-hop-cast --unit-only: the round trips F32 -> F16 / BF16 / Q8_0 -> F32 on the CPU
#   8. model      test-hop-cast on the random qwen4exp, arms off, staged, f16 / q8_0 without a boundary (exact), and the
#                 round trip before layer 1 in f16, bf16 and q8_0 (within bound, layer 0 unchanged)
#   9. identity   test-llama-archs -b CPU on every architecture with LLAMA_ARCHS_LOGITS: the base build writes each
#                 architecture's logits, then the new build compares with the switches off, with GGML_CUDA_HOP_STAGED=1
#                 LLAMA_HOP_CAST=q8_0 (no boundary on the CPU: no change), and with LLAMA_HOP_CAST=q8_0
#                 LLAMA_HOP_CAST_AT=1 (only qwen4exp may change)
#  10. ctx        test-llama-context on the random qwen4exp at the default and with the q8_0 round trip before layer 1
#  11. server     server_check.sh: llama-server's speculative decoding, base against the new build's arms
#  12. sm86       the same tree for compute capability 8.6: llama-perplexity and the tests, for the GPU check and the
#                 KLD job on RTX 30-series cards (not in the default steps)
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-hops] [REMOTE=fork] [BASE=a330672c1] \
#   [JOBS=6] [THREADS=8] [LOCK=<file>] [DOCKER_ARGS=...] [STEPS='build cpubuild ...'] bash cpu_checks.sh
#   WORK    the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs REMOTE as a
#           remote with a fetch URL that resolves inside the container)
#   LOCK    a file to flock around each build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra-<sha>.tar.gz, $WORK/cpu/bin/,
#   $WORK/cpu-base/bin/, $WORK/bundle/ (the bench), $WORK/checks/<check>/, $WORK/logs/, $WORK/summary.tsv (check,
#   verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-hops}
REMOTE=${REMOTE:-fork}
BASE=${BASE:-a330672c1}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
STEPS=${STEPS:-build cpubuild basebuild bench ops models unit model identity ctx server}
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
        -e THREADS="$THREADS" "$IMAGE" bash -c "$2"'
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
        nice -n 19 cmake --build build -j "$JOBS" --target test-hop-cast test-llama-archs test-llama-context \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-hop-cast build/bin/test-llama-archs build/bin/test-llama-context /work/extra/
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
            || { tail -40 "/work/logs/$O-build.log"; exit 1; }
        mkdir -p "/work/$O/bin"
        for t in $T; do cp "/work/$O/build/bin/$t" "/work/$O/bin/"; done
        chown -R "$HOST_UID:$HOST_GID" "/work/$O" /work/logs'
}
export -f build cpubuild
export WORK BRANCH REMOTE JOBS IMAGE D REPO

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
    git -C "$WORK/src" rev-parse HEAD > "$WORK/COMMIT"
    verdict build OK "$(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"
fi

if step cpubuild; then
    echo "=== 2. cpubuild $(date -Is)"
    # the checkout the rig build left in $WORK/src
    locked bash -c 'cpubuild src cpu "llama-server test-backend-ops test-hop-cast test-llama-archs test-llama-context"' \
        > "$WORK/logs/cpubuild.console" 2>&1
    rc=$?
    git -C "$WORK/src" rev-parse HEAD > "$WORK/cpu/bin/COMMIT" 2>/dev/null
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(cat "$WORK/cpu/bin/COMMIT" 2>/dev/null)"
    [ $rc = 0 ] || { date -Is > "$WORK/DONE"; exit 1; }
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

if step bench; then
    echo "=== 4. bench $(date -Is)"
    locked env DOCKER_ARGS="$D" nice -n 19 bash "$REPO/bench/megakernel_p106/build.sh" > "$WORK/logs/bench.log" 2>&1
    rc=$?
    B=$REPO/bench/megakernel_p106
    if [ $rc = 0 ]; then
        rm -rf "$WORK/bundle" && mkdir -p "$WORK/bundle"
        F=$REPO/docs/data/2026-11-03_flash_next_baseline/run
        cp "$B/build/megakernel_p106" "$B/build/BUILDINFO.txt" "$B/run_rig.sh" "$HERE/rig_hops.sh" "$F/fn_line.sh" \
            "$F/fn_requests.py" "$WORK/bundle/"
        (cd "$WORK/bundle" && md5sum megakernel_p106 run_rig.sh rig_hops.sh fn_line.sh fn_requests.py > MD5SUMS)
        cp "$B/build/ptxas.log" "$WORK/logs/bench-ptxas.log"
    fi
    verdict bench "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^(md5|avx|highest)' "$B/build/BUILDINFO.txt" 2>/dev/null | tr '\n' ';')"
fi

if step ops; then
    echo "=== 5. ops $(date -Is)"
    mkdir -p "$WORK/checks/ops"
    cpu_run "$WORK/checks/ops" 'nice -n 19 /x/test-backend-ops -b CPU -o CPY > /k/test-backend-ops.log 2>&1'
    rc=$?
    nh=$(grep -c 'CPY(type_src=[a-z0-9_]*,type_dst=[a-z0-9_]*,ne_src=\[2560,4,' "$WORK/checks/ops/test-backend-ops.log")
    verdict ops "$([ $rc = 0 ] && echo OK || echo FAIL)" \
        "rc $rc; $nh hop-shape cases; $(grep -E 'tests passed|backends passed' "$WORK/checks/ops/test-backend-ops.log" | tr -s ' ' | tr '\n' ';')"
fi

if step models; then
    echo "=== 6. models $(date -Is)"
    mkdir -p "$WORK/checks/models"
    cpu_run "$WORK/checks/models" '/x/test-llama-archs -a "^qwen4exp\$" -s 1 -o /k > /k/save.log 2>&1; ls /k/*.gguf'
    verdict models "$(ls "$WORK"/checks/models/*qwen4exp*.gguf > /dev/null 2>&1 && echo OK || echo FAIL)" \
        "$(cd "$WORK/checks/models" && ls *.gguf 2>/dev/null | tr '\n' ' ')"
fi
Q=$(cd "$WORK/checks/models" 2>/dev/null && ls *qwen4exp*.gguf 2>/dev/null | head -1)

if step unit; then
    echo "=== 7. unit $(date -Is)"
    mkdir -p "$WORK/checks/unit"
    cpu_run "$WORK/checks/unit" 'nice -n 19 /x/test-hop-cast --unit-only > /k/unit.log 2>&1'
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$WORK/checks/unit/unit.log")"
fi

if step model; then
    echo "=== 8. model $(date -Is)"
    mkdir -p "$WORK/checks/model"
    cp "$WORK/checks/models/$Q" "$WORK/checks/model/"
    cpu_run "$WORK/checks/model" 'nice -n 19 /x/test-hop-cast -m /k/'"$Q"' -t $THREADS --tsv /k/model.tsv \
        --arm off --arm "staged exact GGML_CUDA_HOP_STAGED=1" \
        --arm "f16-auto exact LLAMA_HOP_CAST=f16" --arm "q8_0-auto exact LLAMA_HOP_CAST=q8_0" \
        --arm "f16-at1 LLAMA_HOP_CAST=f16 LLAMA_HOP_CAST_AT=1" --arm "bf16-at1 LLAMA_HOP_CAST=bf16 LLAMA_HOP_CAST_AT=1" \
        --arm "q8_0-at1 LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1" > /k/model.log 2>&1
        rc=$?; rm -f /k/*.gguf; exit $rc'
    rc=$?
    verdict model "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$WORK/checks/model/model.log")"
fi

if step identity; then
    echo "=== 9. identity $(date -Is)"
    mkdir -p "$WORK/checks/identity"
    rm -f "$WORK"/checks/identity/*.f32
    cpu_run "$WORK/checks/identity" '
        run() { # <name> <bin dir> <env...>
            n=$1 b=$2; shift 2
            env "$@" LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $b/test-llama-archs -b CPU -s 1 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > /k/$n.log
            echo "$n rc=${PIPESTATUS[0]}"
        }
        run base   /xb
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run off    /x
        run staged /x GGML_CUDA_HOP_STAGED=1 LLAMA_HOP_CAST=q8_0
        run q8at1  /x LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1
        rm -f /k/*.f32' > "$WORK/checks/identity/rc.txt" 2>&1
    # per architecture and run: "<arch>: logits against <file>: nmse X, max abs diff Y, N of M differ". The dump is keyed
    # by the architecture's name, and test-llama-archs builds a second model for some of them (llama, granite, ...): the
    # base run already compares that second model against the first one's file, and every run repeats those lines
    printf 'run\tarch\tnmse\tmax_abs_diff\tdiffer\tof\tbase_self\n' > "$WORK/checks/identity/identity.tsv"
    grep 'logits against' "$WORK/checks/identity/base.log" > "$WORK/checks/identity/base_self.txt"
    for n in base off staged q8at1; do
        grep 'logits against' "$WORK/checks/identity/$n.log" | awk -v r=$n 'NR == FNR { s[$0] = 1; next }
            { if (match($0, /^[a-z0-9_.-]+: logits against [^:]*: nmse [^,]*, max abs diff [^,]*, [0-9]+ of [0-9]+ differ/)) {
                  split($0, f, /: logits against [^:]*: nmse |, max abs diff |, | of | differ/)
                  printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\n", r, f[1], f[2], f[3], f[4], f[5], ($0 in s) } }' \
            "$WORK/checks/identity/base_self.txt" - >> "$WORK/checks/identity/identity.tsv"
    done
    nb=$(cat "$WORK/checks/identity/n_base.txt" 2>/dev/null || echo 0)
    ns=$(wc -l < "$WORK/checks/identity/base_self.txt")
    d=$(awk -F'\t' 'NR > 1 && $1 != "base" { n[$1]++; if ($7) s[$1]++; else if ($5 > 0) { d[$1]++; a[$1] = a[$1] " " $2 } }
        END { for (r in n) printf "%s: %d compared against the base, %d differ%s, %d repeat the base'"'"'s own lines; ", r, n[r] - s[r], d[r], (d[r] ? " (" substr(a[r], 2) ")" : ""), s[r] }' \
        "$WORK/checks/identity/identity.tsv")
    ok=1
    for n in off staged q8at1; do
        c=$(awk -F'\t' -v r=$n '$1 == r && !$7' "$WORK/checks/identity/identity.tsv" | wc -l)
        s=$(awk -F'\t' -v r=$n '$1 == r && $7' "$WORK/checks/identity/identity.tsv" | wc -l)
        # every architecture the base wrote is compared once, and the base's own lines come back unchanged
        [ "$c" = "$nb" ] && [ "$c" -gt 0 ] && [ "$s" = "$ns" ] || ok=0
        # nothing differs from the base, except with the round trip before layer 1 in qwen4exp, the one architecture
        # that calls build_hop_cast
        x=$(awk -F'\t' -v r=$n '$1 == r && !$7 && $5 > 0 && !(r == "q8at1" && $2 == "qwen4exp")' "$WORK/checks/identity/identity.tsv" | wc -l)
        [ "$x" = 0 ] || ok=0
    done
    verdict identity "$([ $ok = 1 ] && echo OK || echo FAIL)" "base wrote $nb archs, $ns of its own lines; $d$(grep ' rc=' "$WORK/checks/identity/rc.txt" | tr '\n' ' ')"
fi

if step ctx; then
    echo "=== 10. ctx $(date -Is)"
    mkdir -p "$WORK/checks/ctx"
    cp "$WORK/checks/models/$Q" "$WORK/checks/ctx/"
    cpu_run "$WORK/checks/ctx" '
        fail=0
        for s in default "LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1"; do
            if [ "$s" = default ]; then e=""; n=default; else e="$s"; n=q8at1; fi
            env $e nice -n 19 /x/test-llama-context -m /k/'"$Q"' -t "$THREADS" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > "/k/context-$n.log"
            rc=${PIPESTATUS[0]}
            echo "$n: exit $rc, $(grep -E "passed   :|failed   :|skipped  :" /k/context-$n.log | tr -s " " | tr "\n" " ")"
            [ $rc = 0 ] || fail=1
            grep -q "failed   : 0" "/k/context-$n.log" || fail=1
        done
        rm -f /k/*.gguf /k/dump_state.*
        exit $fail' > "$WORK/checks/ctx/console.txt" 2>&1
    rc=$?
    verdict ctx "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tr '\n' ' ' < "$WORK/checks/ctx/console.txt")"
fi

if step server; then
    echo "=== 11. server $(date -Is)"
    mkdir -p "$WORK/checks/server"
    cp "$WORK/checks/models/$Q" "$WORK/checks/server/"
    cpu_run "$WORK/checks/server" '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        BASE_BIN=/xb BIN=/x MODEL=/k/'"$Q"' OUT=/k/out THREADS=$THREADS nice -n 19 bash /run/server_check.sh
        rc=$?; rm -f /k/*.gguf; exit $rc' > "$WORK/checks/server/console.txt" 2>&1
    rc=$?
    tail -n 12 "$WORK/checks/server/console.txt"
    verdict server "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tr '\n' ' ' < "$WORK/checks/server/out/verdict.txt" 2>/dev/null)"
fi

if step sm86; then
    echo "=== 12. sm86 $(date -Is)"
    # the same tree for compute capability 8.6 (RTX 30-series): llama-perplexity for kld_hops.sh, whose bases a native
    # build of that kind wrote, and the tests for gpu_check.sh (which can also run the rig build through the JIT)
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S /work/src -B /work/sm86/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
        T="llama-perplexity test-backend-ops test-hop-cast test-llama-archs"
        nice -n 19 cmake --build /work/sm86/build -j "$JOBS" --target $T > /work/logs/sm86-build.log 2>&1 \
            || { tail -40 /work/logs/sm86-build.log; exit 1; }
        mkdir -p /work/sm86/bin
        for t in $T; do cp "/work/sm86/build/bin/$t" /work/sm86/bin/; done
        git -C /work/src -c safe.directory=/work/src rev-parse HEAD > /work/sm86/bin/COMMIT
        chown -R "$HOST_UID:$HOST_GID" /work/sm86 /work/logs' > "$WORK/logs/sm86.console" 2>&1
    rc=$?
    verdict sm86 "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(cat "$WORK/sm86/bin/COMMIT" 2>/dev/null)"
fi

cat "$WORK/md5.txt" 2>/dev/null
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
