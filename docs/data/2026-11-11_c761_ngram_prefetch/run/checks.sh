#!/usr/bin/env bash
# C761 (Q1339): the n-gram table's rows prefetched by a helper thread (LLAMA_NGRAM_PREFETCH), checked on a build
# machine, CPU only: every build and test in a container (without network but for the rig build) at low CPU priority,
# one at a time under LOCK. Made from C738's checks.sh (docs/data/2026-11-11_c738_verify_prepare).
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of TIP, every target, at $WORK/cpu/tip/tree/{src,build}, with
#            one ccache (relative paths)
#   npf      test-llama-archs --ngram-prefetch at each seed of SEEDS: qwen4exp's random models written to a file and
#            loaded with the n-gram table mapped, a server's decode calls (prompt calls of several micro-batches, steps,
#            a second sequence) with the switch off (the inline path, counted) and on at the default cap, at 64 KiB and
#            at 0: every logit the same bytes; per call the same rows, distinct pages and row digest; no helper row
#            that differs from set_input's; the window within the cap
#   server   server_check.sh on the qwen4exp model with an MTP block, its table mapped: BASE_SRV's llama-server against
#            TIP's with the switch unset, on, on at 64 KiB, and the trace with it off and on (the calls compared by
#            ngram_calls.py)
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with
#            nothing set, LLAMA_NGRAM_PREFETCH=1 and LLAMA_NGRAM_PREFETCH_STATS=1 (these models are not mapped: nothing
#            is prefetched, and the n-gram rows come from the function the helper shares)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   vrig     npf with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub driver; its
#            CPU code has no AVX)
#   table    $WORK/npf.tsv and $WORK/npf_calls.tsv again from every --ngram-prefetch log
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [REMOTE=fork] [JOBS=8] [THREADS=8] [SEEDS='1 2 3']
#   [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04]
#   [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/npf.tsv, $WORK/npf_calls.tsv, logs in $WORK/logs and
#   $WORK/checks; done when $WORK/C761_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
BASE_SRV=$(realpath "${BASE_SRV:?BASE_SRV=<dir with the base llama-server>}")
: "${BRANCH:?BRANCH=<fork branch>}"
RUN=$(dirname "$(realpath "$0")")
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
SEEDS=${SEEDS:-1 2 3}
STEPS=${STEPS:-cpu npf server off ctest build vrig table}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C761_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c761 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step cpu; then
    R=$C/tip T=$C/tip/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
    echo "=== cpu $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
    rc=$?
    for b in test-llama-archs llama-server; do
        cp "$T/build/bin/$b" "$R/bin/" 2> /dev/null
    done
    verdict cpu-tip "$([ $rc = 0 ] && [ -x "$R/bin/llama-server" ] && [ -x "$R/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); all targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

# npf_run <tag> <bin dir> <runner: cpu|rig> <seeds>: --ngram-prefetch at every seed
npf_run() {
    local b=$1 bin=$2 how=$3 seeds=$4 s log rc sum nok
    mkdir -p "$K/npf"
    for s in $seeds; do
        log=$K/npf/$b-s$s.log
        if [ "$how" = rig ]; then
            # shellcheck disable=SC2086
            locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                 nice -n 19 '"$bin"'/test-llama-archs --ngram-prefetch -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
        else
            box bash -c 'nice -n 19 '"$bin"'/test-llama-archs --ngram-prefetch -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
        fi
        rc=$?
        sum=$(grep -a -m 1 '^Summary' "$log")
        nok=$(grep -a -c -E '^\|[^|]+\|[^|]+\|[^|]+\|identical\|.*\|OK' "$log")
        verdict "npf-$b-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && [ "$nok" -gt 0 ] && echo OK || echo FAIL)" \
            "rc=$rc; $nok arms OK; $sum"
    done
}

npf_table() {
    { printf 'build\tseed\tarch\ttype\tarm\tlogits\tcalls\tplanned\trows_off_on\thelper_rows\tinline_rows\tpages_off_on\tdigests_same\twindow_max_cap\tstatus\n'
      for f in "$K"/npf/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a -E '^\|[^|]+\| *(Dense|MoE)\|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { k = split(n, p, "-s"); for (i = 2; i <= 14; i++) gsub(/^ +| +$/, "", $i)
                st = $14; sub(/ .*/, "", st)
                print p[1], p[2], $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, st }'
      done; } > "$WORK/npf.tsv"
    { printf 'build\tseed\tarch\ttype\tarm\tcall\tn_tokens\tn_ubatch\tplanned\trows\trows_helper\trows_inline\trows_mismatch\tpages\tmadv_bytes\tdigest\twindow_max\n'
      for f in "$K"/npf/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^ngram-call|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { k = split(n, p, "-s"); $1 = p[1] OFS p[2]; print }'
      done; } > "$WORK/npf_calls.tsv"
}

if step npf; then
    echo "=== npf $(date -Is)"
    rm -rf "$K/npf"
    npf_run tip "$C/tip/bin" cpu "$SEEDS"
    npf_table
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    rm -rf "$S" && mkdir -p "$S/models"
    # the random qwen4exp MoE model with one MTP block, written by TIP's test-llama-archs
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp > $S/models/save.log 2>&1"
    M=$S/models/qwen4exp-moe-mtp.gguf
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin MODEL=$M OUT=$S/out THREADS=$THREADS \
        SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    verdict server "$([ "$rc" = 0 ] && grep -q '^PASS' "$S/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -v '^PASS\|^FAIL' "$S/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-1500)"
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$K/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env> [args]: rc
        local tag=$1 bin=$2 envs=$3
        shift 3
        box bash -c 'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 '"$*"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$BASE_CPU" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; ${BASE_CPU}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    cmp_arm() { # <tag> <env> [args]
        local tag=$1 envs=$2 rc n bad
        shift 2
        rc=$(archs "$tag" "$C/tip/bin" "$envs" "$@")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    }
    cmp_arm tip ''
    cmp_arm pf1 'LLAMA_NGRAM_PREFETCH=1'
    cmp_arm stats 'LLAMA_NGRAM_PREFETCH_STATS=1'
    rm -f "$A"/*.f32
fi

if step ctest; then
    echo "=== ctest $(date -Is)"
    B=$C/tip/tree/build
    box bash -c "cd $B && nice -n 19 ctest -j $JOBS --timeout 1800 --output-on-failure -E '$CTEST_EXCLUDE' --output-junit $K/ctest.xml" \
        > "$K/ctest.log" 2>&1
    rc=$?
    sline=$(grep -a 'tests passed, ' "$K/ctest.log" | tail -n 1)
    failed=$(sed -n '/The following tests FAILED:/,$p' "$K/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
    verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}; not run: $CTEST_EXCLUDE"
fi

if step build; then
    echo "=== build $(date -Is)"
    mkdir -p "$WORK/rig"
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

if step vrig; then
    echo "=== vrig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        npf_run rig "$T/bin" rig "$SEEDS"
        npf_table
    else
        verdict npf-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

if step table; then
    npf_table
fi

echo "=== done $(date -Is)"
touch "$WORK/C761_DONE"
