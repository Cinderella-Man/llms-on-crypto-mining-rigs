#!/usr/bin/env bash
# C619 (Q1104): the MTP draft context's graph kept per batch shape (LLAMA_MTP_GRAPH_KEEP), checked on a build machine,
# CPU only: every build and test in a container (the CPU ones without network) at low CPU priority, one at a time
# under LOCK.
# Steps (STEPS, in this order by default):
#   cpu      static native CPU builds (GGML_CUDA=OFF) of TIP (every target) and, with BEFORE set, BEFORE (the test
#            without the switch: its test-llama-archs only), each at $WORK/cpu/<tag>/tree/{src,build} with one ccache
#            (relative paths); test-llama-archs and llama-server kept in $WORK/cpu/<tag>/bin
#   keep     test-llama-archs --mtp-graph-keep at each seed of SEEDS in each arm of ARMS (switches set in both runs of
#            the test, the switch off and on): the draft-mtp driver with the target over a prompt and 16 steps of
#            every catch-up width, in eight configurations; the draft's states, drafts and probabilities the same off
#            and on, fewer graphs built with the switch. BEFORE's binary must compute the same and build no fewer
#   regress  test-llama-archs --mtp-kv-only and --mtp-catchup-overlap with LLAMA_MTP_GRAPH_KEEP=1 in both of their
#            runs: C620's and C703's checks with the draft's graphs kept
#   server   llama-server on the CPU backend with test-llama-archs's tiny qwen4exp model with one MTP block, the MTP
#            draft at three tokens with backend sampling (run/server_smoke.py): in each variant (plain; the head's
#            shortlist, LLAMA_DRAFT_SHORTLIST; a draft cut, --spec-draft-p-min 0.102) the base build (BASE_BIN) and
#            TIP with the switch off and on answer the same three requests; replies, draft counts and LLAMA_SPEC_DUMP
#            the same, and the draft's graph builds counted from LLAMA_UBATCH_TRACE
#   off      test-llama-archs -b CPU -s 1: BASE_BIN's binary writes every architecture's logits, TIP's compares with
#            every switch off
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   keeprig  keep's arms with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub
#            driver; its CPU code has no AVX)
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_BIN=<bin/ of the base's static native CPU build: test-llama-archs, llama-server>
#   [BEFORE=<commit>] [REMOTE=fork] [JOBS=8] [THREADS=4] [SEEDS='1 2 3'] [ARMS=...] [PORT=18619] [LOCK=<flock file>] [DOCKER_ARGS=...]
#   [IMAGE=pq-c565-cpu:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   ARMS      '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/keep.tsv, $WORK/server/ (replies, logs, dumps, counts,
#   compare.txt), logs in $WORK/logs and $WORK/checks; done when $WORK/C619_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_BIN=$(realpath "${BASE_BIN:?BASE_BIN=<bin/ of the base CPU build>}")
: "${BRANCH:?BRANCH=<fork branch>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
SEEDS=${SEEDS:-1 2 3}
PORT=${PORT:-18619}
K1=LLAMA_QSA_KPOOL_DEV
ARMS=${ARMS:-plain= stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 qsa4check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1,LLAMA_QSA_POOL_MASK_DEV=1,LLAMA_QSA_POOL_MASK_DEV_CHECK=1,LLAMA_QSA_INDEXER_ROWS=1}
STEPS=${STEPS:-cpu keep regress server off ctest build keeprig}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
RUN=$(dirname "$(realpath "$0")")
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C619_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_BIN:$BASE_BIN:ro" -v "$RUN:$RUN:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
BEFORE=${BEFORE:-}
[ -n "$BEFORE" ] && BEFORE=$(git -C "$SRC" rev-parse "$BEFORE")
echo "=== c619 checks started $(date -Is), $BRANCH at ${TIP:0:9}, before ${BEFORE:0:9}, base $BASE_BIN, steps: $STEPS"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi

if step cpu; then
    specs="tip=$TIP"
    [ -n "$BEFORE" ] && specs="$specs before=$BEFORE:test-llama-archs"
    for spec in $specs; do
        tag=${spec%%=*} rest=${spec#*=}
        commit=${rest%%:*} targets=""; [[ "$rest" == *:* ]] && targets=${rest#*:}
        R=$C/$tag T=$C/$tag/tree
        rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
        git -C "$SRC" archive "$commit" | tar -x -C "$T/src"
        echo "commit $commit $(git -C "$SRC" log -1 --format=%s "$commit")" > "$R/tree.txt"
        echo "=== cpu $tag $(date -Is)"
        # shellcheck disable=SC2086
        locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
            cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' '"${targets:+--target $targets}"' -- -k > '"$R/build.log"' 2>&1'
        rc=$?
        cp "$T/build/bin/test-llama-archs" "$R/bin/" 2> /dev/null
        [ -z "$targets" ] && cp "$T/build/bin/llama-server" "$R/bin/" 2> /dev/null
        verdict "cpu-$tag" "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && { [ -n "$targets" ] || [ -x "$R/bin/llama-server" ]; } && echo OK || echo FAIL)" \
            "rc=$rc; $(cat "$R/tree.txt"); ${targets:-all targets}; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
    done
fi

# keep_run <build tag> <bin dir> <runner: cpu|rig>: every arm of ARMS at every seed of SEEDS
keep_run() {
    local b=$1 bin=$2 how=$3 a name envs s log rc sum
    mkdir -p "$K/keep"
    for a in $ARMS; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $SEEDS; do
            log=$K/keep/$b-$name-s$s.log
            if [ "$how" = rig ]; then
                # shellcheck disable=SC2086
                locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                    'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                     env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-graph-keep -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            else
                box bash -c 'env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-graph-keep -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            fi
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            if [ "$b" = before ]; then
                # the test without the switch in the build: every configuration computes the same and builds no fewer
                v=$([ "$rc" != 0 ] && ! grep -aq DIFFERENT "$log" && [ "$(grep -ac 'built no fewer graphs' "$log")" = 8 ] && echo AS-EXPECTED || echo UNEXPECTED)
            else
                v=$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)
            fi
            verdict "keep-$b-$name-s$s" "$v" "rc=$rc; ${envs:-no switch}; $sum"
        done
    done
}

keep_table() {
    { printf 'build\tarm\tseed\tconfig\tprompt\tdrafts\tend\tshortlist\tbuilt_prompt_off_on\tbuilt_steps_off_on_of\tcompute_kib_off_on\tstatus\n'
      for f in "$K"/keep/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= 11; i++) gsub(/^ +| +$/, "", $i)
                print p[1], p[2], s, $3, $4, $5, $6, $7, $8, $9, $10, $11 }'
      done; } > "$WORK/keep.tsv"
}

if step keep; then
    echo "=== keep $(date -Is)"
    rm -rf "$K/keep"
    keep_run tip "$C/tip/bin" cpu
    [ -n "$BEFORE" ] && keep_run before "$C/before/bin" cpu
    keep_table
fi

if step regress; then
    echo "=== regress $(date -Is)"
    rm -rf "$K/regress" && mkdir -p "$K/regress"
    for mode in mtp-kv-only mtp-catchup-overlap; do
        for s in $SEEDS; do
            log=$K/regress/$mode-s$s.log
            box bash -c 'env LLAMA_MTP_GRAPH_KEEP=1 nice -n 19 '"$C/tip/bin"'/test-llama-archs --'"$mode"' -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "regress-$mode-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; LLAMA_MTP_GRAPH_KEEP=1; $sum"
        done
    done
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$WORK/server
    rm -rf "$S" && mkdir -p "$S/models"
    # shellcheck disable=SC2016
    box bash -c '
        set -u
        S='"$S"' B='"$C/tip/bin"' BASE='"$BASE_BIN"' R='"$RUN"' PORT='"$PORT"' TH='"$THREADS"'
        $B/test-llama-archs -a "^qwen4exp$" -s 1 -o $S/models --mtp > $S/models/save-mtp.log 2>&1
        M=$S/models/qwen4exp-moe-mtp.gguf
        echo "model $M ($(stat -c %s $M) bytes)"
        # the draft settings of the release preset that a CPU takes: flash attention, backend sampling, three drafted
        # tokens, a draft micro-batch of 128
        A="-ngl 0 -t $TH -c 4096 -np 1 --no-webui -fa on -bs --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128"
        arm() { # <name> <bin> <extra server args> [VAR=value ...]
            local name=$1 bin=$2 extra=$3; shift 3
            nice -n 19 python3 -I $R/server_smoke.py arm "$name" $S $PORT $bin/llama-server $M "$@" -- $A $extra
        }
        for v in plain short pmin; do
            # short: ids below 100 of the 128 in the tiny model, so that the head flips to the full one after each target
            # token above them; pmin: the top probability of the random draft over its top-k of 10 is about 0.10, so
            # that a cut at 0.102 ends drafts after 0 to 3 tokens and the catch-ups have every width from 1 to 4
            case $v in
                plain) X="";                          E="" ;;
                short) X="";                          E="LLAMA_DRAFT_SHORTLIST=100,8,2" ;;
                pmin)  X="--spec-draft-p-min 0.102";  E="" ;;
            esac
            arm base-$v $BASE "$X" $E
            arm off-$v  $B    "$X" $E
            arm on-$v   $B    "$X" $E LLAMA_MTP_GRAPH_KEEP=1
        done
        python3 -I $R/server_smoke.py compare $S plain=base-plain,off-plain,on-plain short=base-short,off-short,on-short \
            pmin=base-pmin,off-pmin,on-pmin > $S/compare.txt
        echo "compare rc=$?"
        grep -h "LLAMA_MTP_GRAPH_KEEP\|draft head shortlist" $S/on-*.server.log | sort | uniq -c' > "$S/console.txt" 2>&1
    rc=$?
    n_id=$(grep -c $'\tidentical\t' "$S/compare.txt") n_diff=$(grep -c $'\tDIFFER\t' "$S/compare.txt")
    verdict server-identity "$([ "$rc" = 0 ] && [ "$n_diff" = 0 ] && [ "$n_id" = 36 ] && echo OK || echo FAIL)" \
        "$n_id of 36 identical (3 variants x base, off, on x 3 replies and the dump), $n_diff differ; rc=$rc"
    for v in plain short pmin; do
        line_off=$(grep "| $v | off-$v |" "$S/compare.txt") line_on=$(grep "| $v | on-$v |" "$S/compare.txt")
        b_off=$(awk -F'|' '{ gsub(/ /, "", $7); split($7, x, "/"); print x[1] }' <<< "$line_off")
        b_on=$(awk -F'|' '{ gsub(/ /, "", $7); split($7, x, "/"); print x[1] }' <<< "$line_on")
        verdict "server-builds-$v" "$([ -n "$b_off" ] && [ -n "$b_on" ] && [ "$b_on" -lt "$b_off" ] && echo FEWER || echo NOT-FEWER)" \
            "draft graphs of up to 8 tokens rebuilt: off $b_off, on $b_on (compare.txt)"
    done
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
    rc=$(archs base "$BASE_BIN" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; ${BASE_BIN}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    rc=$(archs tip "$C/tip/bin" '')
    n=$(grep -ac 'logits against' "$A/tip.log")
    grep -a 'logits against' "$A/tip.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/tip.differ.txt"
    bad=$(cut -d: -f1 "$A/tip.differ.txt" | tr '\n' ' ')
    verdict off-tip "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; no switch; $n compared, $(wc -l < "$A/tip.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/tip.log"); $(grep -a -m 1 '^Summary' "$A/tip.log")"
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
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_BIN" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

if step keeprig; then
    echo "=== keeprig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        keep_run rig "$T/bin" rig
        keep_table
    else
        verdict keep-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C619_DONE"
