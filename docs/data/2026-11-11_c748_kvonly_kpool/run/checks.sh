#!/usr/bin/env bash
# C748 (Q1320): C620's KV-only MTP pass (LLAMA_MTP_KV_ONLY) with C100's k-pool tables on the devices
# (LLAMA_QSA_KPOOL_DEV) and C720's stage (LLAMA_QSA_KPOOL_STAGE), checked on a build machine, CPU only: every build and
# test in a container (the CPU ones without network) at low CPU priority, one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   cpu      static native CPU builds (GGML_CUDA=OFF) of TIP (every target) and BEFORE (TIP's test without the fix: its
#            test-llama-archs only), each at $WORK/cpu/<tag>/tree/{src,build} with one ccache (relative paths); the
#            binaries kept in $WORK/cpu/<tag>/bin
#   kvonly   test-llama-archs --mtp-kv-only at each seed of SEEDS in each arm of ARMS (switches set on top of the test's
#            own LLAMA_MTP_KV_ONLY off and on), with TIP's and BEFORE's CPU binaries: the draft's KV after the prompt,
#            every draft and the KV at the end the same off and on, and with the tables on the devices their full
#            rewrites and rows uploaded the same off and on; the k-pool copies and bytes of the catch-up ubatches
#   overlap  test-llama-archs --mtp-catchup-overlap (C703's driver with the target; one configuration with
#            LLAMA_MTP_KV_ONLY) in each arm of OVERLAP_ARMS, TIP's CPU binary
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with every
#            switch off, then qwen4exp with the tables on (C100 and C720; with C100's check; the draft's only)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   kvrig    kvonly's arms with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub
#            driver; its CPU code has no AVX), TIP only
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BEFORE=<commit> BASE_CPU=<bin/ of the base's static native CPU build> [REMOTE=fork] [JOBS=8]
#   [THREADS=4] [SEEDS='1 2 3'] [ARMS=...] [OVERLAP_ARMS=...] [LOCK=<flock file>] [DOCKER_ARGS=...]
#   [IMAGE=pq-c565-cpu:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   ARMS      '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/kvonly.tsv, $WORK/overlap.tsv, logs in $WORK/logs and
#   $WORK/checks; done when $WORK/C748_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BEFORE:?BEFORE=<commit>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
SEEDS=${SEEDS:-1 2 3}
K1=LLAMA_QSA_KPOOL_DEV
ARMS=${ARMS:-plain= dev1=$K1=1 dev1check=$K1=1,${K1}_CHECK=1 stage1=$K1=1,LLAMA_QSA_KPOOL_STAGE=1 stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 stage2check=$K1=2,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 qsa4check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1,LLAMA_QSA_POOL_MASK_DEV=1,LLAMA_QSA_POOL_MASK_DEV_CHECK=1,LLAMA_QSA_INDEXER_ROWS=1}
OVERLAP_ARMS=${OVERLAP_ARMS:-plain= stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 stage2check=$K1=2,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1}
STEPS=${STEPS:-cpu kvonly overlap off ctest build kvrig}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C748_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
BEFORE=$(git -C "$SRC" rev-parse "$BEFORE")
echo "=== c748 checks started $(date -Is), $BRANCH at ${TIP:0:9}, before ${BEFORE:0:9}, steps: $STEPS"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi

if step cpu; then
    for spec in "tip=$TIP" "before=$BEFORE:test-llama-archs"; do
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
        verdict "cpu-$tag" "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(cat "$R/tree.txt"); ${targets:-all targets}; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
    done
fi

# kvonly_run <build tag> <bin dir> <runner: cpu|rig>: every arm of ARMS at every seed of SEEDS
kvonly_run() {
    local b=$1 bin=$2 how=$3 a name envs s log rc sum v
    mkdir -p "$K/kvonly"
    for a in $ARMS; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $SEEDS; do
            log=$K/kvonly/$b-$name-s$s.log
            if [ "$how" = rig ]; then
                # shellcheck disable=SC2086
                locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                    'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                     env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-kv-only -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            else
                box bash -c 'env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-kv-only -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            fi
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            if [ "$b" = before ]; then
                # the test without the fix: it passes with the tables on the host and fails with them on the devices
                v=$([ "$name" = plain ] && [ "$rc" = 0 ] || { [ "$name" != plain ] && [ "$rc" != 0 ] && grep -aq 'left the k-pool tables' "$log"; } && echo AS-EXPECTED || echo UNEXPECTED)
            else
                v=$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)
            fi
            verdict "kvonly-$b-$name-s$s" "$v" "rc=$rc; $envs; $sum"
        done
    done
}

kvonly_table() {
    { printf 'build\tarm\tseed\tconfig\tkv_prompt\tdrafts\tkv_end\tnodes_off_on\tfull_off_on\trows_off_on\tcatchup_ubatches\tcopies_off_on\tbytes_off_on\tstatus\n'
      for f in "$K"/kvonly/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= 13; i++) gsub(/^ +| +$/, "", $i)
                print p[1], p[2], s, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13 }'
      done; } > "$WORK/kvonly.tsv"
}

if step kvonly; then
    echo "=== kvonly $(date -Is)"
    rm -rf "$K/kvonly"
    kvonly_run tip "$C/tip/bin" cpu
    kvonly_run before "$C/before/bin" cpu
    kvonly_table
fi

if step overlap; then
    echo "=== overlap $(date -Is)"
    rm -rf "$K/overlap" && mkdir -p "$K/overlap"
    for a in $OVERLAP_ARMS; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $SEEDS; do
            log=$K/overlap/tip-$name-s$s.log
            box bash -c 'env '"$envs"' nice -n 19 '"$C/tip/bin"'/test-llama-archs --mtp-catchup-overlap -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "overlap-$name-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; $envs; $sum"
        done
    done
    { printf 'arm\tseed\tconfig\tcheckpoint\tprompt\tdrafts\tend\tckpt_spec\twaited_off_on\tstatus\n'
      for f in "$K"/overlap/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= 10; i++) gsub(/^ +| +$/, "", $i)
                print p[2], s, $3, $4, $5, $6, $7, $8, $9, $10 }'
      done; } > "$WORK/overlap.tsv"
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
    cmp_arm tip   ''
    cmp_arm stage "$K1=1 LLAMA_QSA_KPOOL_STAGE=1" -a qwen4exp
    cmp_arm check "$K1=1 LLAMA_QSA_KPOOL_STAGE=1 ${K1}_CHECK=1" -a qwen4exp
    cmp_arm draft "$K1=2 LLAMA_QSA_KPOOL_STAGE=1" -a qwen4exp
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

if step kvrig; then
    echo "=== kvrig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        kvonly_run rig "$T/bin" rig
        kvonly_table
    else
        verdict kvonly-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C748_DONE"
