#!/usr/bin/env bash
# C693 (Q1226): the switches for two defaults upstream b12246 brought, LLAMA_MTP_H_DEV (#31927: the h rows an MTP draft
# reads kept on the target's output device) and GGML_SCHED_PREFETCH (#30966: a split's inputs fetched before the split
# that holds them runs), checked on a build machine, CPU only: every build and test in a container at low CPU priority,
# one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of the branch, every target, at $WORK/cpu/tip/tree/{src,build}
#            with one ccache (relative paths); test-llama-archs and llama-server kept in $WORK/cpu/tip/bin
#   hdev     test-llama-archs --mtp-h-dev at each seed of SEEDS in each arm of HDEV_ARMS (switches set on top of the
#            test's own): C703's draft-mtp sequence with the target's h rows on the host (LLAMA_MTP_H_DEV=0) against
#            the default, the device buffer on the CPU (=2) and both with LLAMA_MTP_CATCHUP_OVERLAP; the same caches,
#            drafts and probabilities, and the rows where each arm puts them
#   overlap  test-llama-archs --mtp-catchup-overlap (C703's own check, whose driver the new test shares) at each seed
#   off      test-llama-archs -b CPU -s 1: BASE's binary writes every architecture's logits, the branch's compares in
#            each arm of OFF_ARMS (no switch, each switch flipped, both)
#   server   llama-server's draft-mtp on the CPU with the tiny qwen4exp MTP model (run/server_smoke.py): BASE's server
#            and the branch's in each arm of SERVER_ARMS give the same tokens and draft counts, and the log shows the
#            h rows where the arm's switches put them
#   ctest    ctest in the branch's CPU build tree, without the tests that need a download, the network or torch
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   hdevrig  hdev's plain arm with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub
#            driver; its CPU code has no AVX)
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE=<bin/ with the base's static native CPU test-llama-archs and llama-server> [REMOTE=fork]
#   [JOBS=8] [THREADS=4] [SEEDS='1 2 3'] [HDEV_ARMS=...] [OFF_ARMS=...] [SERVER_ARMS=...] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [IMAGE=pq-c565-cpu:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...]
#   bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   *_ARMS    '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/hdev.tsv, $WORK/server/{arms,replies}.tsv, logs in
#   $WORK/logs and $WORK/checks; done when $WORK/C693_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE=$(realpath "${BASE:?BASE=<bin/ of the base CPU build>}")
: "${BRANCH:?BRANCH=<fork branch>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
SEEDS=${SEEDS:-1 2 3}
H=LLAMA_MTP_H_DEV
P=GGML_SCHED_PREFETCH
K1=LLAMA_QSA_KPOOL_DEV
HDEV_ARMS=${HDEV_ARMS:-plain= nopf=$P=0 stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1}
OFF_ARMS=${OFF_ARMS:-tip= hdev0=$H=0 hdev2=$H=2 nopf=$P=0 both=$H=0,$P=0}
SERVER_ARMS=${SERVER_ARMS:-default= hdev0=$H=0 hdev2=$H=2 nopf=$P=0 both=$H=0,$P=0 ovl=LLAMA_MTP_CATCHUP_OVERLAP=1 hdev0ovl=$H=0,LLAMA_MTP_CATCHUP_OVERLAP=1 hdev2ovl=$H=2,LLAMA_MTP_CATCHUP_OVERLAP=1}
STEPS=${STEPS:-cpu hdev overlap off server ctest build hdevrig}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
RUN=$(dirname "$(realpath "$0")")
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C693_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE:$BASE:ro" -v "$RUN:$RUN:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c693 checks started $(date -Is), $BRANCH at ${TIP:0:9}, base $BASE, steps: $STEPS"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi

if step cpu; then
    R=$C/tip T=$C/tip/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
    echo "=== cpu $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' -- -k > '"$R/build.log"' 2>&1'
    rc=$?
    cp "$T/build/bin/test-llama-archs" "$T/build/bin/llama-server" "$R/bin/" 2> /dev/null
    verdict cpu-tip "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && [ -x "$R/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); all targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

# hdev_run <build tag> <bin dir> <runner: cpu|rig> <arms>: each arm at every seed of SEEDS
hdev_run() {
    local b=$1 bin=$2 how=$3 arms=$4 a name envs s log rc sum
    mkdir -p "$K/hdev"
    for a in $arms; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $SEEDS; do
            log=$K/hdev/$b-$name-s$s.log
            if [ "$how" = rig ]; then
                # shellcheck disable=SC2086
                locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                    'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                     env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-h-dev -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            else
                box bash -c 'env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --mtp-h-dev -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            fi
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "hdev-$b-$name-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; ${envs:-no switch}; $sum"
        done
    done
}

hdev_table() {
    { printf 'build\tarm\tseed\tconfig\tdefault\tdev\tovl\thost_ovl\tdev_ovl\trows_host_default_dev_ovl_hostovl_devovl\twaited\tstatus\n'
      for f in "$K"/hdev/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= 11; i++) gsub(/^ +| +$/, "", $i)
                print p[1], p[2], s, $3, $4, $5, $6, $7, $8, $9, $10, $11 }'
      done; } > "$WORK/hdev.tsv"
}

if step hdev; then
    echo "=== hdev $(date -Is)"
    rm -f "$K"/hdev/tip-*.log
    hdev_run tip "$C/tip/bin" cpu "$HDEV_ARMS"
    hdev_table
fi

if step overlap; then
    echo "=== overlap $(date -Is)"
    rm -rf "$K/overlap" && mkdir -p "$K/overlap"
    for s in $SEEDS; do
        log=$K/overlap/tip-plain-s$s.log
        box bash -c 'nice -n 19 '"$C/tip/bin"'/test-llama-archs --mtp-catchup-overlap -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        sum=$(grep -a -m 1 '^Summary' "$log")
        verdict "overlap-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; $sum"
    done
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$K/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir> <env>: rc
        local tag=$1 bin=$2 envs=$3
        box bash -c 'env LLAMA_ARCHS_LOGITS='"$A"'/logits- '"$envs"' nice -n 19 '"$bin"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$BASE" '')
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; ${BASE}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    for a in $OFF_ARMS; do
        tag=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        rc=$(archs "$tag" "$C/tip/bin" "$envs")
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | grep -vxFf "$A/base_self.txt" > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        verdict "off-$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
    rm -f "$A"/*.f32
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$WORK/server
    rm -rf "$S" "$WORK/models" && mkdir -p "$S" "$WORK/models"
    box bash -c 'nice -n 19 '"$C/tip/bin"'/test-llama-archs -a "^qwen4exp$" -s 1 -o '"$WORK/models"' --mtp > '"$WORK/models/save-mtp.log"' 2>&1'
    M=$WORK/models/qwen4exp-moe-mtp.gguf
    args=(--arm "base=$BASE/llama-server")
    for a in $SERVER_ARMS; do
        name=${a%%=*} envs=${a#*=}
        args+=(--arm "$name=$C/tip/bin/llama-server${envs:+,$envs}")
    done
    if [ -s "$M" ]; then
        box nice -n 19 python3 -I "$RUN/server_smoke.py" --model "$M" --out "$S" --threads "$THREADS" "${args[@]}" > "$S/console.txt" 2>&1
        rc=$?
    else
        rc=1; echo "no model: $M" > "$S/console.txt"
    fi
    n_arms=$(grep -c '^ARM' "$S/console.txt")
    n_ok=$(grep -c '^ARM.*	OK$' "$S/console.txt")
    verdict server "$([ "$rc" = 0 ] && [ "$n_arms" -gt 0 ] && [ "$n_ok" = "$n_arms" ] && echo OK || echo FAIL)" \
        "rc=$rc; $n_ok of $n_arms arms give base's tokens and draft counts on R1-R3 with the h rows where their switches put them; server/arms.tsv"
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
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

if step hdevrig; then
    echo "=== hdevrig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        rm -f "$K"/hdev/rig-*.log
        hdev_run rig "$T/bin" rig "plain="
        hdev_table
    else
        verdict hdev-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C693_DONE"
