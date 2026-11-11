#!/usr/bin/env bash
# C735 (Q1302): context checkpoints inside a prompt's decode call (--ctx-checkpoints-incall, C217) with the MTP draft's
# catch-up queued behind the next call (LLAMA_MTP_CATCHUP_OVERLAP, C703), checked on a build machine, CPU only: every
# build and test in a container (without network but for the rig build) at low CPU priority, one at a time under LOCK.
# Made from C748's checks.sh (docs/data/2026-11-11_c748_kvonly_kpool).
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of TIP, every target, at $WORK/cpu/tip/tree/{src,build} with a
#            ccache (relative paths); test-llama-archs, test-llama-context and llama-server kept in $WORK/cpu/tip/bin
#   incall   test-llama-archs --mtp-incall-overlap at each seed of SEEDS in each arm of ARMS: the target and the
#            draft-mtp driver in the server's order, a prompt's checkpoints in separate calls (the reference, switch off
#            and on) and inside joined calls (off and on, and in the order of the base's server, off and on: the
#            control); every checkpoint's draft part and the driver's state at it, the draft's state after the prompt,
#            the drafts and the end state the same bytes as the reference
#   nohalf   the control of the driver's half: TIP with get_state_at reading the rows from the buffer's start, as the
#            base reads them, its test-llama-archs only ($WORK/cpu/nohalf); --mtp-incall-overlap at seed 1 must fail on
#            the driver's state at the checkpoints with the switch on, and on nothing else of that arm
#   overlap  test-llama-archs --mtp-catchup-overlap (C703's check) in each arm of ARMS
#   kvonly   test-llama-archs --mtp-kv-only (C620's check) in each arm of KV_ARMS
#   context  random-weight models from test-llama-archs (qwen4exp, qwen35, mamba; qwen4exp with an MTP block apart),
#            then test-llama-context on each: C217's check (its store_at and store_at_p cases and every other case)
#   server   server_check.sh on the qwen4exp model with an MTP block: BASE_SRV's llama-server against TIP's, every
#            arm with the MTP draft, the switches alone and together
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   incallrig  incall's arms with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub
#            driver; its CPU code has no AVX)
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [REMOTE=fork] [JOBS=8] [THREADS=8] [SEEDS='1 2 3']
#   [ARMS=...] [KV_ARMS=...] [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=pq-c565-cpu:24.04]
#   [SRV_IMAGE=pq-c735-srv:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   ARMS      '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/incall.tsv, $WORK/overlap.tsv, logs in $WORK/logs and
#   $WORK/checks; done when $WORK/C735_DONE exists
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
K1=LLAMA_QSA_KPOOL_DEV
ARMS=${ARMS:-plain= stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 stage2check=$K1=2,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1}
KV_ARMS=${KV_ARMS:-plain= stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1}
STEPS=${STEPS:-cpu incall nohalf overlap kvonly context server off ctest build incallrig}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-c735-srv:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C735_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c735 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
# the server check's image: the CPU image with curl and jq
if ! docker image inspect "$SRV_IMAGE" > /dev/null 2>&1; then
    printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl jq > /dev/null && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
        | docker build -q --label pascal-quants=1 -t "$SRV_IMAGE" - > "$WORK/logs/srv-image.log" 2>&1 || { echo "server image build failed"; exit 1; }
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
    for b in test-llama-archs test-llama-context llama-server; do cp "$T/build/bin/$b" "$R/bin/" 2> /dev/null; done
    verdict cpu "$([ $rc = 0 ] && [ -x "$R/bin/test-llama-archs" ] && [ -x "$R/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$R/tree.txt"); all targets; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

# archs_run <check> <flag> <build tag> <bin dir> <runner: cpu|rig> <arms>: every arm at every seed
archs_run() {
    local check=$1 flag=$2 b=$3 bin=$4 how=$5 arms=$6 a name envs s log rc sum
    mkdir -p "$K/$check"
    for a in $arms; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $SEEDS; do
            log=$K/$check/$b-$name-s$s.log
            if [ "$how" = rig ]; then
                # shellcheck disable=SC2086
                locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                    'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                     env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs '"$flag"' -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            else
                box bash -c 'env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs '"$flag"' -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            fi
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "$check-$b-$name-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; ${envs:-no switch}; $sum"
        done
    done
}

# table <check> <first column> <last column>: the qwen4exp rows of every log, one per build, arm, seed and configuration
table() {
    local check=$1 hdr=$2 last=$3 f n
    { printf 'build\tarm\tseed\tconfig\t%s\tstatus\n' "$hdr"
      for f in "$K/$check"/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" -v last="$last" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= last; i++) gsub(/^ +| +$/, "", $i)
                line = p[1] OFS p[2] OFS s; for (i = 3; i <= last; i++) line = line OFS $i; print line }'
      done; } > "$WORK/$check.tsv"
}

if step incall; then
    echo "=== incall $(date -Is)"
    rm -rf "$K/incall"
    archs_run incall --mtp-incall-overlap tip "$C/tip/bin" cpu "$ARMS"
    table incall $'ckpt_dft\tckpt_spec\tprompt\tdrafts\tend\tcut_on\tjoined_off\told_off\told_on_control\twaited_cut_joined' 14
fi

if step nohalf; then
    echo "=== nohalf $(date -Is)"
    R=$C/nohalf T=$C/nohalf/tree
    rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin" "$K/nohalf"
    git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
    sed -i 's/params\.ctx_tgt, cu_row_base + i_row - 1, 1,/params.ctx_tgt, i_row - 1, 1,/' "$T/src/common/speculative.cpp"
    np=$(grep -c 'params\.ctx_tgt, i_row - 1, 1,' "$T/src/common/speculative.cpp")
    # shellcheck disable=SC2086
    locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
        cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' --target test-llama-archs > '"$R/build.log"' 2>&1'
    rc=$?
    cp "$T/build/bin/test-llama-archs" "$R/bin/" 2> /dev/null
    log=$K/nohalf/nohalf-plain-s1.log
    if [ $rc = 0 ] && [ "$np" = 1 ] && [ -x "$R/bin/test-llama-archs" ]; then
        box bash -c 'nice -n 19 '"$R/bin"'/test-llama-archs --mtp-incall-overlap -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        # every row: the driver's state at the checkpoints differs, the draft's parts and everything after them do not
        nr=$(grep -a -c '^| *qwen4exp|' "$log")
        nx=$(grep -a '^| *qwen4exp|' "$log" | awk -F'|' '{ for (i = 4; i <= 13; i++) gsub(/^ +| +$/, "", $i) }
            $4 == "identical" && $5 == "DIFFERENT" && $6 == "identical" && $7 == "identical" && $8 == "identical" &&
            $9 == "identical" && $10 == "identical" && $11 == "identical"' | wc -l)
        verdict nohalf "$([ "$rc" != 0 ] && [ "$nr" -gt 0 ] && [ "$nx" = "$nr" ] && echo AS-EXPECTED || echo UNEXPECTED)" \
            "rc=$rc; get_state_at reads from the buffer's start; $nx of $nr configurations differ in the driver's state at the checkpoints only; $(grep -a -m 1 '^Summary' "$log")"
    else
        verdict nohalf FAIL "rc=$rc; $np line(s) changed; $R/build.log"
    fi
fi

if step overlap; then
    echo "=== overlap $(date -Is)"
    rm -rf "$K/overlap"
    archs_run overlap --mtp-catchup-overlap tip "$C/tip/bin" cpu "$ARMS"
    table overlap $'checkpoint\tprompt\tdrafts\tend\tckpt_spec\twaited_off_on' 10
fi

if step kvonly; then
    echo "=== kvonly $(date -Is)"
    rm -rf "$K/kvonly"
    archs_run kvonly --mtp-kv-only tip "$C/tip/bin" cpu "$KV_ARMS"
fi

M=$K/models
if step context; then
    echo "=== context $(date -Is)"
    A=$K/context
    rm -rf "$A" "$M" && mkdir -p "$A" "$M/plain" "$M/mtp"
    for a in qwen4exp qwen35 mamba; do
        box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^$a\$' -s 1 -o $M/plain > $M/save-$a.log 2>&1"
    done
    # a file with an MTP block loads only with the MTP layer requested: apart from the others
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $M/mtp --mtp > $M/save-mtp.log 2>&1"
    find "$M/mtp" -name '*.gguf' ! -name '*-mtp.gguf' -delete
    printf 'model\tcase\tverdict\treason\n' > "$A/context.tsv"
    for m in "$M"/plain/*.gguf; do
        n=$(basename "$m" .gguf)
        box bash -c "cd $A && nice -n 19 $C/tip/bin/test-llama-context -m $m -t $THREADS > $A/context-$n.log 2>&1; echo rc=\$? >> $A/context-$n.log"
        # one verdict line per case and run (C217's reading of the log)
        sed 's/\x1b\[[0-9;]*m//g; s/^[0-9][0-9]*\.[0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9][0-9] //' "$A/context-$n.log" | \
            awk -v m="$n" '$2 ~ /^(OK|SKIP|FAIL)$/ { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; print m "\t" $1 "\t" $2 "\t" r }' \
            >> "$A/context.tsv"
    done
    nf=$(awk -F'\t' '$3 == "FAIL"' "$A/context.tsv" | wc -l)
    ns=$(awk -F'\t' '$2 ~ /^store_at/ && $3 == "OK"' "$A/context.tsv" | wc -l)
    nm=$(ls "$M"/plain/*.gguf 2> /dev/null | wc -l)
    verdict context "$([ "$nm" -gt 0 ] && [ "$nf" = 0 ] && [ "$ns" -gt 0 ] && echo OK || echo FAIL)" \
        "$nm models; $(awk -F'\t' 'NR > 1' "$A/context.tsv" | wc -l) case runs, $nf FAIL; store_at/store_at_p OK in $ns runs; MTP model: $(ls "$M"/mtp/*.gguf 2> /dev/null | xargs -r -n 1 basename)"
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    q=$(ls "$M"/mtp/*-mtp.gguf 2> /dev/null | head -n 1)
    rm -rf "$S" && mkdir -p "$S"
    if [ -n "$q" ]; then
        cp "$RUN/server_check.sh" "$S/"
        # shellcheck disable=SC2086
        locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_SRV:$BASE_SRV:ro" -w "$WORK" "$SRV_IMAGE" bash -c '
            BASE='"$BASE_SRV"' BIN='"$C/tip/bin"' MODEL='"$q"' OUT='"$S"' THREADS='"$THREADS"' \
                SARGS="--override-kv qwen4exp.context_length=int:16384" nice -n 19 bash '"$S"'/server_check.sh' \
            > "$S/console.log" 2>&1
        verdict server "$(tail -n 1 "$S/verdict.txt" 2> /dev/null | grep -q '^PASS$' && echo OK || echo FAIL)" \
            "$(grep -v -e '^PASS$' -e '^FAIL$' "$S/verdict.txt" 2> /dev/null | tr '\n' ';')"
    else
        verdict server FAIL "no qwen4exp model with an MTP block"
    fi
fi

if step off; then
    echo "=== off $(date -Is)"
    A=$K/off
    rm -rf "$A" && mkdir -p "$A"
    archs() { # <tag> <bin dir>: rc
        box bash -c 'env LLAMA_ARCHS_LOGITS='"$A"'/logits- nice -n 19 '"$2"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$1.log"'; exit ${PIPESTATUS[0]}'
        echo $?
    }
    rc=$(archs base "$BASE_CPU")
    nb=$(ls "$A"/logits-*.f32 2> /dev/null | wc -l)
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep -a 'logits against' "$A/base.log" | grep -v ', 0 of ' > "$A/base_self.txt"
    verdict off-base "$([ "$rc" = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; ${BASE_CPU}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log"); $(wc -l < "$A/base_self.txt") self-differences"
    rc=$(archs tip "$C/tip/bin")
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
    env WORK="$WORK/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" REMOTE="$REMOTE" \
        JOBS="$JOBS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=build \
        bash "$REPO/docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$WORK/logs/build.log" 2>&1
    while IFS=$'\t' read -r c v d; do
        [ "$c" = check ] || verdict "$c" "$v" "$d"
    done < "$WORK/rig/summary.tsv"
fi

if step incallrig; then
    echo "=== incallrig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        archs_run incall --mtp-incall-overlap rig "$T/bin" rig "$ARMS"
        table incall $'ckpt_dft\tckpt_spec\tprompt\tdrafts\tend\tcut_on\tjoined_off\told_off\told_on_control\twaited_cut_joined' 14
    else
        verdict incall-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C735_DONE"
