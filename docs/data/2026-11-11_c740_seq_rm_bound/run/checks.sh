#!/usr/bin/env bash
# C740 (Q1309): seq_rm bounded by the cells it can change (LLAMA_KV_SEQ_RM_BOUND), checked on a build machine, CPU only:
# every build and test in a container (the CPU ones without network) at low CPU priority, one at a time under LOCK.
# Steps (STEPS, in this order by default):
#   cpu      a static native CPU build (GGML_CUDA=OFF) of the branch's tip, every target, at $WORK/cpu/tip/tree/{src,build}
#            with one ccache (relative paths); test-llama-archs and llama-server kept in $WORK/cpu/tip/bin
#   bound    test-llama-archs --kv-seq-rm in each arm of ARMS (its architectures, seeds and switches, set in both
#            contexts of the test): two contexts of one model, the switch off and on, through 400 random decodes and
#            memory edits each; the memory's host state, the return values, the logits and the state write the same.
#            All arms in one container (run/bound_loop.sh); a test that aborts the process is run again with the switch
#            off in both contexts, and left out of the arm when it aborts there too (BASE-ABORT)
#   control  the same with LLAMA_ARCHS_SEQ_RM_BOUND_ON=2 (the switch's test hook that leaves the cells unsorted) on
#            CONTROL_ARCHS: the check must find a difference
#   time     test-llama-archs --kv-seq-rm-time: the host time of one seq_rm call off and on at 65,536 cells
#   server   llama-server on the CPU backend with test-llama-archs's tiny qwen4exp model with one MTP block, the MTP
#            draft at three tokens with backend sampling (C619's run/server_smoke.py), -c 4096: in each variant (plain;
#            the head's shortlist; a draft cut, --spec-draft-p-min 0.102) the base build (BASE_BIN) and the tip with the
#            switch off and on answer the same three requests; replies, draft counts and LLAMA_SPEC_DUMP the same
#   trace    the same server at -c 65536 (the model saved with a 65,536-token context) with C405's LLAMA_STEP_TRACE, the
#            switch off, on, off, on; replies and dumps the same, and run/rm_stages.py's two stages that hold seq_rm
#   off      test-llama-archs -b CPU -s 1: BASE_BIN's binary writes every architecture's logits, the tip's compares with
#            every switch off
#   ctest    ctest in the tip's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   boundrig the arms of RIG_ARMS with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the
#            stub driver; its CPU code has no AVX)
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_BIN=<bin/ of the base's static native CPU build: test-llama-archs, llama-server>
#   [REMOTE=fork] [JOBS=8] [THREADS=4] [ARMS=...] [RIG_ARMS=...] [CONTROL_ARCHS=<regex>] [PORT=18740] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [IMAGE=pq-c565-cpu:24.04] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   ARMS      '<name>@<architecture regex>@<seeds, comma-separated>@<VAR=value>[,<VAR=value>...] ...'
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/bound.tsv, $WORK/time.tsv, $WORK/server/ (replies, logs,
#   dumps, traces, compare.txt, rm_stages.md), logs in $WORK/logs and $WORK/checks; done when $WORK/C740_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_BIN=$(realpath "${BASE_BIN:?BASE_BIN=<bin/ of the base CPU build>}")
: "${BRANCH:?BRANCH=<fork branch>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
PORT=${PORT:-18740}
K1=LLAMA_QSA_KPOOL_DEV
KQ=LLAMA_GPU_KQ_MASK=1,LLAMA_GPU_KQ_MASK_CHECK=1
QSA=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1,LLAMA_QSA_POOL_MASK_DEV=1,LLAMA_QSA_POOL_MASK_DEV_CHECK=1,LLAMA_QSA_INDEXER_ROWS=1
SET='^(llama|qwen4exp|qwen3next|qwen35moe|gemma3|mamba2|deepseek2|deepseek4)$'
ARMS=${ARMS:-plain@.@1@ plain-set@$SET@2,3@ kqmask@.@1@$KQ qsa@^qwen4exp\$@1,2,3@$QSA,$KQ}
RIG_ARMS=${RIG_ARMS:-plain-set@$SET@1,2,3@ qsa@^qwen4exp\$@1@$QSA}
CONTROL_ARCHS=${CONTROL_ARCHS:-^(llama|qwen4exp|gemma3)\$}
STEPS=${STEPS:-cpu bound control time server trace off ctest build boundrig}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
RUN=$(dirname "$(realpath "$0")")
SMOKE=$(realpath "$RUN/../../2026-11-11_c619_draft_graph_reuse/run/server_smoke.py")
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C740_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_BIN:$BASE_BIN:ro" -v "$RUN:$RUN:ro" -v "$(dirname "$SMOKE"):$(dirname "$SMOKE"):ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c740 checks started $(date -Is), $BRANCH at ${TIP:0:9}, base $BASE_BIN, steps: $STEPS"

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
        "rc=$rc; $(cat "$R/tree.txt"); $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
fi

# bound_run <build tag> <bin dir> <runner: cpu|rig> <arms> [VAR=value]: every arm at each of its seeds, all of them in one
# container (run/bound_loop.sh: a test that aborts the process is run again with the switch off in both contexts, and
# left out when it aborts there too); its verdict lines go into summary.tsv
bound_run() {
    local b=$1 bin=$2 how=$3 arms=$4 extra=${5:-}
    mkdir -p "$K/bound"
    if [ "$how" = rig ]; then
        # shellcheck disable=SC2086,SC2016
        locked docker run $D --network none -v "$WORK:$WORK" -v "$RUN:$RUN:ro" "$CUDA_IMAGE" bash -c \
            'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
             export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub && bash "$0" "$@"' \
            "$RUN/bound_loop.sh" "$bin" "$K/bound" "$b" "$arms" "$extra" > "$K/bound/console-$b.txt" 2>&1
    else
        box bash "$RUN/bound_loop.sh" "$bin" "$K/bound" "$b" "$arms" "$extra" > "$K/bound/console-$b.txt" 2>&1
    fi
    if [ -s "$K/bound/verdicts-$b.tsv" ]; then
        tee -a "$WORK/summary.tsv" < "$K/bound/verdicts-$b.tsv"
    else
        verdict "bound-$b" FAIL "no verdicts (checks/bound/console-$b.txt)"
    fi
}

bound_table() {
    { printf 'build\tarm\tseed\tarch\tconfig\tcache\tedits\tdecodes_failed\trefused\tseq_rm_on\tstatus\n'
      for f in "$K"/bound/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *[a-z0-9_-]*|' "$f" | grep -v 'Model arch' | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { nb = split(n, p, "-"); s = p[nb]; sub(/^s/, "", s); arm = p[2]; for (i = 3; i < nb; i++) arm = arm "-" p[i]
                for (i = 2; i <= 9; i++) gsub(/^ +| +$/, "", $i)
                print p[1], arm, s, $2, $3, $4, $5, $6, $7, $8, $9 }'
      done; } > "$WORK/bound.tsv"
}

if step bound; then
    echo "=== bound $(date -Is)"
    rm -rf "$K/bound"
    bound_run tip "$C/tip/bin" cpu "$ARMS"
    bound_table
fi

if step control; then
    echo "=== control $(date -Is)"
    bound_run control "$C/tip/bin" cpu "plain@$CONTROL_ARCHS@1@" LLAMA_ARCHS_SEQ_RM_BOUND_ON=2
    bound_table
fi

if step time; then
    echo "=== time $(date -Is)"
    log=$K/time.log
    box bash -c 'nice -n 19 '"$C/tip/bin"'/test-llama-archs --kv-seq-rm-time -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
    rc=$?
    grep -a $'^arch\t\|^[a-z0-9]*\t65536\t' "$log" > "$WORK/time.tsv"
    verdict time "$([ $rc = 0 ] && grep -aq 'all timings ran' "$log" && echo OK || echo FAIL)" \
        "rc=$rc; $(($(wc -l < "$WORK/time.tsv") - 1)) rows in time.tsv; $(grep -a -m 1 '^Summary' "$log")"
fi

if step server || step trace; then
    S=$WORK/server
    mkdir -p "$S"
fi

if step server; then
    echo "=== server $(date -Is)"
    rm -rf "$S/models" && mkdir -p "$S/models"
    # shellcheck disable=SC2016
    box bash -c '
        set -u
        S='"$S"' B='"$C/tip/bin"' BASE='"$BASE_BIN"' SMOKE='"$SMOKE"' PORT='"$PORT"' TH='"$THREADS"'
        $B/test-llama-archs -a "^qwen4exp$" -s 1 -o $S/models --mtp > $S/models/save-mtp.log 2>&1
        M=$S/models/qwen4exp-moe-mtp.gguf
        echo "model $M ($(stat -c %s $M) bytes)"
        A="-ngl 0 -t $TH -c 4096 -np 1 --no-webui -fa on -bs --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128"
        arm() { # <name> <bin> <extra server args> [VAR=value ...]
            local name=$1 bin=$2 extra=$3; shift 3
            nice -n 19 python3 -I $SMOKE arm "$name" $S $PORT $bin/llama-server $M "$@" -- $A $extra
        }
        for v in plain short pmin; do
            case $v in
                plain) X="";                          E="" ;;
                short) X="";                          E="LLAMA_DRAFT_SHORTLIST=100,8,2" ;;
                pmin)  X="--spec-draft-p-min 0.102";  E="" ;;
            esac
            arm base-$v $BASE "$X" $E
            arm off-$v  $B    "$X" $E
            arm on-$v   $B    "$X" $E LLAMA_KV_SEQ_RM_BOUND=1
        done
        python3 -I $SMOKE compare $S plain=base-plain,off-plain,on-plain short=base-short,off-short,on-short \
            pmin=base-pmin,off-pmin,on-pmin > $S/compare.txt
        echo "compare rc=$?"
        grep -h "LLAMA_KV_SEQ_RM_BOUND" $S/on-*.server.log | sort | uniq -c' > "$S/console.txt" 2>&1
    rc=$?
    n_id=$(grep -c $'\tidentical\t' "$S/compare.txt") n_diff=$(grep -c $'\tDIFFER\t' "$S/compare.txt")
    n_on=$(grep -h 'LLAMA_KV_SEQ_RM_BOUND' "$S"/on-*.server.log | wc -l)
    verdict server-identity "$([ "$rc" = 0 ] && [ "$n_diff" = 0 ] && [ "$n_id" = 36 ] && [ "$n_on" -gt 0 ] && echo OK || echo FAIL)" \
        "$n_id of 36 identical (3 variants x base, off, on x 3 replies and the dump), $n_diff differ; switch log lines in the on arms: $n_on; rc=$rc"
fi

if step trace; then
    echo "=== trace $(date -Is)"
    rm -rf "$S/models65k" && mkdir -p "$S/models65k"
    # shellcheck disable=SC2016
    box bash -c '
        set -u
        S='"$S"' B='"$C/tip/bin"' SMOKE='"$SMOKE"' PORT='"$PORT"' TH='"$THREADS"' R='"$RUN"'
        LLAMA_ARCHS_N_CTX=65536 $B/test-llama-archs -a "^qwen4exp$" -s 1 -o $S/models65k --mtp > $S/models65k/save-mtp.log 2>&1
        M=$S/models65k/qwen4exp-moe-mtp.gguf
        echo "model $M ($(stat -c %s $M) bytes)"
        A="-ngl 0 -t $TH -c 65536 -np 1 --no-webui -fa on -bs --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128"
        for a in toff ton toff2 ton2; do
            E=""; [ "${a#ton}" != "$a" ] && E="LLAMA_KV_SEQ_RM_BOUND=1"
            nice -n 19 python3 -I $SMOKE arm $a $S $PORT $B/llama-server $M LLAMA_STEP_TRACE=$S/$a.trace.jsonl $E -- $A
        done
        python3 -I $SMOKE compare $S t65k=toff,ton,toff2,ton2 > $S/compare65k.txt
        echo "compare rc=$?"
        python3 -I $R/rm_stages.py toff=$S/toff.trace.jsonl ton=$S/ton.trace.jsonl toff2=$S/toff2.trace.jsonl \
            ton2=$S/ton2.trace.jsonl > $S/rm_stages.md
        echo "rm_stages rc=$?"' > "$S/console65k.txt" 2>&1
    rc=$?
    n_id=$(grep -c $'\tidentical\t' "$S/compare65k.txt") n_diff=$(grep -c $'\tDIFFER\t' "$S/compare65k.txt")
    verdict server-trace65k "$([ "$rc" = 0 ] && [ "$n_diff" = 0 ] && [ "$n_id" = 16 ] && [ -s "$S/rm_stages.md" ] && echo OK || echo FAIL)" \
        "$n_id of 16 identical (off, on, off, on x 3 replies and the dump), $n_diff differ; rm_stages.md $(grep -c '^| t' "$S/rm_stages.md") loads; rc=$rc"
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

if step boundrig; then
    echo "=== boundrig $(date -Is)"
    T=$(ls -td "$WORK"/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | head -n 1)
    if [ -n "$T" ] && [ -x "$T/bin/test-llama-archs" ]; then
        bound_run rig "$T/bin" rig "$RIG_ARMS"
        bound_table
    else
        verdict bound-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C740_DONE"
