#!/usr/bin/env bash
# C738 (Q1307): the verify pass prepared during draft-mtp's last drafting pass (llama_decode_prepare,
# LLAMA_VERIFY_PREPARE), checked on a build machine, CPU only: every build and test in a container (without network but
# for the rig build) at low CPU priority, one at a time under LOCK. Made from C748's checks.sh
# (docs/data/2026-11-11_c748_kvonly_kpool).
# Steps (STEPS, in this order by default):
#   cpu      static native CPU builds (GGML_CUDA=OFF): TIP, every target, at $WORK/cpu/tip/tree/{src,build}, and TIP with
#            GGML_RPC=ON (llama-server and ggml-rpc-server) at $WORK/cpu/rpc, with one ccache (relative paths)
#   vprep    test-llama-archs --verify-prepare at each seed of SEEDS, in each arm of VP_ARMS: every architecture's
#            server-step calls run plainly and prepared before their last token id (logits, and a backend top-k and
#            dist sampler); every output and the sequence's state must be the same bytes
#   fault    the control: --verify-prepare at seed 1 with LLAMA_VERIFY_PREPARE_FAULT=1, under which a prepared call
#            leaves the stand-in token id in the KV cells: every architecture that prepares must fail
#   server   server_check.sh on the qwen4exp model with an MTP block: BASE_SRV's llama-server against TIP's, the switch
#            unset, =1 and =2, with and without the backend sampler, and the RPC build's arms with every layer on an
#            RPC device on this machine
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with the
#            switch unset, =1 and =2 (nothing there calls the prepare: the variable alone must change nothing)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
#   vrig     vprep's arms with the rig build's test-llama-archs on the CPU (its CUDA backend loads against the stub
#            driver; its CPU code has no AVX)
#   table    $WORK/vprep.tsv again from every --verify-prepare log (vprep, fault and vrig write it too)
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [REMOTE=fork] [JOBS=8] [THREADS=8] [SEEDS='1 2 3']
#   [VP_ARMS=...] [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04]
#   [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   VP_ARMS   '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/vprep.tsv, logs in $WORK/logs and $WORK/checks; done when
#   $WORK/C738_DONE exists
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
VP_ARMS=${VP_ARMS:-vp= vp2=LLAMA_VERIFY_PREPARE=2}
STEPS=${STEPS:-cpu vprep fault server off ctest build vrig table}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C738_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c738 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step cpu; then
    for spec in "tip=:OFF" "rpc=llama-server ggml-rpc-server:ON"; do
        tag=${spec%%=*} rest=${spec#*=}
        targets=${rest%%:*} rpc=${rest#*:}
        R=$C/$tag T=$C/$tag/tree
        rm -rf "$T" "$R/bin" && mkdir -p "$T/src" "$R/bin"
        git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
        echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
        echo "=== cpu $tag $(date -Is)"
        # shellcheck disable=SC2086
        locked docker run $D --network none -v "$WORK:$WORK" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
            cmake -S '"$T/src"' -B '"$T/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DGGML_RPC='"$rpc"' -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$T/build"' -j '"$JOBS"' '"${targets:+--target $targets}"' -- -k > '"$R/build.log"' 2>&1'
        rc=$?
        for b in test-llama-archs llama-server ggml-rpc-server; do
            cp "$T/build/bin/$b" "$R/bin/" 2> /dev/null
        done
        verdict "cpu-$tag" "$([ $rc = 0 ] && [ -x "$R/bin/llama-server" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(cat "$R/tree.txt"); ${targets:-all targets}, GGML_RPC=$rpc; $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
    done
fi

# vp_run <tag> <bin dir> <runner: cpu|rig> <arms> <seeds>: --verify-prepare in every arm at every seed
vp_run() {
    local b=$1 bin=$2 how=$3 arms=$4 seeds=$5 a name envs s log rc sum
    mkdir -p "$K/vprep"
    for a in $arms; do
        name=${a%%=*} envs=${a#*=}
        envs=${envs//,/ }
        for s in $seeds; do
            log=$K/vprep/$b-$name-s$s.log
            if [ "$how" = rig ]; then
                # shellcheck disable=SC2086
                locked docker run $D --network none -v "$WORK:$WORK" "$CUDA_IMAGE" bash -c \
                    'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
                     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub &&
                     env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --verify-prepare -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            else
                box bash -c 'env '"$envs"' nice -n 19 '"$bin"'/test-llama-archs --verify-prepare -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            fi
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            # the architectures whose calls were prepared (Prepared n/m with n > 0)
            nprep=$(grep -a -c -E '^\|[^|]+\|[^|]+\|[^|]+\| *[1-9][0-9]*/[0-9]+\|' "$log")
            if [ "$name" = fault ]; then
                # the cells keep token ids for the PLE n-grams and M-RoPE only: every prepared qwen4exp row (PLE) must
                # fail, and the run with it
                nq=$(grep -a -c -E '^\| *qwen4exp\|[^|]+\|[^|]+\| *[1-9][0-9]*/[0-9]+\|' "$log")
                nqok=$(grep -a -E '^\| *qwen4exp\|[^|]+\|[^|]+\| *[1-9][0-9]*/[0-9]+\|' "$log" | grep -a -c '|OK')
                nfail=$(grep -a -E '^\|[^|]+\|[^|]+\|[^|]+\| *[1-9][0-9]*/[0-9]+\|' "$log" | grep -a -c '|FAIL')
                verdict "vprep-$b-$name-s$s" "$([ "$rc" != 0 ] && [ "$nq" -gt 0 ] && [ "$nqok" = 0 ] && echo AS-EXPECTED || echo UNEXPECTED)" \
                    "rc=$rc; $envs; $nprep rows prepared, $nfail of them failed; qwen4exp: $nq rows prepared, $nqok passed; $sum"
            else
                verdict "vprep-$b-$name-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && [ "$nprep" -gt 0 ] && echo OK || echo FAIL)" \
                    "rc=$rc; ${envs:-no variable}; $nprep rows prepared; $sum"
            fi
        done
    done
}

vp_table() {
    { printf 'build\tarm\tseed\tarch\tmodel\tsampling\tprepared\toutputs\tstate\tstatus\n'
      for f in "$K"/vprep/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a -E '^\|[^|]+\| *(Dense|MoE)\|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { k = split(n, p, "-"); s = p[k]; sub(/^s/, "", s)
                for (i = 2; i <= 8; i++) gsub(/^ +| +$/, "", $i)
                st = $8; sub(/ .*/, "", st)
                print p[1], p[2], s, $2, $3, $4, $5, $6, $7, st }'
      done; } > "$WORK/vprep.tsv"
}

if step vprep; then
    echo "=== vprep $(date -Is)"
    rm -rf "$K/vprep"
    vp_run tip "$C/tip/bin" cpu "$VP_ARMS" "$SEEDS"
    vp_table
fi

if step fault; then
    echo "=== fault $(date -Is)"
    vp_run tip "$C/tip/bin" cpu "fault=LLAMA_VERIFY_PREPARE_FAULT=1" 1
    vp_table
fi

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    rm -rf "$S" && mkdir -p "$S/models"
    # the random qwen4exp MoE model with one MTP block, written by TIP's test-llama-archs
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp > $S/models/save.log 2>&1"
    M=$S/models/qwen4exp-moe-mtp.gguf
    rpc_bin=
    [ -x "$C/rpc/bin/ggml-rpc-server" ] && [ -x "$C/rpc/bin/llama-server" ] && rpc_bin=$C/rpc/bin
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin RPC_BIN='$rpc_bin' MODEL=$M OUT=$S/out THREADS=$THREADS \
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
    cmp_arm vp1 'LLAMA_VERIFY_PREPARE=1'
    cmp_arm vp2 'LLAMA_VERIFY_PREPARE=2'
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
        vp_run rig "$T/bin" rig "$VP_ARMS" "$SEEDS"
        vp_table
    else
        verdict vprep-rig FAIL "no test-llama-archs in the tarball tree"
    fi
fi

if step table; then
    vp_table
fi

echo "=== done $(date -Is)"
touch "$WORK/C738_DONE"
