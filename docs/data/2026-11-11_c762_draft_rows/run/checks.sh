#!/usr/bin/env bash
# C762 (Q1340): the MTP draft's token embeddings on its layer's device (LLAMA_DRAFT_EMBD_DEV=layer), checked on a build
# machine, CPU only: every build and test in a container (without network but for the rig build) at low CPU priority,
# one at a time under LOCK. Made from C738's checks.sh (docs/data/2026-11-11_c738_verify_prepare).
# Steps (STEPS, in this order by default):
#   cpu      static native CPU builds (GGML_CUDA=OFF): TIP, every target, at $WORK/cpu/tip/tree/{src,build}, and TIP with
#            GGML_RPC=ON (llama-server and ggml-rpc-server) at $WORK/cpu/rpc, with one ccache (relative paths)
#   server   server_check.sh: TIP's test-llama-archs writes the random qwen4exp model with one MTP block and that block
#            alone as a draft file (--mtp-draft); BASE_SRV's llama-server against TIP's on the CPU with the switch unset
#            and set, and the RPC build's arms with two RPC devices on this machine, the draft's layer on the second and
#            its table on the first (-otd), the switch unset and set, with -bs, and with the draft's nodes hashed
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with
#            the switch unset, LLAMA_DRAFT_EMBD_DEV=layer, and LLAMA_TENSOR_HASH on a few node names (no model there
#            is an MTP-only file, so the switch places nothing; the hash callback must change no logit)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), logs in $WORK/logs and $WORK/checks; done when $WORK/C762_DONE
#   exists
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
STEPS=${STEPS:-cpu server off ctest build}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C762_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c762 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

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

if step server; then
    echo "=== server $(date -Is)"
    S=$K/server
    rm -rf "$S" && mkdir -p "$S/models"
    # the random qwen4exp MoE model with one MTP block and its MTP-only draft, written by TIP's test-llama-archs
    box bash -c "nice -n 19 $C/tip/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp-draft > $S/models/save.log 2>&1"
    M=$S/models/qwen4exp-moe-mtp.gguf
    DR=$S/models/qwen4exp-moe-mtp-draft.gguf
    if [ -s "$M" ] && [ -s "$DR" ]; then
        verdict models OK "$(grep -a -o 'save_mtp_draft: .*' "$S/models/save.log" | sed 's|.*/||'); $(stat -c '%n %s' "$M" "$DR" | sed 's|[^ ]*/||' | tr '\n' ' ')"
    else
        verdict models FAIL "$(tail -n 3 "$S/models/save.log" | tr '\n' ' ')"
    fi
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin RPC_BIN=$C/rpc/bin MODEL=$M DRAFT=$DR OUT=$S/out THREADS=$THREADS \
        SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    verdict server "$([ "$rc" = 0 ] && grep -q '^PASS' "$S/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -v '^PASS\|^FAIL' "$S/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-2000)"
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
    cmp_arm on 'LLAMA_DRAFT_EMBD_DEV=layer'
    cmp_arm hash 'LLAMA_TENSOR_HASH=attn_norm,ffn_inp,result_norm'
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

echo "=== done $(date -Is)"
touch "$WORK/C762_DONE"
