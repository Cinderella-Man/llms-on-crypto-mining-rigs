#!/usr/bin/env bash
# C771 (Q1357): LLAMA_MTP_EMBD_FIRST=1, the MTP draft's token-row gather built first so that the k-pool tables' update
# (LLAMA_QSA_KPOOL_DEV) joins its layer's split, checked on a build machine, CPU only: every build and test in a
# container (without network but for the rig build) at low CPU priority, one at a time under LOCK. Made from C762's
# checks.sh (docs/data/2026-11-11_c762_draft_rows) and C748's KV-only step (docs/data/2026-11-11_c748_kvonly_kpool).
# Steps (STEPS, in this order by default):
#   cpu      static native CPU builds (GGML_CUDA=OFF): TIP, every target, at $WORK/cpu/tip/tree/{src,build}, and TIP with
#            GGML_RPC=ON (llama-server and ggml-rpc-server) at $WORK/cpu/rpc, with one ccache (relative paths)
#   server   server_check.sh on the random qwen4exp model with one MTP block and that block alone as a draft file (from
#            MODELS: C762's test-llama-archs --mtp-draft); BASE_SRV's llama-server against TIP's on the CPU, unset and with the
#            switch, and the RPC build's arm pairs (without and with the switch) on two RPC devices laid out as cards 4
#            and 8, each with the scheduler's trace: the splits of every draft evaluation (splits.py)
#   kvonly   test-llama-archs --mtp-kv-only (C748's check) in each arm of ARMS with the switch added, at each seed of SEEDS
#   off      test-llama-archs -b CPU -s 1: BASE_CPU's binary writes every architecture's logits, TIP's compares with
#            every switch off, with the switch, and qwen4exp with the switch and the tables on the devices (C100, C720's
#            stage, C100's check)
#   ctest    ctest in TIP's CPU build tree, without the tests that need a download, the network or torch (as C565)
#   build    the rig tarball through C620's checks.sh (scripts/build_llamacpp_docker.sh with TESTS=1, then
#            test-llama-archs from the same CUDA build tree into the tarball tree's bin/); md5 in $WORK/rig/md5.txt
# Usage: WORK=<dir> REPO=<this repository> SRC=<llama.cpp clone with the fork as remote REMOTE (https fetch URL)>
#   BRANCH=<fork branch> BASE_CPU=<bin/ with the base's static native test-llama-archs>
#   BASE_SRV=<dir with the base's static native llama-server> MODELS=<dir with C762's two model files (server)>
#   [REMOTE=fork] [JOBS=8] [THREADS=8] [SEEDS='1 2 3']
#   [ARMS=...] [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=pq-c735-srv:24.04] [STEPS=...] bash checks.sh
#   SRC       for the rig build it is also scripts/build_llamacpp_docker.sh's clone ($WORK/rig/src), which that script
#             checks out and cleans
#   ARMS      '<name>=<VAR=value>[,<VAR=value>...] ...'; '<name>=' sets nothing but the switch
#   IMAGE     an Ubuntu 24.04 image with git, cmake, build-essential, ccache, python3, curl and jq
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/kvonly.tsv, logs in $WORK/logs and $WORK/checks; done when
#   $WORK/C771_DONE exists
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
E=LLAMA_MTP_EMBD_FIRST=1
K1=LLAMA_QSA_KPOOL_DEV
ARMS=${ARMS:-plain= dev1=$K1=1 dev1check=$K1=1,${K1}_CHECK=1 stage1=$K1=1,LLAMA_QSA_KPOOL_STAGE=1 stage1check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 stage2check=$K1=2,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1 qsa4check=$K1=1,LLAMA_QSA_KPOOL_STAGE=1,${K1}_CHECK=1,LLAMA_QSA_POOL_MASK_DEV=1,LLAMA_QSA_POOL_MASK_DEV_CHECK=1,LLAMA_QSA_INDEXER_ROWS=1}
STEPS=${STEPS:-cpu server kvonly off ctest build}
IMAGE=${IMAGE:-pq-c735-srv:24.04}
CTEST_EXCLUDE='^(test-conversion-fp8|test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|test-arg-parser)$'
D="--rm --cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
K=$WORK/checks
mkdir -p "$WORK/logs" "$K" "$C/ccache"
rm -f "$WORK/C771_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { locked docker run $D --network none -v "$WORK:$WORK" -v "$BASE_CPU:$BASE_CPU:ro" -v "$BASE_SRV:$BASE_SRV:ro" "$IMAGE" "$@"; }
git -C "$SRC" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$SRC" rev-parse "$REMOTE/$BRANCH")
echo "=== c771 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

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
    # the random qwen4exp MoE model with one MTP block and its MTP-only draft, written by test-llama-archs -a '^qwen4exp$'
    # -s 1 -o <dir> --mtp-draft, which is C762's (pascal-p106-v4-b12246-c762), not on this branch: MODELS holds the two
    # files C762's run wrote
    M=$S/models/qwen4exp-moe-mtp.gguf
    DR=$S/models/qwen4exp-moe-mtp-draft.gguf
    cp "${MODELS:?MODELS=<dir with the two model files of C762>}"/qwen4exp-moe-mtp{,-draft}.gguf "$S/models/"
    cp "$MODELS/save.log" "$S/models/" 2> /dev/null
    if [ -s "$M" ] && [ -s "$DR" ]; then
        verdict models OK "$(grep -a -o 'save_mtp_draft: .*' "$S/models/save.log" | sed 's|.*/||'); $(md5sum "$M" "$DR" | sed 's|  [^ ]*/| |' | tr '\n' ' ')"
    else
        verdict models FAIL "$(tail -n 3 "$S/models/save.log" | tr '\n' ' ')"
    fi
    box bash -c "BASE=$BASE_SRV BIN=$C/tip/bin RPC_BIN=$C/rpc/bin MODEL=$M DRAFT=$DR OUT=$S/out THREADS=$THREADS \
        SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $RUN/server_check.sh" > "$S/console.txt" 2>&1
    rc=$?
    verdict server "$([ "$rc" = 0 ] && grep -q '^PASS' "$S/out/verdict.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $(grep -v '^PASS\|^FAIL' "$S/out/verdict.txt" 2>/dev/null | tr '\n' ';' | cut -c1-3000)"
fi

if step kvonly; then
    echo "=== kvonly $(date -Is)"
    rm -rf "$K/kvonly" && mkdir -p "$K/kvonly"
    for a in $ARMS; do
        name=${a%%=*} envs=${a#*=}
        envs="${envs//,/ } $E"
        for s in $SEEDS; do
            log=$K/kvonly/tip-$name-s$s.log
            box bash -c 'env '"$envs"' nice -n 19 '"$C/tip/bin"'/test-llama-archs --mtp-kv-only -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$log"'; exit ${PIPESTATUS[0]}'
            rc=$?
            sum=$(grep -a -m 1 '^Summary' "$log")
            verdict "kvonly-$name-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" "rc=$rc; $envs; $sum"
        done
    done
    { printf 'build\tarm\tseed\tconfig\tkv_prompt\tdrafts\tkv_end\tnodes_off_on\tfull_off_on\trows_off_on\tcatchup_ubatches\tcopies_off_on\tbytes_off_on\tstatus\n'
      for f in "$K"/kvonly/*-s*.log; do
          n=$(basename "$f" .log)
          grep -a '^| *qwen4exp|' "$f" | awk -F'|' -v n="$n" 'BEGIN { OFS = "\t" }
              { split(n, p, "-"); s = p[3]; sub(/^s/, "", s)
                for (i = 2; i <= 13; i++) gsub(/^ +| +$/, "", $i)
                print p[1], p[2], s, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13 }'
      done; } > "$WORK/kvonly.tsv"
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
    cmp_arm on    "$E"
    cmp_arm stage "$E $K1=1 LLAMA_QSA_KPOOL_STAGE=1" -a qwen4exp
    cmp_arm check "$E $K1=1 LLAMA_QSA_KPOOL_STAGE=1 ${K1}_CHECK=1" -a qwen4exp
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
touch "$WORK/C771_DONE"
