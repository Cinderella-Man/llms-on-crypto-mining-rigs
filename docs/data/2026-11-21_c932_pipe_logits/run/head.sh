#!/usr/bin/env bash
# C932 (2026-11-21): the fix (fork branch pascal-p106-v5-pipe-logits) checked on a build machine's CPU against its base
# pascal-p106-v5, plus how large the base's differences are and whether they change greedy tokens. No GPU work.
# Steps (STEPS, in this order):
#   src      a local clone of SEED in $W/rig/src (the fork as remote "fork", the branch fetched and checked against
#            HEAD_SHA); both compiler caches hard-linked from CPU_CCACHE and RIG_CCACHE; the base's test-llama-archs
#            logits copied from BASE_LOGITS for the ident step
#   check    check_at.sh on the head: Flash-Next's and Flint's random models (seed 1), depth 1, 0 and 50 % wrong
#            guesses, llama-pipe-recurrent-check
#   cpu      cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on the head: every target, ctest without network,
#            test-llama-archs plain, write, compare, ident against the base's logits files, the hand-run tests
#   tools    C880's tools.sh (docs/data/2026-11-21_c880_batch_api/run/) with the head's full build on C880's models
#            (MD): the pipespec checks (greedy, prompt in calls of 7, seeded, a race of two, two streams),
#            llama-pipe-recurrent-check, screen-check, test-moe-ids, test-hop-cast, the three pool tests
#   measure  llama-pipe-recurrent-check with prc_measure.patch built on the base and on the head: the largest absolute
#            and relative logit difference, pipelined against plain, and the yardstick of a changed order of sums
#            (the same tokens decoded again as one batch, PRC_BATCH=1), on SEEDS random Flash-Next models, PROMPTS,
#            LONG_N tokens; check_identity.sh at LONG_N tokens, depths 1 and 3, 0 and 50 % wrong guesses, on the same
#            models with both builds, and on Flint's random models with the head
#   compare  C880's compare.py on C880's base outputs (BASE_TOOLS, pascal-p106-v5) and the head's (checks/tools.tsv)
#   rig      the rig build (scripts/build_llamacpp_docker.sh REF, TESTS=1, test-llama-archs and test-cuda-keepalive
#            added, HF_UI_VERSION), under LOCK; the tarball stays in $W/rig
# Usage: W=<work dir> SEED=<llama.cpp clone with the base commit and the fork remote> REPO=<folder with check_at.sh
#   cpu_tests.sh tools.sh prc_measure.patch build_llamacpp_docker.sh pipespec/> [MD=<C880's models>]
#   [BASE_LOGITS=<the base's archs/logits>] [BASE_TOOLS=<C880's tools/base>] [CPU_CCACHE=<dir>] [RIG_CCACHE=<dir>] [LOCK=<flock file>] [STEPS=...]
#   [JOBS=8] [THREADS=6] [SEEDS='1 2 3 4 5'] [PROMPTS='rewrite code chat'] [LONG_N=512] bash head.sh
# Every container: labels pascal-quants=1 and c932=1, --cpu-shares=256, --memory=16g; every run of the code without
# network. Done when $W/DONE-head exists (it holds the exit code). At its end it stops every container labelled c932=1:
# run nothing else with that label alongside it.
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-pipe-logits}
HEAD_SHA=${HEAD_SHA:-80d83fa1a0e0ca18d20970b1e023ae8e6ea91e45}
BASE_SHA=${BASE_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src check cpu tools compare measure rig}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
SEEDS=${SEEDS:-1 2 3 4 5}
PROMPTS=${PROMPTS:-rewrite code chat}
LONG_N=${LONG_N:-512}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c932=1"
MEM="--memory=16g --memory-swap=16g"
S=$W/rig/src
mkdir -p "$W/cpu" "$W/rig" "$W/logs" "$W/tools" "$W/measure"
rm -f "$W/DONE-head"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c932=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE-head"; exit "$1"; }
box()    { local lib=$1; shift; docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:$REPO:ro" \
               ${MD:+-v "$MD:$MD:ro"} -e LD_LIBRARY_PATH="$lib" "$CPU_IMAGE" "$@"; }
export LC_ALL=C
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS"

if step src; then
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    if [ -n "${BASE_LOGITS:-}" ]; then rm -rf "$W/cpu/base/archs" && mkdir -p "$W/cpu/base/archs" && cp -a "$BASE_LOGITS" "$W/cpu/base/archs/logits"; fi
    echo "base logits files: $(ls "$W/cpu/base/archs/logits" 2> /dev/null | wc -l)"
    echo "=== sources ready $(date -Is)"
fi

if step check; then
    env SRC="$S" STAGE=head ARCHS="qwen4exp flint" MISSES="0 0.5" PRC=1 bash "$REPO/check_at.sh" "$HEAD_SHA"
    echo "head check rc=$?"
    echo "=== check done $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$S" TREES="c932=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand" JOBS="$JOBS" CTEST_JOBS=6 \
        THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu-head.log" 2>&1
    echo "head cpu tests rc=$?"
    grep -P '^c932\t' "$W/cpu/summary.tsv"
    sed -n '/The following tests FAILED:/,$p' "$W/cpu/c932/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //'
    echo "=== cpu done $(date -Is)"
fi

if step tools; then
    grep -q "commit $HEAD_SHA" "$W/cpu/c932/tree.txt" || { echo "the build tree is not the head's"; finish 1; }
    B=$W/cpu/tree/build/bin
    rm -rf "$W/tools/head" && mkdir -p "$W/tools/head"
    box "$B" env BIN="$B" MD="$MD" PS="$REPO/pipespec" OUT="$W/tools/head" THREADS="$THREADS" \
        nice -n 19 bash "$REPO/tools.sh" > "$W/logs/tools-head.log" 2>&1
    echo "tools head rc=$?"
    tr '\n' ' ' < "$W/tools/head/rc.tsv"; echo
    echo "=== tools done $(date -Is)"
fi

if step compare && [ -n "${BASE_TOOLS:-}" ]; then
    mkdir -p "$W/checks"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:$REPO:ro" -v "$BASE_TOOLS:$BASE_TOOLS:ro" \
        "$CPU_IMAGE" python3 -I "$REPO/compare.py" "$BASE_TOOLS" "$W/tools/head" "$W/checks/tools.tsv" > "$W/logs/compare.log" 2>&1
    echo "compare rc=$?"
    tail -n 5 "$W/logs/compare.log"
    echo "=== compare done $(date -Is)"
fi

if step measure; then
    M=$W/measure
    # the patched llama-pipe-recurrent-check of each build, with that build's libraries
    for t in base:$BASE_SHA head:$HEAD_SHA; do
        tag=${t%%:*} c=${t#*:}
        WORK="$W/cpu" SRC="$S" TREES="m$tag=$c:llama-pipe-recurrent-check" STEPS="tree" bash "$REPO/cpu_tests.sh" > "$M/tree-$tag.log" 2>&1
        GIT_CEILING_DIRECTORIES=$W/cpu/tree git -C "$W/cpu/tree/src" apply "$REPO/prc_measure.patch" || { echo "patch failed ($tag)"; finish 1; }
        echo "patch prc_measure.patch applied to the $tag tree" >> "$W/cpu/m$tag/tree.txt"
        WORK="$W/cpu" SRC="$S" TREES="m$tag=$c:llama-pipe-recurrent-check,test-llama-archs,llama-pipespec" STEPS="build" JOBS="$JOBS" \
            LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$M/build-$tag.log" 2>&1
        rm -rf "$M/bin-$tag" && mkdir -p "$M/bin-$tag"
        cp -a "$W/cpu/tree/build/bin"/*.so* "$W/cpu/tree/build/bin"/{llama-pipe-recurrent-check,test-llama-archs,llama-pipespec} "$M/bin-$tag/" \
            || { echo "no measuring build ($tag)"; finish 1; }
    done
    # the random models, made once with the base's test-llama-archs (the head does not change it)
    for s in $SEEDS; do
        for a in qwen4exp flint; do
            mkdir -p "$M/models/$a-s$s"
            box "$M/bin-base" bash -c "cd $M/models/$a-s$s && LLAMA_TEST_ARCHS_N_VOCAB=4096 $M/bin-base/test-llama-archs -a '^$a\$' -s $s -o . > gen.log 2>&1"
        done
    done
    (cd "$M/models" && md5sum */*.gguf) > "$M/models/md5.txt"
    printf 'build\tmodel\tseed\tprompt\tn_gen\tpipe tokens\tpipe logits\tsteps differing\tfirst\tmax abs\tmax rel\tmax |logit|\tmin top-2 margin\targmax differing\tbatch steps differing\tbatch max abs\tbatch max rel\tbatch argmax differing\n' > "$M/prc.tsv"
    for tag in base head; do
        for s in $SEEDS; do
            m=$(ls "$M/models/qwen4exp-s$s"/*.gguf | head -n 1)
            for p in $PROMPTS; do
                log=$M/prc-$tag-s$s-$p.log
                box "$M/bin-$tag" env PRC_BATCH=1 PRC_STEPS=1 timeout 1800 "$M/bin-$tag/llama-pipe-recurrent-check" -m "$m" \
                    -f "$REPO/pipespec/prompts/$p.txt" -n "$LONG_N" -ngl 0 -t "$THREADS" -c 4096 -fa on > "$log" 2>&1
                rc=$?
                f() { grep -a -o "^$1 .*" "$log" | tail -n 1 | grep -o " $2=[^ ]*" | cut -d= -f2; }
                printf '%s\tqwen4exp\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$s" "$p" \
                    "$(f PIPE-RECURRENT-CHECK n_gen)" "$(f PIPE-RECURRENT-CHECK tokens)" "$(f PIPE-RECURRENT-CHECK logits)" \
                    "$(f PRC-PIPE steps_diff)" "$(f PRC-PIPE first_diff)" "$(f PRC-PIPE max_abs)" "$(f PRC-PIPE max_rel)" \
                    "$(f PRC-PIPE max_logit)" "$(f PRC-PIPE min_margin)" "$(f PRC-PIPE argmax_diff)" \
                    "$(f PRC-BATCH steps_diff)" "$(f PRC-BATCH max_abs)" "$(f PRC-BATCH max_rel)" "$(f PRC-BATCH argmax_diff)" \
                    | tee -a "$M/prc.tsv"
                [ $rc = 0 ] || [ $rc = 2 ] || echo "prc $tag s$s $p rc=$rc"
            done
        done
    done
    printf 'build\tmodel\tseed\tprompt\trc\tpipelined runs\truns differing\truns with other tokens\twrong guesses\n' > "$M/long.tsv"
    long() { # <build> <arch> <seed> <prompt>
        local m out
        m=$(ls "$M/models/$2-s$3"/*.gguf | head -n 1)
        out=$M/long/$1-$2-s$3-$4
        box "$M/bin-$1" env BIN="$M/bin-$1" MODEL="$m" CHAT=0 SPEED=0 TIMEOUT=1800 PROMPT="$REPO/pipespec/prompts/$4.txt" OUT="$out" \
            ARGS="-ngl 0 -t $THREADS -c 4096 -fa on" N="$LONG_N" DEPTHS="1 3" MISSES="0 0.5" \
            bash "$REPO/pipespec/check_identity.sh" > "$out.log" 2>&1
        local rc=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$rc" \
            "$(grep -vc '^plain ' <(grep -E ' (same|DIFF)$' "$out.log"))" "$(grep -c ' DIFF$' "$out.log")" \
            "$(awk '$1 != "plain" && $1 != "run" && $8 == "DIFF"' "$out.log" | wc -l)" \
            "$(awk '$1 != "plain" && $1 != "run" && NF > 10 { s += $5 } END { print s + 0 }' "$out.log")" | tee -a "$M/long.tsv"
    }
    mkdir -p "$M/long"
    for s in $SEEDS; do
        for p in $PROMPTS; do
            long base qwen4exp "$s" "$p"
            long head qwen4exp "$s" "$p"
            long head flint "$s" "$p"
        done
    done
    echo "=== measure done $(date -Is)"
fi

if step rig; then
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rig build rc=$?"
    ls -la "$W"/rig/llama.cpp-*.tar.gz 2> /dev/null && md5sum "$W"/rig/llama.cpp-*.tar.gz
    echo "=== rig done $(date -Is)"
fi
finish 0
