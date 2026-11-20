#!/usr/bin/env bash
# C898 (Q1623): build and check the fork branch pascal-p106-v5-mtp-plain-attn (c9c3134d4, one commit on
# pascal-p106-v5 602691153) on a build machine's CPU, driven from another machine over ssh (git and ssh only there).
# Every step runs on the build machine in Docker, one container at a time under a lock, no GPU.
#
# Environment:
#   BUILDER  ssh destination of the build machine (required; root or a docker user)
#   WORK     work folder on the build machine (default /tmp/c898)
#   SEED     an earlier v5 build folder on the build machine with rig/src (a llama.cpp clone), rig/ccache and
#            cpu-ccache, hard-linked in by `seed` (optional; without it `seed` clones from GitHub and the caches start
#            empty)
#   LOCK     the build lock file on the build machine (default $WORK/build.lock)
#   IMG      the CPU image (default pq-c565-cpu:24.04: Ubuntu 24.04, gcc 13.3.0, cmake 3.28.3, ccache)
#   REPO     this repository's checkout on the driving machine (default: the repository root above this script)
#
# Steps, in order:
#   seed                 WORK/rig/src with the fork as remote `fork`, the caches from SEED
#   cpu REF T            every target from REF, CPU only, static, native, tests on, no network (WORK/cpu-T/build.log,
#                        done marker WORK/cpu-T/RC); T=new REF=fork/pascal-p106-v5-mtp-plain-attn, T=base REF=602691153;
#                        env TEST_FROM=REF2 takes tests/test-llama-archs.cpp from REF2 (T=basetest: the base's code
#                        with the new test, to see the test catch the bug)
#   test T               cpu-T's test-llama-archs on qwen4exp, seed 1: --mtp-kv-only without and with the preset's
#                        three QSA switches, --mtp-graph-keep, --mtp-replan, --mtp-catchup-overlap and (T=new)
#                        --mtp-plain-attn; outputs WORK/test/T-*.txt; env ONLY="<tests>" runs just those
#   compare              base's and new's --mtp-kv-only tables (both switch settings), line for line
#   archs                every architecture, test-llama-archs -s 1, on base and new (WORK/test/T-archs-all.txt), diffed
#   rig REF T            scripts/build_llamacpp_docker.sh REF with TESTS=1 (the rig tarball under WORK/rig, log
#                        WORK/logs/rig-T.log, done marker WORK/rig/RC-T); env EXTRA_TARGETS and HF_UI_VERSION pass
#                        through
#   wait FILE            blocks until WORK/FILE exists (cpu-T/RC or rig/RC-T), then prints it and the build's errors
#   status | stop
# The run of 2026-11-20/21:
#   seed; cpu fork/pascal-p106-v5-mtp-plain-attn new; wait cpu-new/RC; cpu 602691153 base; wait cpu-base/RC;
#   test new; test base; compare; archs;
#   HF_UI_VERSION=b12483 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" rig fork/pascal-p106-v5-mtp-plain-attn final;
#   wait rig/RC-final;
#   TEST_FROM=fork/pascal-p106-v5-mtp-plain-attn cpu 602691153 basetest; wait cpu-basetest/RC;
#   ONLY=--mtp-plain-attn test basetest
set -uo pipefail
: "${BUILDER:?set BUILDER to the ssh destination of the build machine}"
REPO=${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
Q="ssh $BUILDER"
M=${WORK:-/tmp/c898}
SEED=${SEED:-}
LOCK=${LOCK:-$M/build.lock}
IMG=${IMG:-pq-c565-cpu:24.04}
SW="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1"
case "${1:-status}" in
seed)
    timeout 1800 $Q "set -e; mkdir -p $M/rig $M/logs $M/repo $M/test $M/cpu-ccache $(dirname $LOCK)
        if [ ! -d $M/rig/src/.git ]; then
            if [ -n '$SEED' ]; then git clone -q --no-checkout $SEED/rig/src $M/rig/src
            else git clone -q --no-checkout https://github.com/ggml-org/llama.cpp $M/rig/src; fi
        fi
        git -C $M/rig/src remote set-url origin https://github.com/ggml-org/llama.cpp
        git -C $M/rig/src remote get-url fork >/dev/null 2>&1 || git -C $M/rig/src remote add fork https://github.com/Cinderella-Man/miner-llama
        git -C $M/rig/src remote set-url fork https://github.com/Cinderella-Man/miner-llama
        if [ -n '$SEED' ]; then
            [ -d $M/rig/ccache ] || cp -al $SEED/rig/ccache $M/rig/ccache
            [ -n \"\$(ls -A $M/cpu-ccache)\" ] || { rmdir $M/cpu-ccache; cp -al $SEED/cpu-ccache $M/cpu-ccache; }
        fi
        git -C $M/rig/src remote -v; du -sh $M"
    ;;
cpu)
    REF=${2:?ref}; T=${3:?tag}
    timeout 600 $Q "set -e; git -C $M/rig/src fetch -q fork; git -C $M/rig/src fetch -q origin --tags; rm -rf $M/cpu-$T; mkdir -p $M/cpu-$T/src
        git -C $M/rig/src rev-parse $REF > $M/cpu-$T/COMMIT
        git -C $M/rig/src archive $REF | tar -x -C $M/cpu-$T/src
        if [ -n '${TEST_FROM:-}' ]; then
            git -C $M/rig/src show ${TEST_FROM:-}:tests/test-llama-archs.cpp > $M/cpu-$T/src/tests/test-llama-archs.cpp
            echo \"tests/test-llama-archs.cpp from \$(git -C $M/rig/src rev-parse ${TEST_FROM:-})\" >> $M/cpu-$T/COMMIT
        fi
        cd $M/cpu-$T && (setsid nohup flock $LOCK docker run --rm --label pascal-quants=1 --label c898=1 --cpu-shares=256 \
            --network none -v $M:$M -e CCACHE_DIR=$M/cpu-ccache -e CCACHE_BASEDIR=$M/cpu-$T $IMG bash -c '
            cmake -S $M/cpu-$T/src -B $M/cpu-$T/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $M/cpu-$T/cmake.log 2>&1 &&
            nice -n 19 cmake --build $M/cpu-$T/build -j 8 -- -k > $M/cpu-$T/build.log 2>&1; echo rc=\$? > $M/cpu-$T/RC' \
            > $M/logs/cpu-$T.out 2>&1 < /dev/null &) ; cat $M/cpu-$T/COMMIT; echo started"
    ;;
test)
    T=${2:?tag}
    B=$M/cpu-$T/build/bin/test-llama-archs
    PLAIN=--mtp-plain-attn; [ "$T" = base ] && PLAIN=
    LIST=${ONLY:-"--mtp-kv-only sw --mtp-graph-keep --mtp-replan --mtp-catchup-overlap $PLAIN"}
    timeout 3600 $Q "mkdir -p $M/test && flock $LOCK docker run --rm --label pascal-quants=1 --label c898=1 --cpu-shares=256 \
        --network none -v $M:$M -w $M/test $IMG bash -c '
        for t in $LIST; do
            if [ \$t = sw ]; then f=$T-kvonly-sw.txt; env $SW nice -n 19 $B -a qwen4exp -s 1 --mtp-kv-only > \$f 2>&1
            else f=$T-\${t#--}.txt; nice -n 19 $B -a qwen4exp -s 1 \$t > \$f 2>&1; fi
            echo rc=\$? >> \$f
        done'; grep -aH -E 'Summary|^rc=' $M/test/$T-*.txt"
    ;;
archs)
    timeout 3000 $Q "mkdir -p $M/test && flock $LOCK docker run --rm --label pascal-quants=1 --label c898=1 --cpu-shares=256 \
        --network none -v $M:$M -w $M/test $IMG bash -c 'for T in base new; do
            timeout 1200 nice -n 19 $M/cpu-\$T/build/bin/test-llama-archs -s 1 > $M/test/\$T-archs-all.txt 2>&1
            echo rc=\$? >> $M/test/\$T-archs-all.txt; done'
        cd $M/test; tail -n 3 base-archs-all.txt new-archs-all.txt
        diff <(tail -n +2 base-archs-all.txt) <(tail -n +2 new-archs-all.txt) && echo 'same after line 1'"
    ;;
compare)
    timeout 60 $Q "cd $M/test && for f in mtp-kv-only kvonly-sw; do echo \"== \$f\"
        diff <(grep -a '^|' base-\$f.txt) <(grep -a '^|' new-\$f.txt) && echo 'the same lines'; done"
    ;;
rig)
    REF=${2:?ref}; T=${3:?tag}
    cat "$REPO/scripts/build_llamacpp_docker.sh" | timeout 60 $Q "mkdir -p $M/repo && cat > $M/repo/build_llamacpp_docker.sh && md5sum $M/repo/build_llamacpp_docker.sh"
    timeout 60 $Q "rm -f $M/rig/RC-$T; cd $M && (setsid nohup bash -c 'flock $LOCK env WORK=$M/rig TESTS=1 JOBS=8 \
        EXTRA_TARGETS=\"${EXTRA_TARGETS:-}\" \
        DOCKER_ARGS=\"--label pascal-quants=1 --label c898=1 --cpu-shares=256${HF_UI_VERSION:+ -e HF_UI_VERSION=$HF_UI_VERSION}\" \
        bash $M/repo/build_llamacpp_docker.sh $REF; echo rc=\$? > $M/rig/RC-$T' > $M/logs/rig-$T.log 2>&1 < /dev/null &) ; echo started"
    ;;
wait)
    F=${2:?file}
    case $F in cpu-*/RC) L=$M/${F%/RC}/build.log ;; *) L=$M/logs/rig-${F#rig/RC-}.log ;; esac
    timeout 7200 $Q "while [ ! -f $M/$F ]; do sleep 20; done; cat $M/$F
        grep -aE 'error|undefined reference' $L | grep -v Werror | sort -u | head -n 60; tail -n 16 $L"
    ;;
status)
    timeout 60 $Q "ls $M/rig/RC-* $M/cpu-*/RC 2>/dev/null; docker ps --filter label=c898=1 --format '{{.Names}} {{.Status}}'"
    ;;
stop)
    timeout 60 $Q "docker ps -q --filter label=c898=1 | xargs -r docker stop"
    ;;
*)
    echo "seed | cpu REF T | test T | compare | archs | rig REF T | wait FILE | status | stop"; exit 1
    ;;
esac
