#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: llama-server of pascal-p106-v2-rc4-stairstore (rc4 + staircase with the store) on GPU 4 under its lock, one hold
# (~25 min), correctness only. check_server.sh at depths 1 and 3 with the continuation store and the depth-general
# random head in ARGS: cont (copy, continuations, store, continuations of any context) and staircase (copy,
# continuations, store, then the head) must both return the reference's text (depth 1, no guess) for every request, and
# at T 0 the plain server's. flint-imatU-Q4_0 on 3 virtual devices of the card, so the head's taps cross devices. The
# guess counts per config show the order: the store's guesses in both, the head's only in staircase. Then a short run
# of rig_server.sh (the rig e2e script) with configs 0 3s 3ds 3d.
set -u
STORE_SRC=${STORE_SRC:?STORE_SRC=<.pstore built on the strict data list (the old 13.8 M-token store was excluded data)>}
D=${D:?D=<the D3 check dir, see ../README.md>}
MODELS=${MODELS:?MODELS=<directory of the Flint GGUFs>}
S=$D/rc4ss
K=$S/check; mkdir -p $K/cuda-cache
flock "${LOCK:-/tmp/gpu4.lock}" docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-checkss \
    --gpus '"device=4"' --network host -v $D:$D -v $MODELS:$MODELS:ro -e MODELS=$MODELS \
    -e LD_LIBRARY_PATH=$S/out/cuda12-libs -e GGML_CUDA_DEVICES=3 \
    -e CUDA_CACHE_PATH=$K/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 -e D=$D -e STORE_SRC=$STORE_SRC -e S=$S -e K=$K \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq python3 curl > /dev/null 2>&1
    PS=$D/pipespec
    echo "=== server: cont and staircase with the store, depths 1 and 3 $(date +%T)"
    env BIN=$S/out/build/bin MODEL=$MODELS/flint-imatU-Q4_0-p4648b.gguf \
        ARGS="-ngl 99 -fa on -c 8192 -ub 64 -lv 4 --pipe-draft $D/heads/rand-G-h16.gguf --pipe-store $STORE_SRC" \
        DEPTHS="1 3" GUESSES="cont staircase" FLOORS= PROMPTS="$PS/prompts/chat.txt $PS/prompts/code.txt $PS/prompts/edit.txt" \
        NTOK=192 PORT=18641 OUT=$K/server bash $PS/check_server.sh > $K/server.txt 2>&1
    echo "exit $?"; tail -20 $K/server.txt
    for c in d1-cont d1-staircase d3-cont d3-staircase; do
        echo "--- $c"; grep -m1 -o "staircase drafter .*" $K/server/$c.log | cut -c1-250
        grep -m1 -o "continuation store .*" $K/server/$c.log
        grep -o "pipelined = .*" $K/server/$c.log | sed "s/ \+/ /g" | sort | uniq -c | head -12
    done
    echo "=== rig_server.sh smoke (the rig e2e script): configs 0 3s 3ds 3d, short answers $(date +%T)"
    rm -rf $PS/out/ss-smoke
    env BIN=$S/out/build/bin MODEL=$MODELS/flint-imatU-Q4_0-p4648b.gguf TAG=ss-smoke \
        GPUS=0 DEPTHS="0 3s 3ds 3d" GUESS=cont CTX=8192 MAX_TOKENS=96 TOOL_MAX_TOKENS=192 SEED=1 PORT=18642 \
        STORE=$STORE_SRC DRAFT=$D/heads/rand-G-h16.gguf \
        EXTRA_ARGS="-lv 4" bash $PS/rig_server.sh > $K/smoke.out 2>&1
    echo "exit $?"; tail -14 $K/smoke.out
    grep -h -m1 -o "staircase drafter .*" $PS/out/ss-smoke/server-d3ds.log | cut -c1-250
    echo "=== done $(date +%T)"
'
