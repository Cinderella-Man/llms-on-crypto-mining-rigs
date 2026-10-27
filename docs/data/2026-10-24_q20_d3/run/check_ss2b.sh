#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: the long version of check_ss2.sh on the same cards (0, 1, 6):
# check_server.sh at depths 1 and 3, staircase with the continuation store and the random head, 512-token answers to
# chat, code, edit and rewrite (the rewrite answer repeats the prompt's file, so the copy lookup and the store guess
# long stretches and commits pile up between head launches: the case that crashed launch() before 460160e93). Every
# request must equal the depth-1 no-guess reference, and at T 0 the plain server. ~20 min.
set -u
STORE_SRC=${STORE_SRC:?STORE_SRC=<.pstore built on the strict data list (the old 13.8 M-token store was excluded data)>}
D=${D:?D=<the D3 check dir, see ../README.md>}
MODELS=${MODELS:?MODELS=<directory of the Flint GGUFs>}
S=$D/rc4ss
K=$S/check2b; mkdir -p $K/cuda-cache
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-checkss2b \
    --gpus '"device=0,1,6"' --network host -v $D:$D -v $MODELS:$MODELS:ro -e MODELS=$MODELS \
    -e LD_LIBRARY_PATH=$S/out/cuda12-libs \
    -e CUDA_CACHE_PATH=$K/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 -e D=$D -e STORE_SRC=$STORE_SRC -e S=$S -e K=$K \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq python3 curl > /dev/null 2>&1
    PS=$D/pipespec
    M=$MODELS/flint-imatU-Q4_0-p4648b.gguf
    ST=$STORE_SRC
    echo "=== server, long answers: staircase with the store, depths 1 and 3 $(date +%T)"
    env BIN=$S/out/build/bin MODEL=$M ARGS="-ngl 99 -fa on -c 8192 -ub 64 -lv 4 --pipe-draft $D/heads/rand-G-h16.gguf --pipe-store $ST" \
        DEPTHS="1 3" GUESSES="staircase" FLOORS= NTOK=512 PORT=18671 OUT=$K/server \
        PROMPTS="$PS/prompts/chat.txt $PS/prompts/code.txt $PS/prompts/edit.txt $PS/prompts/rewrite.txt" \
        bash $PS/check_server.sh > $K/server.txt 2>&1
    echo "exit $?"; grep -vE "^\s*$" $K/server.txt | tail -12
    for c in d1-staircase d3-staircase; do
        echo "--- $c: requests $(grep -c "pipelined = " $K/server/$c.log)"
        grep -o "pipelined = .*" $K/server/$c.log | sed "s/ \+/ /g"
    done
    echo "=== done $(date +%T)"
'
