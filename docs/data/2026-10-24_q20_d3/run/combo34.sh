#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: check.sh steps 3 and 4 in one container (one hold of the GPU 4 lock): plain decoding with and without the
# drafter loaded (the taps must not change the logits), then llama-server --pipe-guess staircase against the server
# without guesses (check_server.sh). Correctness only.
set -u
D=${D:?D=<the D3 check dir, see ../README.md>}
MODELS=${MODELS:?MODELS=<directory of the Flint GGUFs>}
K=$D/check
M=$MODELS/flint-imatU-Q4_0-p4648b.gguf
flock "${LOCK:-/tmp/gpu4.lock}" docker run --rm --label pascal-quants=1 --name pq-q20-d3-combo --gpus '"device=4"' --network host \
    -v $D:$D -v $MODELS:$MODELS:ro -e MODELS=$MODELS -e LD_LIBRARY_PATH=$D/out/cuda12-libs -e LLAMA_NO_MOE_DOWN_RESCALE=tg \
    -e GGML_CUDA_DEVICES=3 -e D=$D -e K=$K -e M=$M nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    B=$D/out/build/bin
    echo "=== 3. plain decoding with and without the drafter loaded"
    for x in without with; do
        printf "name=plain mode=plain prompt=%s chat=1 n=128 out=%s/taps-%s.tok\nname=cont3 mode=pipe depth=3 guess=cont copy_min=4 prompt=%s chat=1 n=128\n" \
            $D/pipespec/prompts/chat.txt $K $x $D/pipespec/prompts/chat.txt > $K/taps-$x.jobs
        $B/llama-pipespec -m $M -ngl 99 -fa on -c 4096 -ub 64 --temp 0 --pipe-backend-greedy --pipe-slots 8 \
            $([ $x = with ] && echo --pipe-draft $D/heads/rand-A-fuse-h16.gguf) --pipe-jobs $K/taps-$x.jobs > $K/taps-$x.log 2>&1
        echo "$x: $(grep -o "PIPESPEC name=[^ ]* .*" $K/taps-$x.log | grep -o "name=[^ ]*\| tokens=[^ ]*\|logits=[^ ]*\|state=[^ ]*" | paste -sd" ")"
    done
    echo "=== 4. llama-server --pipe-guess staircase"
    apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq python3 curl > /dev/null 2>&1
    BIN=$B MODEL=$M ARGS="-ngl 99 -fa on -c 8192 -ub 64 --pipe-draft $D/heads/rand-A-fuse-h16.gguf" DEPTHS="3 5" GUESSES=staircase \
        FLOORS= PROMPTS="$D/pipespec/prompts/chat.txt $D/pipespec/prompts/rewrite.txt" NTOK=160 PORT=18631 OUT=$K/server \
        bash $D/pipespec/check_server.sh > $K/server.txt 2>&1
    echo "check_server exit $?"; tail -25 $K/server.txt
    grep -h -o "pipelined = .*" $K/server/*.log 2>/dev/null | sort | uniq -c | head
'
