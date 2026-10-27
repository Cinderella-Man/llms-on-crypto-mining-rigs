#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3 after window correction (fork c3a324ba0): one hold of the GPU 4 lock (~25 min), correctness only:
#   a. the cases3/ cases on the card (device table): the C++ head against drafter.Drafter.forward
#   b. check_identity.sh, depth-general random head rand-G-h16, flint-imatU-Q4_0 on 3 virtual devices of the card (the
#      split's window has 2 rows, so depths 3 and 5 put complete tokens in flight: final-state rows), T 1 sampled on the
#      device, embeddings on the host
#   c. check_identity.sh, K2h-r16 (29.4 GB) alone on the card (one device: a one-row window at the final state), the
#      fused random head with an f16 output head (complete tokens in flight take the committed rows' fused rule)
set -u
D=${D:?D=<the D3 check dir, see ../README.md>}
MODELS=${MODELS:?MODELS=<directory of the Flint GGUFs>}
K=$D/check; mkdir -p $K/cuda-cache
flock "${LOCK:-/tmp/gpu4.lock}" docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-check3 \
    --gpus '"device=4"' --network host -v $D:$D -v $MODELS:$MODELS:ro -e MODELS=$MODELS \
    -e LD_LIBRARY_PATH=$D/out/cuda12-libs -e LLAMA_NO_MOE_DOWN_RESCALE=tg \
    -e CUDA_CACHE_PATH=$K/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 -e D=$D -e K=$K \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    B=$D/out/build/bin; C=$D/cases3; PS=$D/pipespec
    SAMP="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"
    echo "=== a. cases on the card $(date +%T)"
    for c in A-f16:9,8,8,8,8,8:: A-fuse-f16:9,8,8,8,8,8:: A-fuse-q8:9,8,8,8,8,8::0.25 B-f16:9,8,8,8,8,8:pre: C-f16:9,8,8,8,8,8:: \
             R1real-f16:9,8,8,8,8,8:: G-c6:9,8,8,8,8,8:: G-c9:6,6,6,6,5,5,5,5,5:: G-c9l:6,6,6,6,6,6,6,5,2:: \
             G-c12:5,4,4,4,4,4,4,4,4,4,4,4:: G-c4:13,12,12,12:: G-c3:17,16,16:: G-c6pre:9,8,8,8,8,8:pre: \
             G-c9-q8:6,6,6,6,5,5,5,5,5::0.25; do
        IFS=: read -r n split kind tol <<< "$c"
        $B/llama-pipe-drafter-check $C/$n/drafter.gguf $C/$n/case.bin --split $split ${kind:+--kind $kind} --tol ${tol:-0.05} \
            > $K/num3-gpu-$n.log 2>&1
        echo "$n: $(grep -o "DRAFTER_CHECK.*" $K/num3-gpu-$n.log || tail -2 $K/num3-gpu-$n.log)"
    done
    echo "=== b. identity, rand-G-h16, imatU-Q4_0 on 3 virtual devices, T 1, host embeddings $(date +%T)"
    env GGML_CUDA_DEVICES=3 BIN=$B MODEL=$MODELS/flint-imatU-Q4_0-p4648b.gguf \
        ARGS="-ngl 99 -fa on -c 4096 -ub 64 --pipe-backend-sample" SAMPLING="$SAMP" PROMPT=$PS/prompts/edit.txt N=128 \
        DEPTHS="1 3 5" MISSES="0.5" DRAFT=$D/heads/rand-G-h16.gguf OUT=$K/id3-G SPEED=0 TIMEOUT=900 \
        bash $PS/check_identity.sh > $K/id3-G.txt 2>&1
    echo "exit $?"; grep -E "^(plain|stair|cont4|oracle)|only|TIMEOUT|no plain" $K/id3-G.txt
    grep -m1 -o "staircase drafter .*" $K/id3-G/run.log
    echo "=== c. identity, K2h-r16 alone on the card, rand-A-fuse-h16, T 1 $(date +%T)"
    env BIN=$B MODEL=$MODELS/flint-q113-K2h-r16.gguf \
        ARGS="-ngl 99 -fa on -c 2304 -ub 32 --pipe-backend-sample" SAMPLING="$SAMP" PROMPT=$PS/prompts/edit.txt N=128 \
        DEPTHS="1 3" MISSES="0.5" DRAFT=$D/heads/rand-A-fuse-h16.gguf OUT=$K/id3-k2h SPEED=0 TIMEOUT=900 \
        bash $PS/check_identity.sh > $K/id3-k2h.txt 2>&1
    echo "exit $?"; grep -E "^(plain|stair|cont4|oracle)|only|TIMEOUT|no plain" $K/id3-k2h.txt
    grep -m1 -o "staircase drafter .*" $K/id3-k2h/run.log; grep -m3 -iE "out of memory|alloc.*failed" $K/id3-k2h/run.log
    echo "=== done $(date +%T)"
'
