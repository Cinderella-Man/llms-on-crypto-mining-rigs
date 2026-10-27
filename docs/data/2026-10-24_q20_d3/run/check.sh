#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3 checks on a CUDA machine, not the rig (correctness only, nothing timed). GPUS=a[,b] (docker device list).
#   STEPS='1 2 3 4' (default all):
#   1. llama-pipe-drafter-check on every case (cases.sh): the C++ head against drafter.Drafter.forward, the embedding
#      table on the device and in host memory, natively and through the compute_61 PTX (the rig's kernels)
#   2. check_identity.sh with a Flint file and random-weight heads of the real shape (A, A fused, B) at depths 1-4: every
#      pipelined job, staircase included, equal to plain decoding (tokens, per-token logits hashes, KV state, probe);
#      T 0 greedy on the device, and T 1 (makers' sampling) on the device with position-keyed draws
#   3. plain decoding with the drafter loaded against a process without it: the taps must not change the logits
#   4. check_server.sh: llama-server --pipe-guess staircase at depths 3 and 5 against the server without guesses
# GPU 4 (the shared check card, 32 GB) by default; every container takes the card's lock file (LOCK) and stays
# under 30 minutes. The model (M, default the 23.1 GB imatU-Q4_0 finalist: any Flint file checks exactness, this one
# leaves room on one card) runs on VDEV=3 virtual devices of that card, so the taps cross devices.
set -u
GPUS=${GPUS:-4}
STEPS=${STEPS:-1 2 3 4}
VDEV=${VDEV-3}
LOCK=${LOCK:-/tmp/gpu4.lock}
D=${D:?D=<the D3 check dir, see ../README.md>}
MODELS=${MODELS:?MODELS=<directory of the Flint GGUFs>}
K=$D/check; mkdir -p $K/cuda-cache
M=${M:-$MODELS/flint-imatU-Q4_0-p4648b.gguf}
dock() { # <space-separated VAR=value list or -> <command...>
    local envs=() e
    [ "$1" != - ] && for e in $1; do envs+=(-e "$e"); done
    shift
    flock "$LOCK" docker run --rm --label pascal-quants=1 --gpus "\"device=$GPUS\"" --network host -v $D:$D \
        -v $MODELS:$MODELS:ro -e MODELS=$MODELS -e LD_LIBRARY_PATH=$D/out/cuda12-libs \
        -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e CUDA_CACHE_PATH=$K/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
        ${VDEV:+-e GGML_CUDA_DEVICES=$VDEV} "${envs[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@"
}
B=$D/out/build/bin
PTX="CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2"

for step in $STEPS; do
case $step in
1)
    echo "=== 1. the C++ head against PyTorch"
    # one container for every case (one hold of the lock, a few minutes)
    VDEV= dock - bash -c '
        for c in '$D'/cases/*/; do
            n=$(basename $c)
            for tag in native ptx; do
                for t in device host; do
                    tol=0.05; case $n in *q8*) tol=0.25 ;; esac
                    if [ $tag = ptx ]; then export CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2; else unset CUDA_FORCE_PTX_JIT GGML_CUDA_MMVQ_PASCAL; fi
                    '$B'/llama-pipe-drafter-check $c/drafter.gguf $c/case.bin --tol $tol $([ $t = host ] && echo --host-table) \
                        > '$K'/num-$n-$tag-$t.log 2>&1
                    echo "$n $tag $t: $(grep -o "DRAFTER_CHECK.*" '$K'/num-$n-$tag-$t.log || tail -3 '$K'/num-$n-$tag-$t.log)"
                done
            done
        done'
    ;;
2)
    echo "=== 2. check_identity.sh with K2h-r16 and random heads"
    for h in ${HEADS:-rand-A rand-A-fuse rand-B}; do
        for s in ${SAMPLINGS:-t0 t1}; do
            if [ $s = t0 ]; then samp="--temp 0"; bg=--pipe-backend-greedy; else samp="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"; bg=--pipe-backend-sample; fi
            for e in host:- gpu:LLAMA_INPUT_ON_GPU=1; do
                [ $h != rand-A ] && [ ${e%%:*} = gpu ] && continue
                tag=$h-$s-${e%%:*}
                dock "${e#*:}" env BIN=$B MODEL=$M ARGS="-ngl 99 -fa on -c 4096 -ub 64 $bg" SAMPLING="$samp" \
                    PROMPT=$D/pipespec/prompts/edit.txt N=${N:-192} DEPTHS="${DEPTHS:-1 2 3 4}" MISSES="0.5" DRAFT=$D/heads/$h.gguf \
                    OUT=$K/id-$tag SPEED=0 TIMEOUT=1500 bash $D/pipespec/check_identity.sh > $K/id-$tag.txt 2>&1
                echo "--- $tag (exit $?)"; grep -E "^(plain|stair|cont4|oracle)" $K/id-$tag.txt
                grep -o "PIPESPEC name=stair[^ ]* .*" $K/id-$tag/results.txt | grep -o "name=[^ ]*\|draft_guesses=[0-9]*\|draft_hits=[0-9]*\|copy_guesses=[0-9]*" | paste -sd' ' | fold -w 200
                grep -m1 -o "staircase drafter .*" $K/id-$tag/run.log
            done
        done
    done
    ;;
2k)
    echo "=== 2k. check_identity.sh with K2h-r16 (29.4 GB) alone on the card, one device, fused head with an f16 output head"
    VDEV= dock - env BIN=$B MODEL=$MODELS/flint-q113-K2h-r16.gguf \
        ARGS="-ngl 99 -fa on -c 2048 -ub 32 --pipe-backend-sample" SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42" \
        PROMPT=$D/pipespec/prompts/edit.txt N=128 DEPTHS="1 2 3" MISSES="0.5" DRAFT=$D/heads/rand-A-fuse-h16.gguf \
        OUT=$K/id-k2h SPEED=0 TIMEOUT=1500 bash $D/pipespec/check_identity.sh > $K/id-k2h.txt 2>&1
    echo "exit $?"; grep -E "^(plain|stair|cont4|oracle)|only|TIMEOUT" $K/id-k2h.txt; grep -m3 -iE "out of memory|alloc.*failed" $K/id-k2h/run.log
    ;;
2c)
    echo "=== 2c. check_identity.sh on the CPU (no GPU: the drafter on the CPU device, host sampling)"
    docker run --rm --label pascal-quants=1 -v $D:$D -v $MODELS:$MODELS:ro -e MODELS=$MODELS \
        -e LD_LIBRARY_PATH=$D/out/cuda12-libs -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e CUDA_VISIBLE_DEVICES= \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 env BIN=$B MODEL=$M ARGS="-ngl 0 -t 8 -fa on -c 2048 -ub 32" SAMPLING="--temp 0" \
        PROMPT=$D/pipespec/prompts/rewrite.txt N=64 DEPTHS="2 3" MISSES="0.5" DRAFT=$D/heads/rand-A-fuse.gguf \
        OUT=$K/id-cpu SPEED=0 TIMEOUT=5400 bash $D/pipespec/check_identity.sh > $K/id-cpu.txt 2>&1
    echo "exit $?"; grep -E "^(plain|stair|cont4|oracle)" $K/id-cpu.txt
    ;;
3)
    echo "=== 3. plain decoding with and without the drafter loaded"
    for x in without with; do
        printf 'name=plain mode=plain prompt=%s chat=1 n=128 out=%s/taps-%s.tok\nname=cont3 mode=pipe depth=3 guess=cont copy_min=4 prompt=%s chat=1 n=128\n' \
            $D/pipespec/prompts/chat.txt $K $x $D/pipespec/prompts/chat.txt > $K/taps-$x.jobs
        dock - $B/llama-pipespec -m $M -ngl 99 -fa on -c 4096 -ub 64 --temp 0 --pipe-backend-greedy --pipe-slots 8 \
            $([ $x = with ] && echo --pipe-draft $D/heads/rand-A-fuse.gguf) --pipe-jobs $K/taps-$x.jobs > $K/taps-$x.log 2>&1
        echo "$x: $(grep -o 'PIPESPEC name=[^ ]* .*' $K/taps-$x.log | grep -o 'name=[^ ]*\|tokens=[^ ]*\|logits=[^ ]*\|state=[^ ]*' | paste -sd' ')"
    done
    ;;
4)
    echo "=== 4. llama-server --pipe-guess staircase"
    dock - bash -c "apt-get update -qq && apt-get install -y -qq python3 curl > /dev/null 2>&1; \
        env BIN=$B MODEL=$M ARGS='-ngl 99 -fa on -c 8192 -ub 64 --pipe-draft $D/heads/rand-A-fuse.gguf' DEPTHS='3 5' \
        GUESSES=staircase FLOORS= PROMPTS='$D/pipespec/prompts/chat.txt $D/pipespec/prompts/rewrite.txt' NTOK=160 PORT=18631 \
        OUT=$K/server bash $D/pipespec/check_server.sh" > $K/server.txt 2>&1
    echo "exit $?"; tail -25 $K/server.txt
    grep -h "drafter" $K/server/*.log 2>/dev/null | grep -o "pipelined = .*" | sort | uniq -c | head
    ;;
esac
done
