#!/usr/bin/env bash
# Q88 v2-next exactness, 2026-10-24 (correctness only, nothing timed): branch pascal-p106-v2-next (combined + persistent MoE kernel +
# the pipelined work), an FA-on test build (BUILD), two GPUs (GPUS=a,b), Pascal kernels forced, MoE down rescale off
# at one token (tg, as the rig runs). Per model:
#   1. llama-perplexity -ub 1: the persistent kernel's self-check per graph evaluation (kernel forced: =2)
#   2. the logits of 2 x 128 one-token steps with the kernel off (=0) and on (=2), CUDA graphs on: byte for byte
#   3. (K2h-r16) greedy llama-completion, 4 prompts x 128 tokens, off against on
#   4. (K2h-r16) CUDA graphs over one completion, off against on
#   5. (K2h-r16) llama-pipespec: plain off, plain on, pipe on at depth 3 and 4: tokens and per-token logit hashes
# Ran on two V100s through the compute_61 PTX. Models: Flint K2h-r16 and G3 (quants/experiments.md, Q113 and Q73).
# Usage: BUILD=<test build out dir> MODELS=<dir> TEXT=<wiki.test.raw> WORK=<dir> GPUS=a,b [PROMPTS=<file>] t5.sh
set -u
GPUS=${GPUS:?GPUS=a,b}
BUILD=$(realpath "${BUILD:?BUILD=<out dir of scripts/build_llamacpp_tests_docker.sh>}")
MODELS=$(realpath "${MODELS:?MODELS=<dir with flint-q113-K2h-r16.gguf and flint-q73-G3.gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wikitext-2 wiki.test.raw>}")
PROMPTS=${PROMPTS:-$(cd "$(dirname "$0")/../../../.." && pwd)/quants/gptq_head/prompts.txt}
K=${WORK:?WORK=<work dir>}/e2e-next; mkdir -p $K/cuda-cache; rm -f $K/*.log $K/*.txt $K/*.kld # the PTX JIT cache stays
dock() { # <space-separated VAR=value list or -> <binary> <args...>
    local envs=() e
    [ "$1" != - ] && for e in $1; do envs+=(-e "$e"); done
    shift
    docker run --rm --gpus "\"device=$GPUS\"" -v "$BUILD:/b:ro" -v "$K:/k" -v "$MODELS:/m:ro" -v "$(dirname "$TEXT"):/t:ro" \
        -e LD_LIBRARY_PATH=/b/cuda12-libs -e GGML_CUDA_MMVQ_PASCAL=2 \
        -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e CUDA_CACHE_PATH=/k/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
        "${envs[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@"
}
ON="GGML_CUDA_PERSISTENT_MOE=2"; OFF="GGML_CUDA_PERSISTENT_MOE=0"
for spec in K2h-r16:/m/flint-q113-K2h-r16.gguf G3:/m/flint-q73-G3.gguf; do
    m=${spec%%:*}; M=${spec#*:}
    echo "##### $m"
    PPL="/b/build/bin/llama-perplexity -m $M -f /t/$(basename "$TEXT") -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on"
    echo "=== 1. self-check, -ub 1, 2 x 128 tokens (count: device, layers run, compared, differed)"
    dock "$ON GGML_CUDA_PERSISTENT_MOE_CHECK=1" $PPL -lv 4 > $K/$m-check.log 2>&1 || echo "check run: exit status $?"
    grep -oE "CUDA[0-9]+: persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed \(.*\)" $K/$m-check.log \
        | sed -E "s/persistent MoE kernel ran //; s/ compared with the per-op path,//" | sort | uniq -c
    grep -oE "Final estimate.*" $K/$m-check.log
    echo "=== 2. logits, kernel off against on (CUDA graphs on)"
    for v in off:"$OFF" on:"$ON"; do
        tag=${v%%:*}; envs=${v#*:}
        dock "$envs" $PPL --kl-divergence-base /k/$m-$tag.kld > $K/$m-ppl-$tag.log 2>&1 || echo "ppl $tag: exit status $?"
        echo "$tag: $(grep -oE 'Final estimate.*' $K/$m-ppl-$tag.log)"
    done
    cmp -s $K/$m-off.kld $K/$m-on.kld && echo "logits: on identical to off ($(stat -c %s $K/$m-off.kld) bytes)" || echo "logits: on DIFFERS from off"
    rm -f $K/$m-off.kld $K/$m-on.kld
    [ $m = G3 ] && continue
    echo "=== 3. llama-completion greedy, 4 prompts x 128 tokens"
    i=0
    while IFS= read -r prompt; do
        i=$((i + 1))
        for v in off:"$OFF" on:"$ON"; do
            tag=${v%%:*}; envs=${v#*:}
            dock "$envs" /b/build/bin/llama-completion -m $M -p "$(printf '%b' "$prompt")" -n 128 --temp 0 --seed 1 -ngl 99 \
                -fa on -c 2048 -no-cnv --no-warmup > $K/$m-p$i-$tag.txt 2> /dev/null || echo "prompt $i $tag: exit status $?"
        done
        n=$(grep -c . $K/$m-p$i-off.txt)
        cmp -s $K/$m-p$i-off.txt $K/$m-p$i-on.txt && echo "prompt $i: on identical to off ($(wc -c < $K/$m-p$i-off.txt) bytes)" \
            || echo "prompt $i: on DIFFERS from off"
    done < <(head -4 "$PROMPTS")
    echo "=== 4. CUDA graphs over one completion (count: event)"
    for v in off:"$OFF" on:"$ON"; do
        tag=${v%%:*}; envs=${v#*:}
        echo "-- $tag"
        dock "$envs GGML_CUDA_GRAPH_DEBUG=1" /b/build/bin/llama-completion -m $M -p "The history of the Roman aqueducts" -n 64 \
            --temp 0 --seed 1 -ngl 99 -fa on -c 2048 -no-cnv --no-warmup -lv 5 2>&1 \
            | grep -oE "CUDA Graph id [0-9]+ reused|CUDA graph warmup (complete|reset)" | sed "s/id [0-9]*/id N/" | sort | uniq -c
    done
    echo "=== 5. llama-pipespec"
    for v in plain-off:plain:3:"$OFF" plain-on:plain:3:"$ON" pipe3-on:pipe:3:"$ON" pipe4-on:pipe:4:"$ON"; do
        IFS=: read -r tag mode depth envs <<< "$v"
        dock "$envs" /b/build/bin/llama-pipespec -m $M -ngl 99 -fa on -c 2048 --temp 0 \
            -p "The history of the Roman aqueducts begins in 312 BC, when" -n 96 \
            --pipe-mode $mode --pipe-depth $depth --pipe-guess cont --pipe-out /k/$m-pipespec-$tag.txt > $K/$m-pipespec-$tag.log 2>&1 \
            || echo "pipespec $tag: exit status $?"
    done
    echo "pipespec tokens: $(wc -l < $K/$m-pipespec-plain-off.txt)"
    for tag in plain-on pipe3-on pipe4-on; do
        cmp -s $K/$m-pipespec-plain-off.txt $K/$m-pipespec-$tag.txt && echo "pipespec: $tag identical to plain-off (tokens and logit hashes)" \
            || echo "pipespec: $tag DIFFERS from plain-off"
    done
done
echo DONE
