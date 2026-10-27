#!/usr/bin/env bash
# Q88 test 2, 2026-10-24 (correctness only, nothing timed): G3 and K2h-r16 end to end on two GPUs (GPUS=a,b), Pascal kernels forced
# (the rig's path), MoE down rescale off at one token (LLAMA_NO_MOE_DOWN_RESCALE=tg, as the rig runs). Per model:
#   1. llama-perplexity -ub 1 (every token a one-token step): the self-check's counts per graph evaluation, then the
#      logits written with the persistent kernel off and on (CUDA graphs on), byte for byte
#   2. llama-completion, greedy, 128 tokens of each of the first 2 Q92 prompts: off against on, byte for byte (K2h-r16)
#   3. CUDA graphs: replays and warmups over one completion, off against on
#   4. llama-pipespec: plain and pipelined (depth 3) decoding with the kernel on, against plain with it off (tokens and
#      per-token logit hashes)
# Build: scripts/build_llamacpp_tests_docker.sh on fork branch pascal-p106-v2-persistent. Models: Flint K2h-r16 and G3
# (quants/experiments.md, Q113 and Q73). Prompts: the first lines of quants/gptq_head/prompts.txt.
# Usage: BUILD=<test build out dir> MODELS=<dir> TEXT=<wiki.test.raw> WORK=<dir> GPUS=a,b [PROMPTS=<file>] t2.sh
set -u
GPUS=${GPUS:?GPUS=a,b}
BUILD=$(realpath "${BUILD:?BUILD=<out dir of scripts/build_llamacpp_tests_docker.sh>}")
MODELS=$(realpath "${MODELS:?MODELS=<dir with flint-q113-K2h-r16.gguf and flint-q73-G3.gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wikitext-2 wiki.test.raw>}")
PROMPTS=${PROMPTS:-$(cd "$(dirname "$0")/../../../.." && pwd)/quants/gptq_head/prompts.txt}
K=${WORK:?WORK=<work dir>}/e2e; mkdir -p $K/cuda-cache; rm -f $K/*.log $K/*.txt $K/*.kld # the compute_61 PTX is JIT-compiled once and cached across runs
dock() { # <space-separated VAR=value list or -> <binary> <args...>
    local envs=() e
    [ "$1" != - ] && for e in $1; do envs+=(-e "$e"); done
    shift
    docker run --rm --gpus "\"device=$GPUS\"" -v "$BUILD:/b:ro" -v "$K:/k" -v "$MODELS:/m:ro" -v "$(dirname "$TEXT"):/t:ro" \
        -e LD_LIBRARY_PATH=/b/cuda12-libs -e GGML_CUDA_MMVQ_PASCAL=2 \
        -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e CUDA_CACHE_PATH=/k/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
        "${envs[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@"
}
ON="GGML_CUDA_PERSISTENT_MOE=2"
for spec in K2h-r16:/m/flint-q113-K2h-r16.gguf G3:/m/flint-q73-G3.gguf; do
    m=${spec%%:*}; M=${spec#*:}
    echo "##### $m"
    PPL="/b/build/bin/llama-perplexity -m $M -f /t/$(basename "$TEXT") -c 128 --chunks 2 -b 128 -ub 1 -ngl 99 -fa on"
    echo "=== 1. llama-perplexity -ub 1, 2 x 128 tokens"
    dock "$ON GGML_CUDA_PERSISTENT_MOE_CHECK=1" $PPL -lv 4 > $K/$m-check.log 2>&1 || echo "check run: exit status $?"
    echo "-- self-check, per graph evaluation and device (count: device, layers run, compared, differed)"
    grep -oE "CUDA[0-9]+: persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed \(.*\)" $K/$m-check.log \
        | sed -E "s/persistent MoE kernel ran //; s/ compared with the per-op path,//" | sort | uniq -c
    grep -oE "Final estimate.*" $K/$m-check.log
    for v in off:- on:"$ON"; do
        tag=${v%%:*}; envs=${v#*:}
        dock "$envs" $PPL --kl-divergence-base /k/$m-$tag.kld > $K/$m-ppl-$tag.log 2>&1 || echo "ppl $tag: exit status $?"
        echo "$tag: $(grep -oE 'Final estimate.*' $K/$m-ppl-$tag.log)"
    done
    cmp -s $K/$m-off.kld $K/$m-on.kld && echo "logits: on identical to off ($(stat -c %s $K/$m-off.kld) bytes)" || echo "logits: on DIFFERS from off"
    rm -f $K/$m-off.kld $K/$m-on.kld
    [ $m = G3 ] && continue # G3: the self-check and the logits; the rest on K2h-r16 (time slot)
    echo "=== 2. llama-completion greedy, 2 prompts x 128 tokens"
    i=0
    while IFS= read -r prompt; do
        i=$((i + 1))
        for v in off:- on:"$ON"; do
            tag=${v%%:*}; envs=${v#*:}
            dock "$envs" /b/build/bin/llama-completion -m $M -p "$(printf '%b' "$prompt")" -n 128 --temp 0 --seed 1 -ngl 99 \
                -fa on -c 2048 -no-cnv --no-warmup > $K/$m-p$i-$tag.txt 2> /dev/null || echo "prompt $i $tag: exit status $?"
        done
        cmp -s $K/$m-p$i-off.txt $K/$m-p$i-on.txt && echo "prompt $i: on identical to off ($(wc -c < $K/$m-p$i-off.txt) bytes)" \
            || echo "prompt $i: on DIFFERS from off"
    done < <(head -2 "$PROMPTS")
    echo "=== 3. CUDA graphs over one completion (count: event)"
    for v in off:- on:"$ON"; do
        tag=${v%%:*}; envs=${v#*:}
        echo "-- $tag"
        dock "$envs GGML_CUDA_GRAPH_DEBUG=1" /b/build/bin/llama-completion -m $M -p "The history of the Roman aqueducts" -n 64 \
            --temp 0 --seed 1 -ngl 99 -fa on -c 2048 -no-cnv --no-warmup -lv 5 2>&1 \
            | grep -oE "CUDA Graph id [0-9]+ reused|CUDA graph warmup (complete|reset)" | sed "s/id [0-9]*/id N/" | sort | uniq -c
    done
    echo "=== 4. llama-pipespec"
    for v in plain-off:plain:- plain-on:plain:"$ON" pipe-on:pipe:"$ON"; do
        tag=${v%%:*}; rest=${v#*:}; mode=${rest%%:*}; envs=${rest#*:}
        dock "$envs" /b/build/bin/llama-pipespec -m $M -ngl 99 -fa on -c 2048 --temp 0 \
            -p "The history of the Roman aqueducts begins in 312 BC, when" -n 96 \
            --pipe-mode $mode --pipe-depth 3 --pipe-guess cont --pipe-out /k/$m-pipespec-$tag.txt > /dev/null 2>&1 \
            || echo "pipespec $tag: exit status $?"
    done
    echo "pipespec tokens: $(wc -l < $K/$m-pipespec-plain-off.txt)"
    for tag in plain-on pipe-on; do
        cmp -s $K/$m-pipespec-plain-off.txt $K/$m-pipespec-$tag.txt && echo "pipespec: $tag identical to plain-off (tokens and logit hashes)" \
            || echo "pipespec: $tag DIFFERS from plain-off"
    done
done
echo DONE
