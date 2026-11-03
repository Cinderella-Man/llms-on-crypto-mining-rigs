#!/usr/bin/env bash
# Q550: pipelined speculation for Qwen3.8-Flash-Next (qwen4exp) must reproduce plain decoding exactly. Runs the
# token-equality checks of bench/pipespec/check_identity.sh (tokens, every token's logits, the sequence state with its
# recurrent part, a probe decoded afterwards) and llama-pipe-recurrent-check (forced stash copies and the rollbacks that
# read them, the token in flight dropped as llama-server's pipe_leave does), on
#   rand: a random-weight qwen4exp model from test-llama-archs (2 layers: a Gated DeltaNet layer with the n-gram
#         embedding's conv history, an attention layer with the sparse-attention indexer; hyper-connections; MoE);
#         on the CPU (8 threads) and on 2 virtual devices of one GPU;
#   real: a real Flash-Next file (e.g. Swift-1.5 GSQ-RCO Q2_0, 48 layers) on the CPU, over the GPUs given to the
#         container, and on 4 virtual devices over them.
# Correctness only: nothing here is timed.
#
# Usage (environment): BIN=<dir with llama-pipespec, llama-pipe-recurrent-check, test-llama-archs> PS=<bench/pipespec>
#   OUT=<dir> [MODEL=<real gguf, first shard>] [RAND=<random qwen4exp gguf, else saved to OUT/models>] [STEPS='rand-cpu rand-prc-cpu rand-gpu rand-seeded rand-prc-gpu real
#   real-seeded real-prc real-vdev4'] [N=128] [T=1500] bash checks.sh
# real-cpu (the real file on the CPU, 64 tokens, depths 1 and 3) is not in the default list: it needs ~70 GB of RAM.
set -uo pipefail
: "${BIN:?set BIN}" "${PS:?set PS}" "${OUT:?set OUT}"
STEPS=${STEPS:-rand-cpu rand-prc-cpu rand-gpu rand-seeded rand-prc-gpu real real-seeded real-prc real-vdev4}
N=${N:-128}
T=${T:-1500}
mkdir -p "$OUT"
has() { [[ " $STEPS " == *" $1 "* ]]; }
step() { echo "=== $1 $(date +%T)"; }
ci() { # <name> <env...>: check_identity.sh, its table into $OUT/<name>.log
    local name=$1; shift
    env "$@" BIN="$BIN" SPEED=0 OUT="$OUT/$name" TIMEOUT="$T" bash "$PS/check_identity.sh" > "$OUT/$name.log" 2>&1
    echo "$name: exit $?"
    grep -E "^(run|plain|pipe|oracle|copy|cont|simple|mapk|floor|stair)[- ]|only|TIMEOUT|exit|no plain" "$OUT/$name.log" | tail -32
}
prc() { # <name> <model> <args...>: llama-pipe-recurrent-check
    local name=$1 model=$2; shift 2
    timeout 600 "$BIN/llama-pipe-recurrent-check" -m "$model" "$@" > "$OUT/$name.log" 2>&1
    echo "$name: exit $? $(grep -o 'PIPE-RECURRENT-CHECK.*' "$OUT/$name.log")"
}

RAND=${RAND:-$OUT/models/qwen4exp-moe.gguf}
if has rand-cpu || has rand-prc-cpu || has rand-gpu || has rand-seeded || has rand-prc-gpu; then
    if [ ! -s "$RAND" ]; then
        step "random qwen4exp model (test-llama-archs, seed 1)"
        mkdir -p "$OUT/models"
        "$BIN/test-llama-archs" -a '^qwen4exp$' -s 1 -o "$OUT/models" > "$OUT/models/save.log" 2>&1
        echo "save: exit $?"; ls -la "$OUT/models"
    fi
    RARGS="-c 2048 -fa on"
fi

if has rand-cpu; then
    step "rand-cpu: random qwen4exp on the CPU, greedy, depths 1-4"
    ci rand-cpu MODEL="$RAND" CHAT=0 ARGS="-ngl 0 -t 8 $RARGS" N=$N DEPTHS="1 2 3 4"
fi
if has rand-gpu; then
    step "rand-gpu: random qwen4exp on 2 virtual devices (GDN layer | attention layer), greedy, depths 1-4"
    ci rand-gpu GGML_CUDA_DEVICES=2 MODEL="$RAND" CHAT=0 ARGS="-ngl 99 $RARGS" N=$N DEPTHS="1 2 3 4"
fi
if has rand-seeded; then
    step "rand-seeded: random qwen4exp, 2 virtual devices, T 1 seed 42, depths 1 and 3"
    ci rand-seeded GGML_CUDA_DEVICES=2 MODEL="$RAND" CHAT=0 ARGS="-ngl 99 $RARGS" N=$N DEPTHS="1 3" \
        SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"
fi
if has rand-prc-cpu; then
    step "rand-prc-cpu: llama-pipe-recurrent-check, random qwen4exp, CPU"
    prc rand-prc-cpu "$RAND" -f "$PS/prompts/rewrite.txt" -n 96 -ngl 0 -t 8 $RARGS
fi
if has rand-prc-gpu; then
    step "rand-prc-gpu: llama-pipe-recurrent-check, random qwen4exp, 2 virtual devices"
    GGML_CUDA_DEVICES=2 prc rand-prc-vdev2 "$RAND" -f "$PS/prompts/rewrite.txt" -n 96 -ngl 99 $RARGS
fi

AARGS="-ngl 99 -fa on -c 4096"
if has real-cpu; then
    step "real-cpu: $(basename "${MODEL:-none}") on the CPU (8 threads), greedy, 64 tokens, depths 1 and 3"
    ci real-cpu MODEL="$MODEL" ARGS="-ngl 0 -t 8 -fa on -c 4096" N=64 DEPTHS="1 3" MISSES="0 0.5"
fi
if has real; then
    step "real: $(basename "${MODEL:-none}") over the container's GPUs, greedy, depths 1-3"
    ci real MODEL="$MODEL" ARGS="$AARGS" N=$N DEPTHS="1 2 3"
fi
if has real-seeded; then
    step "real-seeded: T 0.6 top-k 20 top-p 0.95 seed 42 (the model card's sampling), chat prompt, depths 1 and 3"
    ci real-seeded MODEL="$MODEL" ARGS="$AARGS" N=$N DEPTHS="1 3" PROMPT="$PS/prompts/chat.txt" \
        SAMPLING="--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"
fi
if has real-prc; then
    step "real-prc: llama-pipe-recurrent-check on the real file"
    prc real-prc "$MODEL" -f "$PS/prompts/chat.txt" -n 96 $AARGS
fi
if has real-vdev4; then
    step "real-vdev4: 4 virtual devices over the GPUs (4 stages), greedy, depths 2 and 4"
    ci real-vdev4 GGML_CUDA_DEVICES=4 MODEL="$MODEL" ARGS="$AARGS" N=$N DEPTHS="2 4" MISSES="0 0.5"
fi
step "checks ended"
