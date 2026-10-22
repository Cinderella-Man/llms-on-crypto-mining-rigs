#!/usr/bin/env bash
# A/B test of two runtime switches of llama.cpp's CUDA backend that cost nothing to try (no rebuild, no new model file):
#
#   GGML_CUDA_GRAPH_OPT=1       Inside a CUDA graph, runs the Q, K and V projections that follow each attn_norm on
#                               three concurrent streams. Only active while CUDA graphs are in use, which on Pascal
#                               needs a build with the CUDA-graphs-on-Pascal patch (build_llamacpp_docker.sh
#                               --cuda-graphs-pascal), and only for models with separate Q, K and V matrices (a fused
#                               QKV tensor has one consumer, not three). The value must be exactly 1.
#   GGML_CUDA_DISABLE_FUSION=1  Turns off every kernel fusion of the CUDA backend: fused gate/up mat-vecs with the GLU,
#                               shared expert + routed experts, top-k MoE routing, RMS norm + mul (+ add/rope/set_rows),
#                               multi-add, MoE weighted reduction, mat-mul + bias. Any non-zero integer enables it.
#
# Runs llama-bench for: no switch, each switch alone, both, and no switch again (to show drift over the run).
# Each run reloads the model, which for a 26 GB file takes a few minutes on the rig's x1 links.
#
#   BIN=/opt/<cuda-graphs-build>/bin MODEL=/mnt/models/flint-int4-q4_1.gguf TAG=flint-env bash scripts/env_ab.sh
#   ssh rig "BIN=... MODEL=... TAG=... bash -s" < scripts/env_ab.sh
#
# Env GPUS: CUDA_VISIBLE_DEVICES for every run (default 0,1,2,3,4)
#     BENCH_ARGS: llama-bench arguments (default "-ngl 99 -fa on -p 0 -n 128 -r 5": generation only)
# Results go to /tmp/bench/<TAG>/<variant>.jsonl and .log.
set -u

BIN=${BIN:?set BIN to the bin/ directory of a CUDA build}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:?set TAG}
GPUS=${GPUS:-0,1,2,3,4}
BENCH_ARGS=${BENCH_ARGS:-"-ngl 99 -fa on -p 0 -n 128 -r 5"}
OUT=/tmp/bench/$TAG
mkdir -p "$OUT"
unset GGML_CUDA_GRAPH_OPT GGML_CUDA_DISABLE_FUSION

run() {
    local name=$1; shift
    echo "== $name $(date +%T) $*"
    env "$@" CUDA_VISIBLE_DEVICES="$GPUS" "$BIN/llama-bench" -m "$MODEL" $BENCH_ARGS -o jsonl -oe md \
        > "$OUT/$name.jsonl" 2> "$OUT/$name.log"
    grep -E "^\| [a-zA-Z]" "$OUT/$name.log" | grep -v "^| model"
    grep -E "error|failed" "$OUT/$name.log" | head -3
}

run base
run graph_opt GGML_CUDA_GRAPH_OPT=1
run no_fusion GGML_CUDA_DISABLE_FUSION=1
run both GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_DISABLE_FUSION=1
run base_again
echo "== done $(date +%T); results in $OUT"
