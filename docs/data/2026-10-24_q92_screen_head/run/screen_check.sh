#!/usr/bin/env bash
# Q92 (2026-10-24): llama-screen-check on G3 + the GPTQ Q4_0 screen: greedy (bitwise candidate logits, tokens) and the
# makers' sampling with a fixed seed, screened against the full head, on quants/gptq_head/prompts.txt; Pascal kernels
# forced (the rig's path), then the generic kernels; then plain G3 with the screen from the side file
# (LLAMA_SCREEN_HEAD_FILE), Pascal kernels forced. Env: BIN (bin/ of a build of the screened-head branch), MODELS, GPUS.
set -e
P=$PWD/quants/gptq_head/prompts.txt
run() { docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e LLAMA_NO_MOE_DOWN_RESCALE=1 \
  -v $MODELS:$MODELS:ro -v $PWD:$PWD:ro -v $(dirname "$BIN"):$(dirname "$BIN"):ro "$@" -m "$MODEL" -f $P \
  -n 128 -ngl 99 -fa on -c 4096 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234 2>&1 \
  | grep -E "prompt [0-9]+:|steps|compared|SCREEN CHECK|screen|error|failed|WARN" || true; }
B=$BIN/llama-screen-check
for pascal in 2 0; do
  echo "== G3 + screen, GGML_CUDA_MMVQ_PASCAL=$pascal"
  MODEL=$MODELS/flint-q73-G3-screen.gguf run -e GGML_CUDA_MMVQ_PASCAL=$pascal nvidia/cuda:12.9.2-devel-ubuntu24.04 $B
done
echo "== G3, screen from the side file, GGML_CUDA_MMVQ_PASCAL=2"
MODEL=$MODELS/flint-q73-G3.gguf run -e GGML_CUDA_MMVQ_PASCAL=2 \
  -e LLAMA_SCREEN_HEAD_FILE=$MODELS/flint-output-screen-gptq-q4_0.gguf nvidia/cuda:12.9.2-devel-ubuntu24.04 $B
