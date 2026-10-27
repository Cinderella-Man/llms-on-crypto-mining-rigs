# Q112 E2/E3 (2026-10-24) on K2-r16 (K2 with F16 routers, the lead body; its body tensors are K2h-r16's): shared
# definitions, sourced by the job scripts in this folder; run them from the repository root (tools in quants/gptq_head).
# Env: MODELS (flint-q126-K2-r16.gguf, flint-BF16.gguf, imatrix_unsloth.gguf), WORK (small outputs; holds
# wiki.train.raw and wikitext-2-raw/), BIG (Hessians, donors, split files: tens of GB), BIN (bin/ of a CUDA build with
# act_hessian), LLAMA_SRC (its source tree, for gguf-py), GPUS.
K=$MODELS/flint-q126-K2-r16.gguf
BF16=$MODELS/flint-BF16.gguf; IMAT=$MODELS/imatrix_unsloth.gguf
D=$WORK; HD=$BIG/work/q112e2; HM=$BIG/models/q112e2
T=$PWD/quants/gptq_head; B=$BIN; IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
MOUNTS="-v $MODELS:$MODELS:ro -v $WORK:$WORK -v $BIG:$BIG -v $PWD:$PWD:ro -v $LLAMA_SRC:$LLAMA_SRC:ro -v $(dirname "$BIN"):$(dirname "$BIN"):ro"
# every always-read matrix (attention q/k/v/output, shared expert, layers 0-1's dense FFN): the tensors any
# always-read variant may change. The shared base leaves them all out; each variant's second file holds them.
UA='^blk\.[0-9]+\.(attn_(q|k|v|output)|ffn_(gate|up|down)(_shexp)?)\.weight$'
mkdir -p $D $HD $HM
gpu() { docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID $MOUNTS "$@"; }
# python:3.12-slim with numpy and gguf-py; OPENBLAS_NUM_THREADS fixed, since GPTQ's codes depend on BLAS summation order
py() { docker run --rm $MOUNTS -e PYTHONPATH=$LLAMA_SRC/gguf-py -e OPENBLAS_NUM_THREADS=${BLAS:-12} python:3.12-slim \
  bash -c "pip install -q numpy pyyaml 2>/dev/null; $1"; }
