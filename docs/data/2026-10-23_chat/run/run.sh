#!/bin/bash
# Flint chat benchmark, 2026-10-23, part 1: the five-message chat (scripts/server_bench.sh PROMPTS=chat) on all nine
# cards at 262,144 tokens of context with the fork build (pascal-p106 at eeaa59660). Run from the repository root.
# The UD-Q4_K_XL run is chat-ud (8,192 tokens per reply); with MAX_TOKENS=3072 and TAG=chat-ud-max3072 it is
# chat-ud-max3072. The S4-L0 and Q8_0 runs of this script were stopped and redone by run2.sh with 65,536 tokens.
# Env: RIG (ssh alias), FORK (the fork's bin directory on the rig), UD, L0, Q8 (model paths on the rig), MAX_TOKENS.
RIG=${RIG:-rig}
FORK=${FORK:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
L0=${L0:-/mnt/models2/flint/flint-S4L0.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
MAX_TOKENS=${MAX_TOKENS:-8192}
A="-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4"
E="BIN=$FORK PROMPTS=chat MAX_TOKENS=$MAX_TOKENS GPUS=0,1,2,3,4,5,6,7,8"
echo "== UD-Q4_K_XL $(date +%T)"
ssh "$RIG" "$E MODEL=$UD TAG=${TAG:-chat-ud} SERVER_ARGS='$A' bash -s" < scripts/server_bench.sh
echo "== S4-L0 $(date +%T)"
ssh "$RIG" "$E LLAMA_NO_MOE_DOWN_RESCALE=1 MODEL=$L0 TAG=chat-s4l0 SERVER_ARGS='$A' bash -s" < scripts/server_bench.sh
echo "== Q8_0 $(date +%T)"
ssh "$RIG" "$E LLAMA_NO_MOE_DOWN_RESCALE=1 MODEL=$Q8 TAG=chat-q80 SERVER_ARGS='$A -ts 6,6,5,5,6,5,5,6,6' bash -s" < scripts/server_bench.sh
echo DONE $(date +%T)
