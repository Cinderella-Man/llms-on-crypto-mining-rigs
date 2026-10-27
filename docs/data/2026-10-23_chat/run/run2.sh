#!/bin/bash
# Flint chat benchmark, 2026-10-23, part 2 (after run.sh's UD-Q4_K_XL run): the remaining five conversations with up
# to 65,536 tokens per reply: the fork (S4-L0, Q8_0), then stock llama.cpp d77256999 (UD-Q4_K_XL, S4-L0, Q8_0).
# Run from the repository root. Env: RIG (ssh alias), FORK and STOCK (bin directories on the rig), UD, L0, Q8 (model
# paths on the rig).
RIG=${RIG:-rig}
F=${FORK:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
S=${STOCK:-/opt/llama-builds/llama.cpp-20261022-d77256999/bin}
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
L0=${L0:-/mnt/models2/flint/flint-S4L0.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
A="-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4"
E="PROMPTS=chat MAX_TOKENS=65536 GPUS=0,1,2,3,4,5,6,7,8"
run() {  # run <label> <env and args before bash -s>
    echo "== $1 $(date +%T)"
    ssh "$RIG" "$E $2 bash -s" < scripts/server_bench.sh
}
run "fork S4-L0"        "BIN=$F LLAMA_NO_MOE_DOWN_RESCALE=1 MODEL=$L0 TAG=chat-s4l0 SERVER_ARGS='$A'"
run "fork Q8_0"         "BIN=$F LLAMA_NO_MOE_DOWN_RESCALE=1 MODEL=$Q8 TAG=chat-q80 SERVER_ARGS='$A -ts 6,6,5,5,6,5,5,6,6'"
run "stock UD-Q4_K_XL"  "BIN=$S MODEL=$UD TAG=chat-stock-ud SERVER_ARGS='$A'"
run "stock S4-L0"       "BIN=$S MODEL=$L0 TAG=chat-stock-s4l0 SERVER_ARGS='$A'"
run "stock Q8_0"        "BIN=$S MODEL=$Q8 TAG=chat-stock-q80 SERVER_ARGS='$A -ts 6,6,5,5,6,5,5,6,6'"
echo DONE $(date +%T)
