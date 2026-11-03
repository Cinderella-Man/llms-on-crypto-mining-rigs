#!/usr/bin/env bash
# The two bytes-per-token switches on a CPU, no GPU needed (run inside a container of the build image, from a CPU build
# of the branch with the tests: test-llama-archs, test-bytes-switches, llama-quantize, llama-server):
#   1. random qwen4exp models from test-llama-archs (4,096 tokens, 32 routed experts, 8 used), with an F32 head and a
#      Q5_K one (llama-quantize, everything else F32), each with the head as it is and with its rows from k_low on
#      zeroed (--zero-tail), the routers scaled by 30 (a random router weighs its experts alike); test-bytes-switches
#      compares every arm with the switch off;
#   2. the real model (MODEL, e.g. Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next) in llama-server, run/server_arms.sh:
#      greedy and sampled requests with each switch on and off, same seed.
# Env: BIN (bin/ of the build), OUT, MODEL (empty: step 1 only), THREADS (16), PORT (18191), NPRED (64),
#      HEAD (LLAMA_HEAD_BOUND, default 32768,8192), FREQ (LLAMA_HEAD_BOUND_FILE, optional), TAILS (the LLAMA_MOE_TAIL
#      arms on the real model, default "p=1 p=0.9,renorm=1 p=0.8,renorm=1")
# Writes $OUT/random/*.log and server_arms.sh's files in $OUT/real.
set -uo pipefail

: "${BIN:?}" "${OUT:?}"
THREADS=${THREADS:-16}
PORT=${PORT:-18191}
NPRED=${NPRED:-64}
HEAD=${HEAD:-32768,8192}
TAILS=${TAILS:-"p=1 p=0.9,renorm=1 p=0.8,renorm=1"}
mkdir -p "$OUT/random" "$OUT/real"

# 1. random models
M=$OUT/random
if [ ! -f "$M/qwen4exp-moe.gguf" ]; then
    LLAMA_TEST_ARCHS_N_VOCAB=4096 LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$BIN/test-llama-archs" -a qwen4exp -o "$M" > "$M/archs.log" 2>&1
fi
if [ ! -f "$M/qwen4exp-moe-q5k.gguf" ]; then
    "$BIN/llama-quantize" --allow-requantize --output-tensor-type q5_k \
        "$M/qwen4exp-moe.gguf" "$M/qwen4exp-moe-q5k.gguf" F32 "$THREADS" > "$M/quantize.log" 2>&1
fi
for m in qwen4exp-moe qwen4exp-moe-q5k; do
    for z in "" --zero-tail; do
        log=$M/$m${z:+-zero}.log
        "$BIN/test-bytes-switches" -m "$M/$m.gguf" -t "$THREADS" -n 24 -g 24 -k 1024 --router-scale 30 $z > "$log" 2>&1
        echo "$m ${z:-(head as it is)}: exit $?; $(tail -n 1 "$log")"
    done
done

[ -n "${MODEL:-}" ] || exit 0

# 2. the real model, in llama-server
ARMS="off head$(for t in $TAILS; do printf " tail:%s" "$t"; done)" SERVER_ARGS="-c 4096 -np 1 -t $THREADS -tb $THREADS" \
    BIN=$BIN MODEL=$MODEL OUT=$OUT/real PORT=$PORT NPRED=$NPRED HEAD=$HEAD bash "$(dirname "$0")/server_arms.sh"
