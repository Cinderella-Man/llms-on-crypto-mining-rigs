#!/usr/bin/env bash
# C912's Ornith agent run (registry Q1649): Ornith's coding-agent check with its v5 preset (F16 cache, as it is) on the
# upstream b12674 candidate (pascal-p106-v5-cand-b12674 eee615cdb), against R232 (b)'s control runs of the same preset
# on pascal-p106-v5 af00bb13b (o5f16a, o5f16b; docs/data/2026-11-24_r232b_ornith_q8_agent/).
# This is R232 (b)'s run script (docs/data/2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh) with its ARM=f16 line
# exactly as it is: the same environment, arguments, context (49,152 tokens), sampling, 12 tasks and time caps. Only
# two things differ: the engine (BIN, set by the caller) and the harness's arm name (NAME, default o5cand), so the run's
# rows do not mix with R232 (b)'s. The script stops if the v5 preset file no longer holds the environment and arguments
# it was written for.
# STEPS=run (the default) runs the 12 tasks once; STEPS="report collect" then writes the harness's report on the client
# and copies the client's results folder to OUT.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   MODEL, BIN   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf and the engine's bin/llama-server, paths on the rig
#   RIG_LOGDIR   a folder on the rig for the server's and the host's logs (moved to the client at the end)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client (one CLIENT_OUT a run: the client's
#                container names come from it, so another agent check on the same client at the same time is fine)
#   OUT          a local folder
#   optional: NAME (o5cand; lower-case letters and digits), RIG_PORT, RELAY_PORT, PROXY_PORT (8282, 8286, 18783)
NAME=${NAME:-o5cand}
[[ $NAME =~ ^[a-z0-9]+$ ]] || { echo "bad NAME: $NAME (lower-case letters and digits)"; exit 9; }
V5_ENV="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1"
V5_ENV="$V5_ENV GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1"
V5_ARGS="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek"
V5_ARGS="$V5_ARGS --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
V5_ARGS="$V5_ARGS --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
. release/ornith-v5/settings.env || exit 1
[ "$PRESET_ENV" = "$V5_ENV" ] || { echo "the v5 preset's environment changed: $PRESET_ENV"; exit 9; }
[ "$PRESET_ARGS" = "$V5_ARGS" ] || { echo "the v5 preset's arguments changed: $PRESET_ARGS"; exit 9; }
# The harness puts its common arguments first (-ngl 99 -fa on -c CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0
# --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui) and the arm's
# after them; with the arm's below the server runs the preset's line (the later --temp wins), as in R17 and R232 (b).
ARM_ARGS="--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
ENV="$PRESET_ENV"
env BENCH="$PWD/bench/agent_check" RIG="${RIG:?}" CLIENT="${CLIENT:?}" CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=256 \
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}' \
MODEL="${MODEL:?}" RIG_LOGDIR="${RIG_LOGDIR:?}" \
RIG_PORT="${RIG_PORT:-8282}" RELAY_PORT="${RELAY_PORT:-8286}" PROXY_PORT="${PROXY_PORT:-18783}" \
CLIENT_DIR="${CLIENT_DIR:?}" CLIENT_OUT="${CLIENT_OUT:?}" \
REL_BIN="${BIN:?}" REL_ENV="$ENV" \
"ARGS_$NAME=$ARM_ARGS" "BUDGET_MIN_$NAME=50" ARMS="$NAME" STEPS="${STEPS:-run}" \
timeout 7200 bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "${OUT:?}"
