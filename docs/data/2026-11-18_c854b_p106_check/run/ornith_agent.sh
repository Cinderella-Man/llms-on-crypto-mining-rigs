#!/usr/bin/env bash
# C854 step 4 (Q1525), Ornith: arm o4c2 = R166a's o4sna, the released v4.0 preset (release/ornith-v4/settings.env),
# with the candidate's llama-server. STEPS=run (the default) runs the agent check's 12 tasks once; STEPS="report collect"
# then writes its report on the client and copies the client's results folder to OUT.
# R206 (Q1597), the same-rig reference: the same call with ARM=o4v5a and BIN = pascal-p106-v5's own build
# (1f1110c67); ARM only names the arm (its results, its server log), the settings are the same.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   MODEL, BIN   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf and the engine's bin/llama-server, paths on the rig
#   RIG_LOGDIR   a folder on the rig for the server's and the host's logs (moved to the client at the end)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client
#   OUT          a local folder
#   ARM          the arm's name, lower-case letters and digits (default o4c2)
. release/ornith-v4/settings.env || exit 1
[ "$PRESET_ENV" = "GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1" ] \
  || { echo "unexpected environment: $PRESET_ENV"; exit 9; }
ARM=${ARM:-o4c2}
[[ $ARM =~ ^[a-z0-9]+$ ]] || { echo "bad arm name: $ARM"; exit 9; }
env BENCH="$PWD/bench/agent_check" RIG="${RIG:?}" CLIENT="${CLIENT:?}" CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=256 \
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}' \
MODEL="${MODEL:?}" RIG_LOGDIR="${RIG_LOGDIR:?}" \
RIG_PORT="${RIG_PORT:-8282}" RELAY_PORT="${RELAY_PORT:-8286}" PROXY_PORT="${PROXY_PORT:-18783}" \
CLIENT_DIR="${CLIENT_DIR:?}" CLIENT_OUT="${CLIENT_OUT:?}" \
REL_BIN="${BIN:?}" REL_ENV="$PRESET_ENV" \
"ARGS_$ARM=--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall" \
"BUDGET_MIN_$ARM=50" ARMS="$ARM" STEPS="${STEPS:-run}" \
timeout 7200 bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "${OUT:?}"
