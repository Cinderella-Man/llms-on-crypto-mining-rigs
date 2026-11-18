#!/usr/bin/env bash
# R203 (Q1592), Ornith's agent check at stock and with clock offsets, one arm per call: R166a's o4sna, the released
# v4.0 preset (release/ornith-v4/settings.env) on v4.0's llama-server, the same started command byte for byte. Arms:
# o4oc0, the cards at stock; o4oc1, the same run after the use profile's offsets were applied on the same boot (by the
# offset kit, outside this script). This script sets no offset, clock or power limit.
# STEPS=run (the default) runs the agent check's 12 tasks once; STEPS="report collect" then writes the report on the
# client and copies the client's results folder to OUT.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   ARM          the arm's name (o4oc0 or o4oc1)
#   MODEL, BIN   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf and v4.0's bin/llama-server, paths on the rig
#   RIG_LOGDIR   a folder on the rig for the server's, the cards' and the host's logs (moved to the client at the end)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client
#   OUT          a local folder
#   RIG_PORT, RELAY_PORT, PROXY_PORT   (8282, 8286, 18783) other ports when another check shares the local client
. release/ornith-v4/settings.env || exit 1
[ "$PRESET_ENV" = "GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1" ] \
  || { echo "unexpected environment: $PRESET_ENV"; exit 9; }
ARM=${ARM:?}
export "ARGS_$ARM=--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
export "BUDGET_MIN_$ARM=50"
BENCH=$PWD/bench/agent_check RIG=${RIG:?} CLIENT=${CLIENT:?} CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=256 \
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}' \
MODEL=${MODEL:?} RIG_LOGDIR=${RIG_LOGDIR:?} \
RIG_PORT=${RIG_PORT:-8282} RELAY_PORT=${RELAY_PORT:-8286} PROXY_PORT=${PROXY_PORT:-18783} \
CLIENT_DIR=${CLIENT_DIR:?} CLIENT_OUT=${CLIENT_OUT:?} \
REL_BIN=${BIN:?} REL_ENV="$PRESET_ENV" \
ARMS=$ARM STEPS="${STEPS:-run}" \
timeout 7200 bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "${OUT:?}"
