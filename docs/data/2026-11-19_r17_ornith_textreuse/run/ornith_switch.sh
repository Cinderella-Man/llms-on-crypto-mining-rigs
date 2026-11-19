#!/usr/bin/env bash
# R17's Ornith arm (Q260): Ornith's agent check with its v5 preset, with or without one more server variable. This is
# R204's launcher (docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh: v4.0's preset
# release/ornith-v4/settings.env plus GGML_CUDA_MMID_GLU_UNFUSE=2, which is the v5 preset release/ornith-v5/settings.env)
# with EXTRA_ENV appended to the server's environment. With EXTRA_ENV empty the started command is that launcher's byte
# for byte; the text-reuse runs set EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1. The arguments, the context, the sampling, the
# 12 tasks and the time caps are that launcher's, so the runs differ only in ARM and EXTRA_ENV.
# STEPS=run (the default) runs the agent check's 12 tasks once; STEPS="report collect" then writes its report on the
# client and copies the client's results folder to OUT.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   MODEL, BIN   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf and the v5 engine's bin/llama-server, paths on the rig
#   RIG_LOGDIR   a folder on the rig for the server's and the host's logs (moved to the client at the end)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client
#   OUT          a local folder
#   ARM          the arm's name, lower-case letters and digits
#   EXTRA_ENV    more server variables as "VAR=value ..." (default none)
#   optional: RIG_PORT, RELAY_PORT, PROXY_PORT (8282, 8286, 18783); PREFIX (the client's container prefix, default
#   acrel; with CLIENT=local two checks on one client at once need their own, docs/data/2026-11-03_flash_next_mtp/run/)
. release/ornith-v4/settings.env || exit 1
[ "$PRESET_ENV" = "GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1" ] \
  || { echo "unexpected environment: $PRESET_ENV"; exit 9; }
V5_ENV=$(. release/ornith-v5/settings.env && echo "$PRESET_ENV")
[ "$V5_ENV" = "$PRESET_ENV GGML_CUDA_MMID_GLU_UNFUSE=2" ] || { echo "the v5 preset is not v4.0's plus the switch: $V5_ENV"; exit 9; }
ARM=${ARM:?}
[[ $ARM =~ ^[a-z0-9]+$ ]] || { echo "bad arm name: $ARM"; exit 9; }
EXTRA_ENV=${EXTRA_ENV:-}
[[ -z $EXTRA_ENV || $EXTRA_ENV =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:=-]*( [A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:=-]*)*$ ]] \
  || { echo "bad EXTRA_ENV: $EXTRA_ENV"; exit 9; }
env BENCH="$PWD/bench/agent_check" RIG="${RIG:?}" CLIENT="${CLIENT:?}" CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=256 \
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}' \
MODEL="${MODEL:?}" RIG_LOGDIR="${RIG_LOGDIR:?}" \
RIG_PORT="${RIG_PORT:-8282}" RELAY_PORT="${RELAY_PORT:-8286}" PROXY_PORT="${PROXY_PORT:-18783}" \
CLIENT_DIR="${CLIENT_DIR:?}" CLIENT_OUT="${CLIENT_OUT:?}" \
REL_BIN="${BIN:?}" REL_ENV="$PRESET_ENV GGML_CUDA_MMID_GLU_UNFUSE=2${EXTRA_ENV:+ $EXTRA_ENV}" \
"ARGS_$ARM=--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall" \
"BUDGET_MIN_$ARM=50" ARMS="$ARM" STEPS="${STEPS:-run}" \
timeout 7200 bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "${OUT:?}"
