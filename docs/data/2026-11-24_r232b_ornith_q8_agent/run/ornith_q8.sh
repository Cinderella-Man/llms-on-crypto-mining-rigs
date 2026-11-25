#!/usr/bin/env bash
# R232 (b) (Q1706): Ornith's coding-agent check with an 8-bit (Q8_0) attention cache and pull request 9's Pascal
# attention kernels, against its v5 preset's 16-bit (F16) cache. Two arms, two runs each, alternated on one boot:
#   ARM=f16  the v5 preset as it is (release/ornith-v5/settings.env), started exactly as R17's text-reuse runs
#            started it (docs/data/2026-11-19_r17_ornith_textreuse/run/ornith_switch.sh with
#            EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1, the v5 preset since 2026-11-19): the same environment, arguments,
#            context (49,152 tokens), sampling, 12 tasks and time caps
#   ARM=q8   the same plus -ctk q8_0 -ctv q8_0 at the end of the arguments and GGML_CUDA_FATTN_TILE_PASCAL_MIN=1
#            GGML_CUDA_FATTN_DEC_PASCAL=4 at the end of the environment (R232 (a): with a Q8_0 cache these two route
#            Ornith's one-token attention to the decode kernel, docs/data/2026-11-24_r232a_ornith_q8_kernels/)
#   ARM=kdp  the q8 arm plus GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 at the end of the environment (C1102, Q1835: pull
#            request 9's prompt kernel scores the Q8_0 cache in int8 with dp4a; docs/data/2026-11-25_c1102_kdp_agent/)
#   RUN=a|b  which run of the arm; the harness's arm name is o5<ARM><RUN> (o5f16a, o5q8a, o5f16b, o5q8b, o5kdpa,
#            o5kdpb)
# Nothing else differs between the arms. The script stops if the v5 preset file no longer holds the environment and
# arguments it was written for. Since 2026-11-24 (Q1706) the preset file holds the q8 arm's line, the F16 line with the
# two switches and -ctk q8_0 -ctv q8_0 added; the script accepts the file in either form and builds every arm from the
# F16 line, so each arm's command is the same with either.
# STEPS=run (the default) runs the 12 tasks once; STEPS="report collect" then writes the harness's report on the client
# and copies the client's results folder to OUT.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   MODEL, BIN   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf and the v5 engine's bin/llama-server, paths on the rig
#                (the Q8_0 arm needs pull request 9's kernels: pascal-p106-v5 af00bb13b or later)
#   RIG_LOGDIR   a folder on the rig for the server's and the host's logs (moved to the client at the end)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client (one CLIENT_OUT a run: the client's
#                container names come from it, so another agent check on the same client at the same time is fine)
#   OUT          a local folder
#   optional: RIG_PORT, RELAY_PORT, PROXY_PORT (8282, 8286, 18783)
ARM=${ARM:?ARM=f16, q8 or kdp}; RUN=${RUN:?RUN=a or b}
[[ $RUN =~ ^[ab]$ ]] || { echo "bad RUN: $RUN (a or b)"; exit 9; }
V5_ENV="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1"
V5_ENV="$V5_ENV GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1"
V5_ARGS="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek"
V5_ARGS="$V5_ARGS --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
V5_ARGS="$V5_ARGS --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
. release/ornith-v5/settings.env || exit 1
if [ "$PRESET_ENV" = "$V5_ENV GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4" ] \
  && [ "$PRESET_ARGS" = "$V5_ARGS -ctk q8_0 -ctv q8_0" ]; then PRESET_ENV=$V5_ENV PRESET_ARGS=$V5_ARGS; fi
[ "$PRESET_ENV" = "$V5_ENV" ] || { echo "the v5 preset's environment changed: $PRESET_ENV"; exit 9; }
[ "$PRESET_ARGS" = "$V5_ARGS" ] || { echo "the v5 preset's arguments changed: $PRESET_ARGS"; exit 9; }
# The harness puts its common arguments first (-ngl 99 -fa on -c CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0
# --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui) and the arm's
# after them; with the arm's below the server runs the preset's line (the later --temp wins), as in R17.
ARM_ARGS="--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
case $ARM in
  f16) ENV="$PRESET_ENV";;
  q8) ENV="$PRESET_ENV GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
      ARM_ARGS="$ARM_ARGS -ctk q8_0 -ctv q8_0";;
  kdp) ENV="$PRESET_ENV GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1"
      ARM_ARGS="$ARM_ARGS -ctk q8_0 -ctv q8_0";;
  *) echo "bad ARM: $ARM (f16, q8 or kdp)"; exit 9;;
esac
NAME=o5$ARM$RUN
env BENCH="$PWD/bench/agent_check" RIG="${RIG:?}" CLIENT="${CLIENT:?}" CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=256 \
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}' \
MODEL="${MODEL:?}" RIG_LOGDIR="${RIG_LOGDIR:?}" \
RIG_PORT="${RIG_PORT:-8282}" RELAY_PORT="${RELAY_PORT:-8286}" PROXY_PORT="${PROXY_PORT:-18783}" \
CLIENT_DIR="${CLIENT_DIR:?}" CLIENT_OUT="${CLIENT_OUT:?}" \
REL_BIN="${BIN:?}" REL_ENV="$ENV" \
"ARGS_$NAME=$ARM_ARGS" "BUDGET_MIN_$NAME=50" ARMS="$NAME" STEPS="${STEPS:-run}" \
timeout 7200 bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "${OUT:?}"
