#!/usr/bin/env bash
# R169 (Q1496): Ornith's v4 preset without the eager keep-alive stop on the release agent check, one run a call.
# The agent check (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 50-minute budget an arm, the makers'
# sampling at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1) at 49,152 tokens on nine P106-090 cards.
# Arms o4nea and o4neb (two runs of the same arm) are R166's o4sn* (docs/data/2026-11-16_r166_ornith_stack_no_taper/
# run/r166.sh), which is the released preset release/ornith-v4/settings.env, less GGML_CUDA_KEEPALIVE_EAGER=1:
#   BIN_V4 (v4-rc2) with the settings file's PRESET_ENV less that one variable (the keep-alive GGML_CUDA_KEEPALIVE=decode
#   stays, with LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1), and the
#   preset's flags.
# The arm's flags are the preset's PRESET_ARGS less what the harness's COMMON line already passes (-ngl 99 -fa on
# -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek and the sampling, whose
# --temp 1.0 the arm's --temp 0.6 overrides). The script stops if the settings file's PRESET_ARGS or PRESET_ENV is not
# the released v4 preset's. Every request also carries the sampling (SAMPLING below).
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: ARM (o4nea or o4neb), RIG, CLIENT, MODEL (the file
# on the rig), BIN_V4 (llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT
# (and CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); STEPS (default run; "report collect" with
# ARMS set for the report).
set -uo pipefail
: "${ARM:?ARM=o4nea|o4neb}" "${OUT:?}"
case $ARM in
  o4ne*) SETTINGS=release/ornith-v4/settings.env; BIN=${BIN_V4:?};;
  *) echo "unknown arm $ARM"; exit 1;;
esac
LINE="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
ENV_V4="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1"
. "$SETTINGS"
[ "$PRESET_ARGS" = "$LINE" ] || { echo "$SETTINGS: PRESET_ARGS is not the released v4 Ornith line"; exit 1; }
[ "$PRESET_ENV" = "$ENV_V4" ] || { echo "$SETTINGS: PRESET_ENV is not the released v4 Ornith environment"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="${PRESET_ENV/ GGML_CUDA_KEEPALIVE_EAGER=1/}" ARMS=${ARMS:-$ARM} STEPS=${STEPS:-run} \
  SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'
export "ARGS_$ARM=--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall" \
  "BUDGET_MIN_$ARM=50"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
