#!/usr/bin/env bash
# R166 (Q1491): Ornith's candidate stack without the micro-batch taper on the release agent check, one run a call.
# The agent check (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 50-minute budget an arm, the makers'
# sampling at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1) at 49,152 tokens on nine P106-090 cards.
# Arms o4sna and o4snb (two runs of the same arm) are R160's o4s* (docs/data/2026-11-14_r160_ornith_release_agent/run/
# r160.sh) without --ubatch-taper short=600:64:
#   BIN_V4 (v4-rc2) with release/ornith-v4/settings.env (its PRESET_ENV: the keep-alive and three switches), plus
#   GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1, --ctx-checkpoints-vram 4 --ctx-checkpoints-incall, and -b 4096 in
#   place of -b 2048.
# The arm's flags are the preset's PRESET_ARGS less what the harness's COMMON line already passes (-ngl 99 -fa on
# -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek and the sampling, whose
# --temp 1.0 the arm's --temp 0.6 overrides). The script stops if the settings file's PRESET_ARGS is not v3.0's Ornith
# line. Every request also carries the sampling (SAMPLING below).
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: ARM (o4sna or o4snb), RIG, CLIENT, MODEL (the file
# on the rig), BIN_V4 (llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT
# (and CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); STEPS (default run; "report collect" with
# ARMS set for the report).
set -uo pipefail
: "${ARM:?ARM=o4sna|o4snb}" "${OUT:?}"
case $ARM in
  o4sn*) SETTINGS=release/ornith-v4/settings.env; BIN=${BIN_V4:?};;
  *) echo "unknown arm $ARM"; exit 1;;
esac
LINE="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
. "$SETTINGS"
[ "$PRESET_ARGS" = "$LINE" ] || { echo "$SETTINGS: PRESET_ARGS is not v3.0's Ornith line"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1" ARMS=${ARMS:-$ARM} STEPS=${STEPS:-run} \
  SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'
export "ARGS_$ARM=--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall" \
  "BUDGET_MIN_$ARM=50"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
