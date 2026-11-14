#!/usr/bin/env bash
# R160 (Q1479): Ornith's release agent check, one arm a run. The agent check (bench/agent_check: 12 bug-fix tasks, a
# 12-minute cap a task, a 50-minute budget an arm, the makers' sampling at temperature 0.6) on Ornith-1.5-122B Pascal
# N2P GGUF (valid1) at 49,152 tokens on nine P106-090 cards, alternated over six runs: o4a, o4sa, o3a, o4b, o4sb, o3b.
# Each arm is the release harness's custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh: the
# arm's ARGS after the harness's COMMON line, REL_ENV the switches) on v3.0's Ornith line:
#   o4*   BIN_V4 (v4-rc2) with release/ornith-v4/settings.env (its PRESET_ENV: the keep-alive and three switches)
#   o4s*  BIN_V4 with o4's switches plus the candidate stack of R162's arm v4s
#         (docs/data/2026-11-14_r162_ornith_release_deep): GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1,
#         --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64, -b 4096 in place of -b 2048
#   o3*   BIN_V3 (the v3.0 release) with release/ornith-v3/settings.env (the keep-alive)
# The arm's flags are the preset's PRESET_ARGS less what COMMON already passes (-ngl 99 -fa on -c 49152 -np 1 -bs
# --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek and the sampling, whose --temp 1.0 the arm's
# --temp 0.6 overrides): --temp 0.6 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off. The script stops if the settings
# file's PRESET_ARGS is not that line. Every request also carries the sampling (SAMPLING below).
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: ARM (o4a, o4sa, o3a, o4b, o4sb or o3b), RIG, CLIENT,
# MODEL (the file on the rig), BIN_V4 and BIN_V3 (llama-server on the rig), RIG_LOGDIR (a folder on the rig),
# RIG_PORT, RELAY_PORT, PROXY_PORT (and CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); STEPS
# (default run; "report collect" with ARMS="o3a o4a o4sa o3b o4b o4sb" for the report over the runs).
set -uo pipefail
: "${ARM:?ARM=o4a|o4sa|o3a|o4b|o4sb|o3b}" "${OUT:?}"
EXTRA_ENV= B=2048 EXTRA_ARGS=
case $ARM in
  o4s*) SETTINGS=release/ornith-v4/settings.env; BIN=${BIN_V4:?}
        EXTRA_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1" B=4096
        EXTRA_ARGS=" --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64";;
  o4*) SETTINGS=release/ornith-v4/settings.env; BIN=${BIN_V4:?};;
  o3*) SETTINGS=release/ornith-v3/settings.env; BIN=${BIN_V3:?};;
  *) echo "unknown arm $ARM"; exit 1;;
esac
LINE="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
. "$SETTINGS"
[ "$PRESET_ARGS" = "$LINE" ] || { echo "$SETTINGS: PRESET_ARGS is not v3.0's Ornith line"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=49152 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV${EXTRA_ENV:+ $EXTRA_ENV}" ARMS=${ARMS:-$ARM} STEPS=${STEPS:-run} \
  SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'
export "ARGS_$ARM=--temp 0.6 -b $B -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off$EXTRA_ARGS" "BUDGET_MIN_$ARM=50"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
