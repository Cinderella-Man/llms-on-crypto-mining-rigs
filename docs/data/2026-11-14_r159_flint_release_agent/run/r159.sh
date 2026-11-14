#!/usr/bin/env bash
# R159 (Q1478): Flint's release agent check, one arm a run. The agent check (bench/agent_check: 12 bug-fix tasks, a
# 12-minute cap a task, a 40-minute budget an arm) on Flint Pascal Q8 GGUF (p065) at 262,144 tokens on nine P106-090
# cards, the v4 preset on the release candidate v4-rc2 (pascal-p106-v4 eb82916d3) against the v3.0 release
# (pascal-p106-v3-b12076-ladder5 1551f5fe1), alternated over four runs: f4a, f3a, f4b, f3b. Each arm is the release
# harness's custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh: the arm's ARGS after the harness's
# COMMON line, REL_ENV the preset's switches) on v3.0's Flint line:
#   f4*   BIN_V4 with release/flint-v4/settings.env (its PRESET_ENV, three switches)
#   f3*   BIN_V3 with release/flint-v3/settings.env (no switch)
# The arm's flags are the preset's PRESET_ARGS less what COMMON already passes (-ngl 99 -fa on -c 262144 -np 1 -bs
# --ctx-checkpoints 4 --cache-ram 0 and Flint's sampling): --alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6
# --pipe-depth 4. The script stops if the settings file's PRESET_ARGS is not that line.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: ARM (f4a, f3a, f4b or f3b), RIG, CLIENT, MODEL (the
# file on the rig), BIN_V4 and BIN_V3 (llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT,
# RELAY_PORT, PROXY_PORT (and CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); STEPS (default
# run; "report collect" with ARMS="f3a f4a f3b f4b" for the report over the runs).
set -uo pipefail
: "${ARM:?ARM=f4a|f3a|f4b|f3b}" "${OUT:?}"
case $ARM in
  f4*) SETTINGS=release/flint-v4/settings.env; BIN=${BIN_V4:?};;
  f3*) SETTINGS=release/flint-v3/settings.env; BIN=${BIN_V3:?};;
  *) echo "unknown arm $ARM"; exit 1;;
esac
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4"
. "$SETTINGS"
[ "$PRESET_ARGS" = "$LINE" ] || { echo "$SETTINGS: PRESET_ARGS is not v3.0's Flint line"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV" ARMS=${ARMS:-$ARM} STEPS=${STEPS:-run}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4" "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
