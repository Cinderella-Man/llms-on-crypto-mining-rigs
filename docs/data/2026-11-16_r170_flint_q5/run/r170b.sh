#!/usr/bin/env bash
# R170b (Q1497): the agent check (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 40-minute budget) on
# Flint Pascal Q5 GGUF at 262,144 tokens on nine P106-090 cards, with v3.0 (the release build pascal-p106-v3-b12076-ladder5
# 1551f5fe1, tag miner-llama-v3.0) and its preset release/flint-v3/settings.env, layers by count: the preset's line
# without -ts, as v3's Q5 release run (docs/data/2026-11-05_release_regression, arm v3base on the 5-bit file). The same
# night and rig as R170a (run/r170a.sh, v4's preset on v4-rc2), so the two compare v4 with v3.0 on this file.
# The arm q5f3 is the release harness's custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh:
# ARGS_q5f3 after the harness's COMMON line, REL_ENV the preset's PRESET_ENV, which is empty: no switch). Its flags are
# the preset's PRESET_ARGS less what COMMON already passes (-ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4
# --cache-ram 0 and Flint's sampling) and less -ts; the script stops if the settings file is not that line with
# -ts 6,6,5,5,6,5,5,6,6 and an empty PRESET_ENV.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: RIG, CLIENT, MODEL (flint-pascal-Q5.gguf on the
# rig), BIN (v3.0's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); STEPS (default run; "report collect" for the
# report).
set -uo pipefail
: "${OUT:?}" "${BIN:?}"
ARM=q5f3
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4"
. release/flint-v3/settings.env
[ "$PRESET_ARGS" = "$LINE" ] && [ -z "$PRESET_ENV" ] || { echo "release/flint-v3/settings.env is not the preset R170b ran"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV" ARMS=$ARM STEPS=${STEPS:-run}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 --pipe-depth 4" "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
