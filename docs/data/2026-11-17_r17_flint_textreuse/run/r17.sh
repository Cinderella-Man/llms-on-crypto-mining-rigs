#!/usr/bin/env bash
# R17 (Q260): the agent check (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 40-minute budget) on
# Flint Pascal Q8 GGUF (flint-near-q8-p065.gguf) at 262,144 tokens on nine P106-090 cards, with Flint's released v4
# preset (release/flint-v4/settings.env) on Miner Llama v4.0 (tag miner-llama-v4.0, pascal-p106-v4 eb82916d3) and one
# variable added: LLAMA_SERVER_TEXT_REUSE=1. The arm (ARM, default f4tr; run b names its own) is the release harness's
# custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh: ARGS_<arm> after the harness's COMMON line,
# REL_ENV the preset's five switches and then LLAMA_SERVER_TEXT_REUSE=1). Its flags are the preset's PRESET_ARGS less
# what COMMON already passes (-ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 and Flint's
# sampling): the release check's tuned arm f4t (R159e, R159f), so the started command is f4t's plus the one variable.
# The script stops if the settings file's PRESET_ARGS or PRESET_ENV is not the line and the switches R159's f4t ran.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1 prepared with STEPS=prep). Env: RIG, CLIENT, MODEL (flint-near-q8-p065.gguf on the
# rig), BIN (v4.0's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); ARM (f4tr); STEPS (default run; "report collect"
# for the report).
set -uo pipefail
: "${OUT:?}" "${BIN:?}"
ARM=${ARM:-f4tr}
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64"
SWITCHES="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2"
. release/flint-v4/settings.env
[ "$PRESET_ARGS" = "$LINE" ] || { echo "release/flint-v4/settings.env: PRESET_ARGS is not the line R17 ran"; exit 1; }
[ "$PRESET_ENV" = "$SWITCHES" ] || { echo "release/flint-v4/settings.env: PRESET_ENV is not the switches R17 ran"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV LLAMA_SERVER_TEXT_REUSE=1" ARMS=$ARM STEPS=${STEPS:-run}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
