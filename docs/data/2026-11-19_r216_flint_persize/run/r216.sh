#!/usr/bin/env bash
# R216 (Q1627): agent checks (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 40-minute budget) on
# Flint Pascal Q8 GGUF (flint-near-q8-p065.gguf) at 262,144 tokens on nine P106-090 cards, with Flint's v5 preset
# (release/flint-v5/settings.env) on the v5 engine (pascal-p106-v5 602691153) and one variable added:
# LLAMA_SCHED_PER_SIZE=1. Every pass of up to 8 tokens (the one-token passes under the tool-call grammar, a prompt's
# 4-token output pass) then runs on a scheduler of its own for its size and outputs, and the preset's
# LLAMA_SCHED_PER_SIZE_RESERVE=2, which without per-size schedulers the server ignores, plans each of those once over
# the whole memory. The prompt scheduler then sees only prompt graphs and should keep its load-time plan (R181, Q1522:
# without the variable it re-planned 96 times with the cards busy in an agent run).
# TRACE=1 (the first run, R216a) also sets LLAMA_UBATCH_TRACE=1, so the server log names every re-plan and its cause, as
# in R181 (docs/data/2026-11-19_r181_flint_realloc/run/r181.sh); TRACE=0 runs are the timing runs.
# The arm is the release harness's custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh: ARGS_<arm>
# after the harness's COMMON line, REL_ENV the preset's six switches and then the added ones); its flags are the
# preset's PRESET_ARGS less what COMMON already passes (R181's arm). The script stops if the settings file's
# PRESET_ARGS or PRESET_ENV is not the line and the switches R216 ran.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1). Env: RIG, CLIENT, MODEL (flint-near-q8-p065.gguf on the rig), BIN (the v5
# engine's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); TRACE (1 or 0); ARM (f5pst traced, f5psa and
# f5psb untraced); PREFIX (the client's container names, PREFIX-ARM-*: a name of its own when another check shares the
# client); STEPS (run report collect).
set -uo pipefail
: "${OUT:?}" "${BIN:?}" "${TRACE:?}"
ARM=${ARM:-f5pst}
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64"
SWITCHES="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1"
. release/flint-v5/settings.env
[ "$PRESET_ARGS" = "$LINE" ] || { echo "release/flint-v5/settings.env: PRESET_ARGS is not the line R216 ran"; exit 1; }
[ "$PRESET_ENV" = "$SWITCHES" ] || { echo "release/flint-v5/settings.env: PRESET_ENV is not the switches R216 ran"; exit 1; }
ADDED="LLAMA_SCHED_PER_SIZE=1"
[ "$TRACE" = 1 ] && ADDED="$ADDED LLAMA_UBATCH_TRACE=1"
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV $ADDED" ARMS=$ARM STEPS=${STEPS:-run report collect}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
