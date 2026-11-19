#!/usr/bin/env bash
# R181 (Q1522): one agent check (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 40-minute budget) on
# Flint Pascal Q8 GGUF (flint-near-q8-p065.gguf) at 262,144 tokens on nine P106-090 cards, with Flint's v5 preset
# (release/flint-v5/settings.env: v4.0's preset plus LLAMA_SERVER_TEXT_REUSE=1) on the v5 engine (pascal-p106-v5
# 602691153) and one variable added: LLAMA_UBATCH_TRACE=1. It is a trace run, not a timing run: the variable makes the
# server log, at its default verbosity,
#   - sched-replan: every re-plan of a scheduler's compute buffers (whether a node changed device, and which; whether
#     the graph's size changed; the nodes, leafs and splits; the wait for every card, drain, and the plan, in us),
#   - galloc-trigger: the tensor that did not fit the plan (node, source or leaf; its name, op, shape, the bytes it
#     needs and the bytes its place was planned for), just before its re-plan,
#   - ubatch-trace and decode-trace: every micro-batch the context evaluates (tokens, outputs, graph reused or rebuilt
#     and why) and every decode call (tokens, outputs).
# The arm is the release harness's custom arm (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh: ARGS_<arm>
# after the harness's COMMON line, REL_ENV the preset's six switches and then LLAMA_UBATCH_TRACE=1); its flags are the
# preset's PRESET_ARGS less what COMMON already passes (R17's arm, run/r17.sh in 2026-11-17_r17_flint_textreuse).
# The script stops if the settings file's PRESET_ARGS or PRESET_ENV is not the line and the switches R181 ran.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1). Env: RIG, CLIENT, MODEL (flint-near-q8-p065.gguf on the rig), BIN (the v5
# engine's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); ARM (f5ut); PREFIX (the client's container
# names, PREFIX-ARM-*: a name of its own when another check shares the client); STEPS (run report collect).
set -uo pipefail
: "${OUT:?}" "${BIN:?}"
ARM=${ARM:-f5ut}
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64"
SWITCHES="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1"
. release/flint-v5/settings.env
[ "$PRESET_ARGS" = "$LINE" ] || { echo "release/flint-v5/settings.env: PRESET_ARGS is not the line R181 ran"; exit 1; }
[ "$PRESET_ENV" = "$SWITCHES" ] || { echo "release/flint-v5/settings.env: PRESET_ENV is not the switches R181 ran"; exit 1; }
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV LLAMA_UBATCH_TRACE=1" ARMS=$ARM STEPS=${STEPS:-run report collect}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
