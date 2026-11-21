#!/usr/bin/env bash
# C901 (Q1629): R181's traced P106 run (docs/data/2026-11-19_r181_flint_realloc/run/r181.sh, the same harness, arm
# settings and trace) on the C901 build, with LLAMA_PIPE_SLOT_PLAN=1: a pipe slot that re-plans its compute buffers in a
# step is planned again over the whole memory for that step's samplers, so that later steps fit it. One agent check
# (bench/agent_check: 12 bug-fix tasks, a 12-minute cap a task, a 40-minute budget) on Flint Pascal Q8 GGUF
# (flint-near-q8-p065.gguf) at 262,144 tokens on nine P106-090 cards, with Flint's v5 preset
# (release/flint-v5/settings.env) plus LLAMA_UBATCH_TRACE=1 and EXTRA (default LLAMA_PIPE_SLOT_PLAN=1; EXTRA=
# LLAMA_PIPE_SLOT_PLAN=0 is the control on the same build, R181's behaviour with the new count line).
# The C901 build logs every pipe slot's re-plans per request after the graph count (`pipe reallocs = S / D ..., F plans
# over the whole memory`), and the agent check's report adds them to its graph reallocations line.
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1). Env: RIG, CLIENT, MODEL (flint-near-q8-p065.gguf on the rig), BIN (the C901
# build's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); ARM (f5spt: the switch on, traced; f5s0t: the
# control, with EXTRA=LLAMA_PIPE_SLOT_PLAN=0); EXTRA; PREFIX (the client's container names); STEPS (run report collect).
set -uo pipefail
: "${OUT:?}" "${BIN:?}"
ARM=${ARM:-f5spt}
EXTRA=${EXTRA-LLAMA_PIPE_SLOT_PLAN=1}
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64"
SWITCHES="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1"
. release/flint-v5/settings.env
[ "$PRESET_ARGS" = "$LINE" ] || { echo "release/flint-v5/settings.env: PRESET_ARGS is not the line R181 ran"; exit 1; }
[ "$PRESET_ENV" = "$SWITCHES" ] || { echo "release/flint-v5/settings.env: PRESET_ENV is not the switches R181 ran"; exit 1; }
ADDED="LLAMA_UBATCH_TRACE=1"
[ -n "$EXTRA" ] && ADDED="$ADDED $EXTRA"
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV $ADDED" ARMS=$ARM STEPS=${STEPS:-run report collect}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
