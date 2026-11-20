#!/usr/bin/env bash
# C903 (Q1632): the traced P106 run of R216a (docs/data/2026-11-19_r216_flint_persize/run/r216.sh, the same harness, arm
# and settings) on the C903 build, whose per-size schedulers are made with pipeline parallelism when the context has it
# (LLAMA_SCHED_PER_SIZE_PP=1, the default with LLAMA_SCHED_PER_SIZE=1). Agent checks (bench/agent_check: 12 bug-fix
# tasks, a 12-minute cap a task, a 40-minute budget) on Flint Pascal Q8 GGUF (flint-near-q8-p065.gguf) at 262,144
# tokens on nine P106-090 cards, with Flint's v5 preset (release/flint-v5/settings.env) plus LLAMA_SCHED_PER_SIZE=1 and,
# with TRACE=1, LLAMA_UBATCH_TRACE=1, as R216a. The one difference from r216.sh: EXTRA, more switches appended after
# those (EXTRA=LLAMA_SCHED_PER_SIZE_PP=0 gives the per-size schedulers of before C903 on the same build, the control).
# Run from the repository's root on the machine that drives the rig (CLIENT=local makes it the client too: Docker, the
# sandbox image agent-check:0.87.1). Env: RIG, CLIENT, MODEL (flint-near-q8-p065.gguf on the rig), BIN (the C903
# build's llama-server on the rig), RIG_LOGDIR (a folder on the rig), RIG_PORT, RELAY_PORT, PROXY_PORT (and
# CLIENT_PORT unless CLIENT=local), CLIENT_DIR, CLIENT_OUT, OUT (here); TRACE (1 or 0); ARM (f5ppt: C903 traced; f5p0t:
# the control, with EXTRA=LLAMA_SCHED_PER_SIZE_PP=0); EXTRA; PREFIX (the client's container names); STEPS (run report
# collect).
set -uo pipefail
: "${OUT:?}" "${BIN:?}" "${TRACE:?}"
ARM=${ARM:-f5ppt}
LINE="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64"
SWITCHES="LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1"
. release/flint-v5/settings.env
[ "$PRESET_ARGS" = "$LINE" ] || { echo "release/flint-v5/settings.env: PRESET_ARGS is not the line R216 ran"; exit 1; }
[ "$PRESET_ENV" = "$SWITCHES" ] || { echo "release/flint-v5/settings.env: PRESET_ENV is not the switches R216 ran"; exit 1; }
ADDED="LLAMA_SCHED_PER_SIZE=1"
[ "$TRACE" = 1 ] && ADDED="$ADDED LLAMA_UBATCH_TRACE=1"
[ -n "${EXTRA:-}" ] && ADDED="$ADDED $EXTRA"
export BENCH=$PWD/bench/agent_check CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=${CPU_SHARES:-256} REL_BIN=$BIN \
  REL_ENV="$PRESET_ENV $ADDED" ARMS=$ARM STEPS=${STEPS:-run report collect}
export "ARGS_$ARM=--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  "BUDGET_MIN_$ARM=40"
exec bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh "$OUT"
