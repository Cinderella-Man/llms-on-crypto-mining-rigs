#!/usr/bin/env bash
# R205 (Q1594), its Flint half: GGML_CUDA_MMID_GLU_UNFUSE unset, 1 and 2 on Flint Pascal Q8 GGUF (p065) as its v4 preset
# runs it (release/flint-v4/settings.env: Q8_0 and Q5_0 weights, F16 routers, one token a step with up to four
# pipelined guesses, --pipe-depth 4, no draft model) on nine P106-090 cards, all three arms on one build (a
# pascal-p106-v5 tree that carries the switch). This is the Flash-Next half's script
# (docs/data/2026-11-19_r205_flash_glu/run/r205.sh) with Flint's preset and file, no draft, and R176's watchdog floor
# for Flint on a 4 GB host (docs/data/2026-11-17_r176_flint_step_map/run/r176.sh: MIN_AVAIL_MB=60).
# Arms:
#   off  the switch unset: v4.0's fusions
#   u1   GGML_CUDA_MMID_GLU_UNFUSE=1: the routed experts' gate and up products (MUL_MAT_ID) as two launches, then the
#        GLU; the shared expert stays fused
#   u2   GGML_CUDA_MMID_GLU_UNFUSE=2: upstream's #32239, no fused mul_mat_vec_q on Pascal: the routed and the shared
#        expert's gate, up and GLU apart, and every product with its bias or scale
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (the recorded agent text here: the release check's
# proxy.jsonl.bodies.gz, as R176 and the Flash-Next half), PRESET (default release/flint-v4/settings.env), OUT (here,
# for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: loads warm:off, then off, u1, u2 twice, alternated; depths 0 and
#            8,192; 256 tokens at temperature 0 twice per depth, then a pass with the preset's sampler
#            (SAMPLER_PASS=1), every reply kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control off.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r205f}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SW=GGML_CUDA_MMID_GLU_UNFUSE
common() {
  export PRESET=${PRESET:-release/flint-v4/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} DRAFT=off \
    TEXT_LOCAL=${TEXT_LOCAL:?} TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
    SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 8192" \
    ARMS="off u1 u2" ARM_ENV_u1="$SW=1" ARM_ENV_u2="$SW=2" \
    LOADS="warm:off f1:off a1:u1 b1:u2 f2:off a2:u1 b2:u2"
}
case ${STEP:-start} in
dry) common; DRY=1 bash $L start ;;
start) common; bash $L start ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "STEP=dry|start|wait|status|collect|stop"; exit 1 ;;
esac
