#!/usr/bin/env bash
# R227 (Q1641): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it is, which reads a
# prompt call of at most 600 tokens in micro-batches of 64 (--ubatch-taper short=600:64), against the same preset with
# --ubatch-taper short=1400:64 (calls of up to 1,400 tokens in 64s; R225, docs/data/2026-11-20_r225_flash_taper1400/:
# the agent check's 62 recorded prompts read 12.56 s a pass faster that way). Four runs on one rig, engine and boot,
# alternated: c600a, t1400a, c600b, t1400b.
# It runs C854's launcher (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh: R156's driver, harness arm v4r,
# then the report), as R220's run/flash_ratio.sh does. R156's v4r line with LLAMA_SERVER_TEXT_REUSE=1 added to its
# V4R_ENV and GAINS_ENV's GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 replaced by =0.75 is the v5 preset: the same 40
# switches in the same order as settings.env's PRESET_ENV, and its PRESET_ARGS (run/preset_check.sh checks a started
# command against settings.env).
#   ARM=c600a|c600b    the v5 preset (R156's taper V4C2_TAPER=short=600:64, the preset's own)
#   ARM=t1400a|t1400b  the same with V4C2_TAPER=short=1400:64, so the started command differs from the control's in
#                      that one value
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT. Give every run its own CLIENT_OUT (the driver skips attempts it finds finished there)
# and, when another agent check shares the client, its own PREFIX (the client's container names, acrel-<arm>-* by
# default). DRY=1 prints the GAINS_ENV, V4R_ENV and V4C2_TAPER the run would use and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
TAPER_DEFAULT=$(sed -n 's/^V4C2_TAPER=\${V4C2_TAPER:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] && [ -n "$TAPER_DEFAULT" ] \
  || { echo "no V4R_ENV, GAINS_ENV or V4C2_TAPER default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
[ "$TAPER_DEFAULT" = short=600:64 ] || { echo "the V4C2_TAPER default is $TAPER_DEFAULT, not short=600:64"; exit 1; }
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
export GAINS_ENV="${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=0.75"}"
case ${ARM:?} in
  c600a|c600b) export V4C2_TAPER=short=600:64 ;;
  t1400a|t1400b) export V4C2_TAPER=short=1400:64 ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
