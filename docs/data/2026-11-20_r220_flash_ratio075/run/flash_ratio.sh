#!/usr/bin/env bash
# R220 (Q1635): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it is, where the
# sparse decode attention takes over from a cache of 1.0 × the kept cells, against the same preset with
# GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75 (R69, docs/data/2026-11-20_r69_sparse_switch/: a four-token verify pass
# moves to the sparse kernel from 6,153 cached cells instead of 8,204). Four runs on one rig, engine and boot,
# alternated: c10a, r075a, c10b, r075b.
# It runs C854's launcher (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh: R156's driver, harness arm v4r,
# then the report). R156's v4r line with LLAMA_SERVER_TEXT_REUSE=1 added to its V4R_ENV is the v5 preset: the same 40
# switches in the same order as settings.env's PRESET_ENV, and its PRESET_ARGS (run/preset_check.sh checks a started
# command against settings.env). This is R210's text-reuse arm, which became the preset.
#   ARM=c10a|c10b    the v5 preset (R156's GAINS_ENV left at its default, ratio 1.0)
#   ARM=r075a|r075b  the same with GAINS_ENV's GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 replaced by =0.75, so the
#                    started command differs from the control's in that one value
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT. Give every run its own CLIENT_OUT (the driver skips attempts it finds finished there)
# and, when another agent check shares the client, its own PREFIX (the client's container names, acrel-<arm>-* by
# default). DRY=1 prints the GAINS_ENV and V4R_ENV the run would use and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] || { echo "no V4R_ENV or GAINS_ENV default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
case ${ARM:?} in
  c10a|c10b) unset GAINS_ENV ;;
  r075a|r075b) export GAINS_ENV="${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=0.75"}" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
if [ -n "${GAINS_ENV+set}" ]; then shown=$GAINS_ENV; else shown="$GAINS_DEFAULT (unset: the default of the driver)"; fi
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$shown"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
