#!/usr/bin/env bash
# C1173 (Q1906): Flash-Next's agent check with the better-quality file gmix-q2 on the hand-written PTX kernels of its
# 2-bit grid formats (GGML_CUDA_MTOK_IQ_FAST=2; C1146, docs/data/2026-11-26_c1146_iq_kernels_rig1/: a step 11.66 and
# 11.78 ms shorter, the same replies), against the file pmix-q2-fast, both with Flash-Next's v5 preset
# (release/flash-next-v5/settings.env). Four runs on one rig, engine and boot, alternated: gmixa, pmixa, gmixb, pmixb.
# It starts the preset as R229's preset arm and C912 did (docs/data/2026-11-25_r229_flash_shortlist_agent/): R227's
# t1400a line (docs/data/2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh), that is C854's launcher
# (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh),
# the harness's arm v4r, then the report. R156's v4r line with LLAMA_SERVER_TEXT_REUSE=1 added to its V4R_ENV, GAINS_ENV's
# GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 replaced by =0.75 and V4C2_TAPER=short=1400:64 is the v5 preset: the
# same 40 switches in the same order as settings.env's PRESET_ENV, and its PRESET_ARGS (run/preset_check.sh checks a
# started command against settings.env).
#   ARM=gmixa|gmixb  MODEL_G (gmix-q2's first shard), V4R_ENV with GGML_CUDA_MTOK_IQ_FAST=2 added at its end, so the
#                    started command is the preset's with that one switch after the last
#   ARM=pmixa|pmixb  MODEL_P (pmix-q2-fast's first shard), the preset's command as it is
# The model reaches the rig as MODEL (the server's -m, which the started command does not print; each server log's
# general.name and file lines name the file loaded).
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and RIG_PORT
# RELAY_PORT PROXY_PORT; here MODEL_G and MODEL_P in place of MODEL. Give every run its own OUT and CLIENT_OUT (the
# driver skips attempts it finds finished there) and, when another agent check shares the client, its own PREFIX (the
# client's container names). DRY=1 prints the environment the run would use and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
IQ=GGML_CUDA_MTOK_IQ_FAST=2
case ${ARM:?} in
  gmixa|gmixb) export MODEL=${MODEL_G:?}; ADD=" $IQ" ;;
  pmixa|pmixb) export MODEL=${MODEL_P:?}; ADD= ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
# R227's t1400a arm (run/flash_taper.sh): R156's v4r line with text reuse, the sparse ratio 0.75 and the 1,400 taper
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] || { echo "no V4R_ENV or GAINS_ENV default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1$ADD"
export GAINS_ENV="${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=0.75"}"
export V4C2_TAPER=short=1400:64
echo "$(date +%H:%M:%S) arm $ARM: MODEL=$MODEL"
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
