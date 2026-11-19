#!/usr/bin/env bash
# R208 (Q1602) and R210 (Q1604): Flash-Next's agent check with the released v4.0 preset (release/flash-next-v4) on the
# v5 engine, alone or with one switch added. Six runs on one rig and one boot, alternated so that both switches share
# the two control runs: f5c1, f5s1, f5t1, f5c2, f5s2, f5t2.
# It runs C854's launcher (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh: R156's driver, harness arm v4r,
# then the report) with R156's V4R_ENV either left at the driver's default or extended by one switch:
#   ARM=f5c1|f5c2  the preset, nothing added (the control)
#   ARM=f5s1|f5s2  + GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all (C861: the K-quant prompt
#                  kernels' scale multiply as one FP32 multiply, for every K-quant type)
#   ARM=f5t1|f5t2  + LLAMA_SERVER_TEXT_REUSE=1 (a prompt whose text matches the cached reply but whose tokens split it
#                  differently keeps the cache instead of reading the reply again; R17a, R17b on Flint)
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT. Give every run its own CLIENT_OUT: the driver skips attempts it finds finished there.
# DRY=1 prints the V4R_ENV the run would use and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] || { echo "no V4R_ENV default in $DRIVER"; exit 1; }
case ${ARM:?} in
  f5c1|f5c2) unset V4R_ENV ;;
  f5s1|f5s2) export V4R_ENV="$V4R_DEFAULT GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all" ;;
  f5t1|f5t2) export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
if [ -n "${V4R_ENV+set}" ]; then shown=$V4R_ENV; else shown="$V4R_DEFAULT (unset: the default of the driver)"; fi
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$shown"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
