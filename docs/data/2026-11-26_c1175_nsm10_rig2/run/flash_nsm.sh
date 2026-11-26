#!/usr/bin/env bash
# C1175 (Q1908): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it is, whose
# per-token sparse attention kernel splits a four-token verify into 3 pieces (GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5),
# against the same preset with the kernel's splits counted for 10 SMs (_NSM=10: 5 pieces at 4 tokens), alone or with
# the sparse kernel taking four-token verifies from 4,102 cells instead of 6,153 (_MIN_RATIO=0.5). Four runs on one
# rig, engine and boot, alternated: the control, the arm, the control, the arm.
# It starts the preset as C912 and R229 did (docs/data/2026-11-24_c912_flash_agent/,
# docs/data/2026-11-25_r229_flash_shortlist_agent/): R227's run/flash_taper.sh arm t1400a, that is C854's launcher
# (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh),
# the harness's arm v4r, then the report. R156's v4r line with LLAMA_SERVER_TEXT_REUSE=1 added to its V4R_ENV,
# GAINS_ENV's GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 replaced by =0.75 and V4C2_TAPER=short=1400:64 is the v5
# preset. Both switches live in GAINS_ENV, so an arm replaces its values there and nothing else:
#   ARM=ca|cb      the v5 preset (_NSM=5, _MIN_RATIO=0.75)
#   ARM=n10a|n10b  the same with _NSM=10
#   ARM=r05a|r05b  the same with _NSM=10 and _MIN_RATIO=0.5
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT. Give every run its own OUT and CLIENT_OUT (the driver skips attempts it finds
# finished there) and, when another agent check shares the client, its own PREFIX (the client's container names,
# acrel-<arm>-* by default). DRY=1 prints the GAINS_ENV, V4R_ENV and V4C2_TAPER the run would use and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] || { echo "no V4R_ENV or GAINS_ENV default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
NSM=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
[[ " $GAINS_DEFAULT " == *" $NSM=5 "* ]] || { echo "the GAINS_ENV default has no $NSM=5"; exit 1; }
case ${ARM:?} in
  ca|cb) nsm=5; ratio=0.75 ;;
  n10a|n10b) nsm=10; ratio=0.75 ;;
  r05a|r05b) nsm=10; ratio=0.5 ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
GAINS=${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=$ratio"}
export GAINS_ENV="${GAINS/"$NSM=5"/"$NSM=$nsm"}"
export V4C2_TAPER=short=1400:64
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
