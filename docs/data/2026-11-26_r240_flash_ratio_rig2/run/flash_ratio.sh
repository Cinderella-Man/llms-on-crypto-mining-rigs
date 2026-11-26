#!/usr/bin/env bash
# R240 (Q1910): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it is, whose
# per-token sparse attention kernel takes four-token verifies from 4,102 cells (GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5,
# in 5 splits with _NSM=10), against the same preset with a lower switch point (_MIN_RATIO=0.375: from 3,077 cells;
# 0.25: from 2,051). Four runs on one rig, engine and boot, alternated: the control, the arm, the control, the arm.
# It starts the preset as C1175 did (docs/data/2026-11-26_c1175_nsm10_rig2/run/flash_nsm.sh, itself C912's and R229's
# line): R227's run/flash_taper.sh arm t1400a, that is C854's launcher
# (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh),
# the harness's arm v4r, then the report. R156's v4r line with LLAMA_SERVER_TEXT_REUSE=1 added to its V4R_ENV,
# GAINS_ENV's GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 and _NSM=5 replaced by the preset's values and V4C2_TAPER
# set to the preset's --ubatch-taper (read from release/flash-next-v5/settings.env: short=1400:64,cuts=1 since
# 2026-11-26 16:08, C905) is the v5 preset. Both switches live in GAINS_ENV, so an arm replaces the ratio there and
# nothing else:
#   ARM=ca|cb             the v5 preset (_NSM=10, _MIN_RATIO=0.5)
#   ARM=r<digits>a|b      the same with _MIN_RATIO=0.<digits>: r375a and r375b 0.375, r25a and r25b 0.25, r49a 0.49
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
  ca|cb) ratio=0.5 ;;
  r[0-9]*[ab]) d=${ARM#r}; d=${d%[ab]}; [[ $d =~ ^[0-9]+$ ]] || { echo "unknown ARM $ARM"; exit 1; }; ratio=0.$d ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
GAINS=${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=$ratio"}
export GAINS_ENV="${GAINS/"$NSM=5"/"$NSM=10"}"
TAPER=$(. release/flash-next-v5/settings.env && sed -n 's/.*--ubatch-taper \([^ ]*\).*/\1/p' <<< "$PRESET_ARGS")
[ -n "$TAPER" ] || { echo "no --ubatch-taper in release/flash-next-v5/settings.env"; exit 1; }
export V4C2_TAPER=$TAPER
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
[ "${DRY:-0}" = 1 ] && exit 0
exec bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
