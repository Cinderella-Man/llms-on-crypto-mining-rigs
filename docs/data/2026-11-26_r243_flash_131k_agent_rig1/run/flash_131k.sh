#!/usr/bin/env bash
# R243 (Q1921): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it stands against
# the same preset with a 131,072-token context and prompt micro-batches of 256 (-c 131072 -ub 256, R124's line:
# docs/data/2026-11-26_r124_flash_131k_rig2/), nothing else changed. Four runs on one rig, engine and boot,
# alternated: the preset, the line, the preset, the line.
# It starts the preset as C1175's arm r05 did (docs/data/2026-11-26_c1175_nsm10_rig2/run/flash_nsm.sh; that arm is the
# preset since 2026-11-26): C854's launcher (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver
# (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm v4r, then the report. R156's v4r line with
# LLAMA_SERVER_TEXT_REUSE=1 added to its V4R_ENV, GAINS_ENV's GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 replaced by
# =0.5 and GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 by =10, and V4C2_TAPER=short=1400:64 is the v5 preset.
# R156's driver fixes the context (CTX=65536: the server's -c and pi's context window) and the micro-batch (-ub 512 in
# its MT line), so the line's runs start a copy of it in WORK with those two values replaced, each exactly once, and a
# copy of C854's launcher that runs that copy; the preset's runs start the launcher itself.
#   ARM=ca|cb        the v5 preset (-c 65536 -ub 512)
#   ARM=l131a|l131b  the same with -c 131072 -ub 256 (pi's context window 131,072 with it)
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT; WORK is a local folder for the copies. Give every run its own OUT and CLIENT_OUT (the
# driver skips attempts it finds finished there) and, when another agent check shares the client, its own PREFIX and
# ports. DRY=1 prints the switches, the context and the micro-batch the run would use, writes the copies, and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
LAUNCHER=docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] || { echo "no V4R_ENV or GAINS_ENV default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
NSM=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
[[ " $GAINS_DEFAULT " == *" $NSM=5 "* ]] || { echo "the GAINS_ENV default has no $NSM=5"; exit 1; }
case ${ARM:?} in
  ca|cb) ctx=65536; ub=512 ;;
  l131a|l131b) ctx=131072; ub=256 ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
GAINS=${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=0.5"}
export GAINS_ENV="${GAINS/"$NSM=5"/"$NSM=10"}"
export V4C2_TAPER=short=1400:64
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
echo "$(date +%H:%M:%S) arm $ARM: context $ctx, micro-batch $ub"
run=$LAUNCHER
if [ "$ctx" != 65536 ] || [ "$ub" != 512 ]; then
  : "${WORK:?WORK: a local folder for the copies}"
  mkdir -p "$WORK"
  W=$(cd "$WORK" && pwd)
  d=$W/r156-$ARM.sh
  run=$W/flash_agent-$ARM.sh
  [ "$(grep -c '^export BENCH=.* CTX=65536 ' "$DRIVER")" = 1 ] || { echo "CTX=65536 not found once in $DRIVER"; exit 1; }
  [ "$(grep -c '^MT="-b 2048 -ub 512 ' "$DRIVER")" = 1 ] || { echo "the MT line's -ub 512 not found once in $DRIVER"; exit 1; }
  [ "$(grep -c "^exec timeout 5400 bash $DRIVER\$" "$LAUNCHER")" = 1 ] || { echo "the driver's call not found once in $LAUNCHER"; exit 1; }
  sed -e "/^export BENCH=/s/ CTX=65536 / CTX=$ctx /" -e "/^MT=\"/s/^MT=\"-b 2048 -ub 512 /MT=\"-b 2048 -ub $ub /" "$DRIVER" > "$d"
  sed "s#^exec timeout 5400 bash $DRIVER\$#exec timeout 5400 bash $d#" "$LAUNCHER" > "$run"
  # the copies differ from the files they copy in those lines only
  [ "$(diff "$DRIVER" "$d" | grep -c '^>')" = 2 ] && [ "$(diff "$LAUNCHER" "$run" | grep -c '^>')" = 1 ] \
    || { echo "the copies differ in more than the replaced lines"; diff "$DRIVER" "$d"; diff "$LAUNCHER" "$run"; exit 1; }
  echo "$(date +%H:%M:%S) arm $ARM: driver $d ($(md5sum < "$d" | cut -c1-32)), launcher $run"
  diff "$DRIVER" "$d" | sed "s/^/$(date +%H:%M:%S) arm $ARM: driver diff: /"
fi
[ "${DRY:-0}" = 1 ] && exit 0
exec bash "$run"
