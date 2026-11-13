#!/usr/bin/env bash
# R88 part (2a) (Q1315): the deep lane's three decode arms for C103's union gather on Miner Llama v4-rc2.
# The command of docs/data/2026-11-11_c103_union_gather/README.md ("Commands") with ARMS and LOADS cut to the three
# decode arms (ub512, r100, u150) and BIN the v4-rc2 build. The preset (release/flash-next-v3/settings.env) carries
# --spec-draft-n-ubatch 128, which the MTP draft needs on a build with upstream #32078
# (docs/data/2026-11-13_c831_upstream/README.md).
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on
# the rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (llama-server on the rig), MODEL (the
# model's first shard on the rig), MODEL_DRAFT (the MTP draft file on the rig), TEXT_LOCAL (the recorded agent text
# here), OUT (here, for collect). STEP=start|wait|status|collect; DRY=1 prints the three server lines and starts none.
set -uo pipefail
S=GGML_CUDA_FATTN_SPARSE_PASCAL
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r88}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 \
    DEPTHS="0 4096 8192 16384 32768 49152" N_PREDICT=256 ENV_ADD=${S}_NSM=5 \
    ARMS="ub512 r100 u150" \
    ARM_ENV_ub512="$S=1" \
    ARM_ENV_r100="$S=1 ${S}_MIN_RATIO=1.0" \
    ARM_ENV_u150="$S=1 ${S}_UNION=1" \
    LOADS="warm:ub512 ub512_1:ub512 r100_1:r100 u150_1:u150 ub512_2:ub512 r100_2:r100 u150_2:u150" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
