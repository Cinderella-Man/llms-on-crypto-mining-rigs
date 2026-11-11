#!/usr/bin/env bash
# c729_r42.sh: R116's prompt timing for C729 (Q1288), R42's driver (2026-11-10_prompt_timeline/run/r42_run.sh) on C729's
# build, from a machine that only runs ssh, as 2026-11-11_r82_c702_p106/run/r82_run.sh's r42 step. Both arms export the 2-4-token read-once switches
# (GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1, the v4 line's); ARM=on also exports
# GGML_CUDA_MMVQ_PASCAL_RO8=$RO8. Three loads per arm (R42_LOADS: unprofiled, graph profile, host profile), each the
# replay of one agent-check task (its reset and five turns) and two fresh 4,097-token prompts with the micro-batch taper
# off and on: with the MTP draft every micro-batch of a planned split runs at least five tokens (C678), so the taper's
# last micro-batches are the five-to-eight-token passes this switch takes. Detached on the rig; ends with
# $WORK/r42-$ARM/DONE. Then R42's timeline.py per arm on a build machine, as R82 ran it.
# Env: RIG (the rig's ssh alias), WORK (a folder on the rig), BUILDS (the builds folder on the rig), BUILD (the build's
# folder name in it), MODELS (the rig folder with swift_q2_0/ and the MTP draft), R42 (the rig folder with r42_run.sh),
# R43 (R43's folder on the rig), ARM (off or on), RO8 (default 1; R116 sets the perf step's faster of 1 and 2 at 5 tokens)
set -uo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODELS:?}" "${R42:?}" "${R43:?}" "${ARM:?}"
RO8=${RO8:-1}
e="GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1"
case $ARM in
    off) ;;
    on)  e="$e GGML_CUDA_MMVQ_PASCAL_RO8=$RO8" ;;
    *)   echo "ARM=off|on"; exit 1 ;;
esac
timeout 30 ssh "$RIG" "mkdir -p $WORK && cd $R42 && env $e R43=$R43 BUILDS=$BUILDS BUILD=$BUILD \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/r42-$ARM R42_LOADS='warm4096 graph4096 host4096' \
  setsid nohup bash $R42/r42_run.sh > $WORK/r42-$ARM.out 2>&1 < /dev/null &"
