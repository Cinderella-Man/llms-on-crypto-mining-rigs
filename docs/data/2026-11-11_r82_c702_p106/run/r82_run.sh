#!/usr/bin/env bash
# r82_run.sh: R82 (Q1297), C702's prompt graphs (GGML_CUDA_GRAPHS_FORCE=1) on a P106 rig. From a machine that only runs
# ssh, at the repository's root; every step but `r42` waits in one ssh call or reads files here.
#   STEP=lane       the identity, untimed: the v4 candidate (R70's five switches) eagerly twice and with the switch once,
#                   16 tokens at depths 0 and 32,768, every verify step dumped per load (bench/deep_lane, detached)
#   STEP=lanewait   one ssh call that returns when the lane has ended
#   STEP=lanecollect OUT=<folder>   the lane's results to OUT (kept on the rig)
#   STEP=r42 ARM=off|on   R42's driver on the build, three loads (unprofiled, graph profile, host profile), detached;
#                   ARM=on exports GGML_CUDA_GRAPHS_FORCE=1; it ends with $WORK/r42-$ARM/DONE
# Env: RIG (the rig's ssh alias), WORK (a folder on the rig), BUILDS (the builds folder on the rig), BUILD (the build's
# folder name in it), MODELS (the rig folder with swift_q2_0/ and the MTP draft), R42 (the rig folder with
# r42_run.sh), R43 (R43's folder on the rig: r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv, settings.env),
# RECORDED (the release check's recorded agent text here, proxy.jsonl.bodies.gz).
set -uo pipefail
: "${STEP:?}" "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODELS:?}"
B=$BUILDS/$BUILD/bin
V4C_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
V4C_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512"
case $STEP in
lane)
  RIG=$RIG WORK=$WORK/lane BIN=$B/llama-server \
    MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
    MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
    TEXT_LOCAL=${RECORDED:?}/proxy.jsonl.bodies.gz TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 N_PREDICT=16 DEPTHS="0 32768" \
    ARMS="eager graphs eagerb" \
    ARM_ENV_eager="$V4C_ENV LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eager.jsonl" ARM_ARGS_eager="$V4C_ARGS" \
    ARM_ENV_graphs="$V4C_ENV GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-graphs.jsonl" ARM_ARGS_graphs="$V4C_ARGS" \
    ARM_ENV_eagerb="$V4C_ENV LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eagerb.jsonl" ARM_ARGS_eagerb="$V4C_ARGS" \
    LOADS="eager1:eager graphs1:graphs eager2:eagerb" \
    bash bench/deep_lane/remote.sh start
  ;;
lanewait) RIG=$RIG WORK=$WORK/lane bash bench/deep_lane/remote.sh wait ;;
lanecollect) RIG=$RIG WORK=$WORK/lane CLEAN=0 OUT=${OUT:?} bash bench/deep_lane/remote.sh collect ;;
r42)
  : "${R42:?}" "${R43:?}"
  e=; [ "${ARM:?}" = on ] && e="GGML_CUDA_GRAPHS_FORCE=1"
  timeout 30 ssh "$RIG" "mkdir -p $WORK && cd $R42 && env $e R43=$R43 BUILDS=$BUILDS BUILD=$BUILD \
    MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
    DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/r42-$ARM R42_LOADS='warm4096 graph4096 host4096' \
    setsid nohup bash $R42/r42_run.sh > $WORK/r42-$ARM.out 2>&1 < /dev/null &"
  ;;
*) echo "STEP=lane|lanewait|lanecollect|r42"; exit 1 ;;
esac
