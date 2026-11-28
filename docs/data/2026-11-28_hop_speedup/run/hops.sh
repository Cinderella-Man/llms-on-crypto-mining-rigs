#!/usr/bin/env bash
# Card-to-card hops on Qwen3.8-Flash-Next (2026-11-28): where a hop's time goes in the engine, and what the mailbox
# hop gives. The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table in a
# third shard, and the MTP draft, on; 131,072 tokens, -ub 256) on nine P106-090 cards, one build (a pascal-p106-v5
# tree), on the deep lane. In a layer split one card works at a time and hands its residual (four streams of 2,560
# values a token) to the next card through host memory, eight hops a pass: the preset casts it to Q8_0 at each card
# boundary (LLAMA_HOP_CAST=q8_0, 10,880 B a token) and sends it through a pinned host ring (GGML_CUDA_HOP_STAGED=1).
# Two lanes, each on one rig and boot, loads alternating, the second round in reverse order (Q1253):
#   LANE=bytes   how a hop's cost scales with its bytes, which decides what a faster link (Gen2, Q994) can give:
#                q8   the preset (control): Q8_0 hops through the staged ring, 10,880 B a token
#                f16  F16 hops through the staged ring, 20,480 B a token (KLD passed, Q231)
#                f32  no cast: F32 hops through the staged ring, 40,960 B a token (exact)
#                drv  Q8_0 hops through the driver's own staged peer copy (GGML_CUDA_HOP_STAGED unset)
#   LANE=mailbox the mailbox hop (Q682, C143): a kernel on each card through a ring in mapped pinned host memory, the
#                residual streamed in chunks with a sequence flag each, so the second leg starts on the first chunk:
#                q8   the preset (control)
#                mbd  the mailbox for hops up to 64 KiB (decode and verify passes; prompt micro-batches stay on the
#                     staged ring): GGML_CUDA_HOP_MAILBOX=1 GGML_CUDA_HOP_MAILBOX_MAX_KB=64
#                mba  the mailbox for every hop: GGML_CUDA_HOP_MAILBOX=1
#                mbg  the mailbox, hops above 64 KiB through its gate buffers ordered by events (C886, Q1606), 1 MiB
#                     each (4 MiB pinned a boundary instead of the default 16): GGML_CUDA_HOP_MAILBOX=1
#                     GGML_CUDA_HOP_MAILBOX_GATE_KB=64 GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB=1024
# Depths 0, 8,192 and 32,768 (the prompt read at each depth is the prompt measure; 256 tokens at temperature 0 twice per
# depth, then once with the preset's sampler, every reply kept).
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here, as R245), PRESET (default release/flash-next-v5/settings.env), OUT (here, for collect),
# LANE (bytes or mailbox). STEP: dry, start, wait, status, collect, stop (as R245's run/r245.sh).
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
: "${LANE:?LANE=bytes|mailbox}"
export RIG WORK=${WORK:-/tmp/hops-$LANE}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
CAST=LLAMA_HOP_CAST
STAGED=GGML_CUDA_HOP_STAGED
MB=GGML_CUDA_HOP_MAILBOX
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 8192 32768"
  case $LANE in
  bytes)
    export ENV_DROP="$CAST $STAGED" ARMS="q8 f16 f32 drv" \
      ARM_ENV_q8="$CAST=q8_0 $STAGED=1" ARM_ENV_f16="$CAST=f16 $STAGED=1" ARM_ENV_f32="$STAGED=1" \
      ARM_ENV_drv="$CAST=q8_0 $STAGED=0" \
      LOADS="warm:q8 q8a:q8 f16a:f16 f32a:f32 drva:drv drvb:drv f32b:f32 f16b:f16 q8b:q8" ;;
  mailbox)
    export ENV_DROP="" ARMS="q8 mbd mba mbg" \
      ARM_ENV_q8="$MB=0" ARM_ENV_mbd="$MB=1 ${MB}_MAX_KB=64" ARM_ENV_mba="$MB=1" \
      ARM_ENV_mbg="$MB=1 ${MB}_GATE_KB=64 ${MB}_GATE_BUF_KB=1024" \
      LOADS="warm:q8 q8a:q8 mbda:mbd mbaa:mba mbga:mbg mbgb:mbg mbab:mba mbdb:mbd q8b:q8" ;;
  *) echo "LANE=bytes|mailbox"; exit 1 ;;
  esac
}
case ${STEP:-start} in
dry) common; DRY=1 bash $L start ;;
start) common; bash $L start ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "STEP=dry|start|wait|status|collect|stop"; exit 1 ;;
esac
