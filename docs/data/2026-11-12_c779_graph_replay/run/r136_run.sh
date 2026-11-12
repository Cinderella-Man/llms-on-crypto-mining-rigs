#!/usr/bin/env bash
# r136_run.sh: R136, C779's GGML_CUDA_GRAPHS_PATCH on a P106 rig, against GGML_CUDA_GRAPHS_FORCE=1 alone (the force
# switch is in v4c2's line since R123a). From a machine that only runs ssh, at the repository's root. Three parts, each a
# run of its own; every load starts on a quiet rig and nothing is read from the rig while one runs.
#   STEP=lane           1. identity, untimed (R82's lane): the v4 candidate with the force switch, then with the patch
#                       too, 16 tokens at depths 0 and 32,768 (the 32,768-token prime is 64 forced micro-batches), every
#                       verify step dumped per load (bench/deep_lane, detached); the patch must give the force load's
#                       replies and dumps
#   STEP=lanewait       one ssh call that returns when the lane has ended
#   STEP=lanecollect OUT=<folder>   the lane's results to OUT (kept on the rig)
#   STEP=r42 ARM=force|patch   2. R42's driver (three loads: unprofiled, graph profile, host profile; a task's replay
#                       and two fresh 4,097-token prompts with the taper off and on) with the force switch, then with
#                       the patch too; detached, it ends with $WORK/r42-$ARM/DONE. The server log's
#                       "GGML_CUDA_GRAPHS_PATCH: ... captures patched" lines count the patches and full captures
#   STEP=agent ARM=force|patch   3. R43's replay of an agent session's 62 prompts (two passes) on v4c2's line: R123b's
#                       kept line (R123a's line, LLAMA_DRAFT_EMBD_DEV=layer without -otd token_embd=CUDA4,
#                       GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1, -b 4096), then with the patch too; detached, it ends with
#                       $R43/r136_$ARM/DONE
# Env: RIG (the rig's ssh alias), WORK (a folder on the rig), BUILDS (the builds folder on the rig), BUILD (the -c779
# tarball's folder in it), MODELS (the rig folder with swift_q2_0/ and the MTP draft), R42 (the rig folder with
# r42_run.sh), R43 (R43's folder on the rig: r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv, settings.env),
# RECORDED (the release check's recorded agent text here, proxy.jsonl.bodies.gz; lane only). DRY=1 prints the rig
# command of r42 and agent only. ARM=patchw (and the lane's LANE_LOADS="force1:force patchw1:patchw") runs the patch with
# GGML_CUDA_GRAPHS_PATCH_WINDOW=$W (8 by default): fewer groups dispatched again, if the dumps stay the force load's.
set -uo pipefail
: "${STEP:?}" "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODELS:?}"
B=$BUILDS/$BUILD/bin
MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf
DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf
FORCE=GGML_CUDA_GRAPHS_FORCE=1
PATCH="GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1"
V4C_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
V4C_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512"
arm_env() {
    case "${ARM:?ARM=force|patch|patchw}" in
        force) echo "$FORCE" ;;
        patch) echo "$PATCH" ;;
        patchw) echo "$PATCH GGML_CUDA_GRAPHS_PATCH_WINDOW=${W:-8}" ;;
        *) echo "ARM is force, patch or patchw" >&2; exit 1 ;;
    esac
}
case $STEP in
lane)
  RIG=$RIG WORK=$WORK/lane BIN=$B/llama-server MODEL=$MODEL MODEL_DRAFT=$DRAFT \
    TEXT_LOCAL=${RECORDED:?}/proxy.jsonl.bodies.gz TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 N_PREDICT=16 DEPTHS="0 32768" \
    ARMS="force patch patchw" \
    ARM_ENV_force="$V4C_ENV $FORCE LLAMA_SPEC_DUMP=$WORK/lane/out/dump-force.jsonl" ARM_ARGS_force="$V4C_ARGS" \
    ARM_ENV_patch="$V4C_ENV $PATCH LLAMA_SPEC_DUMP=$WORK/lane/out/dump-patch.jsonl" ARM_ARGS_patch="$V4C_ARGS" \
    ARM_ENV_patchw="$V4C_ENV $PATCH GGML_CUDA_GRAPHS_PATCH_WINDOW=${W:-8} LLAMA_SPEC_DUMP=$WORK/lane/out/dump-patchw.jsonl" ARM_ARGS_patchw="$V4C_ARGS" \
    LOADS="${LANE_LOADS:-force1:force patch1:patch}" \
    bash bench/deep_lane/remote.sh start
  ;;
lanewait) RIG=$RIG WORK=$WORK/lane bash bench/deep_lane/remote.sh wait ;;
lanecollect) RIG=$RIG WORK=$WORK/lane CLEAN=0 OUT=${OUT:?} bash bench/deep_lane/remote.sh collect ;;
r42)
  : "${R42:?}" "${R43:?}"
  e=$(arm_env) || exit 1
  cmd="mkdir -p $WORK && cd $R42 && env $e R43=$R43 BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT \
    OUT=$WORK/r42-$ARM R42_LOADS='warm4096 graph4096 host4096' setsid nohup bash $R42/r42_run.sh > $WORK/r42-$ARM.out 2>&1 < /dev/null &"
  echo "$cmd"
  [ "${DRY:-0}" = 1 ] && exit 0
  timeout 30 ssh "$RIG" "$cmd" || true
  ;;
agent)
  : "${R43:?}"
  e=$(arm_env) || exit 1
  # R123b's kept line (docs/data/2026-11-12_r123b_prompt_ladder/run/r123b_loads.sh: base, embd, sparse and b4096; not
  # queues), the force switch in it; the patch arm adds GGML_CUDA_GRAPHS_PATCH=1
  LINE_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1"
  HOST_ENV="LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1"
  KEPT_ENV="LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1"
  ENV="$LINE_ENV $HOST_ENV $KEPT_ENV $e"
  TAPER=short=600:64
  ARGS_ADD="--spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper $TAPER -b 4096"
  OUTN=r136_$ARM
  MK="sed 's/ -otd token_embd=CUDA4//' r43_run.sh > r43_run_noemb.sh && ! grep -q 'token_embd=CUDA4' r43_run_noemb.sh"
  cmd="cd $R43 && $MK && export $ENV && { md5sum r43_run_noemb.sh; grep -m1 '^export ARGS=' r43_run_noemb.sh; env | grep -E '^(LLAMA_|GGML_|CUDA_)' | sort; } > $R43/$OUTN.launch && \
BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$R43/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A='$TAPER $TAPER' FRESH=0 setsid nohup bash $R43/r43_run_noemb.sh > $R43/$OUTN.out 2>&1 < /dev/null &"
  echo "$cmd"
  [ "${DRY:-0}" = 1 ] && exit 0
  # an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
  timeout 20 ssh "$RIG" "$cmd" || true
  echo "started $OUTN at $(date -Is); wait for $R43/$OUTN/DONE"
  ;;
*) echo "STEP=lane|lanewait|lanecollect|r42|agent"; exit 1 ;;
esac
