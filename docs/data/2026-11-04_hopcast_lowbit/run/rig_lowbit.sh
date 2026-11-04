#!/usr/bin/env bash
# rig_lowbit.sh: C408's rig run (Q963), ON the rig and detached, from a folder that holds fn_line.sh and fn_requests.py
# (docs/data/2026-11-03_flash_next_baseline/run/) and this script. One llama-server line per arm (fn_line.sh,
# MODE=speed: decode at short context, fresh prompts of ~1k, ~8k and ~32k tokens and one from a checkpoint, two
# identity prompts at temperature 0), as C200r ran its f32 and q8 lines (../../2026-11-04_card_hops/run/rig_hops.sh),
# on the build BUILD with the arm's cast:
#   f32      no LLAMA_HOP_CAST (the residual crosses in F32, 40,960 B a token)
#   <type>   LLAMA_HOP_CAST=<type>: q8_0 (10,880 B a token, the release line's cast), q6_K (8,400), q5_1 (7,680),
#            q5_0 (7,040), q4_1 (6,400), q4_0 or iq4_nl (5,760)
# Run only the types whose KL divergence passed the gate (gate_q2_0.tsv). The cast arms' identity hashes differ from
# f32's and from each other, as a cast's must.
# Env: OUT (default ./out), ARMS (default "q8_0 q6_K q5_0 q4_0"), BUILDS (folder of the builds), BUILD (the build's
#      folder name), MODEL (first shard), CTX (65536), ARGS (default '-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off'),
#      ENV_ALL (default 'GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 GGML_CUDA_HOP_STAGED=1':
#      the release line's kernel settings and staged hop, before every arm's cast), PORT, SSD_DEV, USB_DEV; FORCE=1
#      starts even with a compute process on the cards. Ends with OUT/DONE.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$here/out}
ARMS=${ARMS:-q8_0 q6_K q5_0 q4_0}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off}
ENV_ALL=${ENV_ALL-GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 GGML_CUDA_HOP_STAGED=1}
FORCE=${FORCE:-0}
: "${BUILDS:?BUILDS=<folder of the builds>}" "${BUILD:?BUILD=<the hopcast4 build>}" "${MODEL:?MODEL=<first shard>}"
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$here"/fn_line.sh "$here"/fn_requests.py "$0" > "$OUT/md5.txt" 2>&1

for arm in $ARMS; do
    case $arm in
        f32) e="" ;;
        f16|bf16|q8_0|q6_K|q6_k|q5_1|q5_0|q4_1|q4_0|iq4_nl) e="LLAMA_HOP_CAST=$arm" ;;
        *) log "unknown arm $arm"; continue ;;
    esac
    log "line $arm: ${ENV_ALL:+$ENV_ALL }${e:-no cast}"
    LABEL=$arm BUILD=$BUILD BUILDS=$BUILDS MODEL=$MODEL CTX=$CTX ARGS="$ARGS" MODE=speed OUT=$OUT \
        ENVS="${ENV_ALL:+$ENV_ALL }$e" bash "$here/fn_line.sh" > "$OUT/line-$arm.out" 2>&1
    # the cast's own log line: the type and the boundaries it found
    grep -E 'card hops in|card hop .* staged through|staged hops,' "$OUT/server-$arm.log" > "$OUT/hops-$arm.txt" 2>/dev/null
    log "line $arm: $(grep -E '^(decode|load|ident)' "$OUT/line-$arm.out" | cut -c1-120 | tr '\n' ' ')"
done
log "done"
touch "$OUT/DONE"
