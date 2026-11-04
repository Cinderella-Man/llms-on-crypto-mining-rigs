#!/usr/bin/env bash
# rig_hops.sh: C200/C201's rig run, ON the rig and detached, from a folder that holds the bundle (the hop
# microbenchmark megakernel_p106 built from bench/megakernel_p106, this script, fn_line.sh and fn_requests.py from
# docs/data/2026-11-03_flash_next_baseline/run/). Ends with OUT/DONE.
#   1. the hop between two cards, for each pair in HOPS (default "0,1 3,4 4,5 3,5": two cards behind one switch, a
#      switch to the card on its own root port and back, two switches): megakernel_p106 --hop A,B --hop-only at the
#      sizes of the switches (one token's Flash-Next residual in Q8_0, F16 and F32: 10,880 / 20,480 / 40,960 B; a
#      4-token pass: 43,520 / 81,920 / 163,840 B), as host-staged copies, the driver's peer copy (the default
#      path), a mailbox and the fork's manual staged hop in 1, 2 and 4 chunks (GGML_CUDA_HOP_STAGED, _CHUNKS)
#   2. unless LINES=0: one server line per arm in ARMS (fn_line.sh, MODE=speed: decode at short context, fresh prompts
#      of ~1k, ~8k and ~32k tokens, two identity prompts at temperature 0 whose output hash compares the arms), the
#      build BUILD with the arm's switches:
#        f32       no switch (the hops as before: F32 residual, cudaMemcpyPeerAsync)
#        q8        LLAMA_HOP_CAST=q8_0                 the residual crosses each boundary in Q8_0
#        f16       LLAMA_HOP_CAST=f16                  ... in F16
#        staged    GGML_CUDA_HOP_STAGED=1              the manual staged hop (2 chunks; prompts on a copy stream)
#        q8staged  LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1
#        staged1   GGML_CUDA_HOP_STAGED=1 GGML_CUDA_HOP_CHUNKS=1
#        stageds0  GGML_CUDA_HOP_STAGED=1 GGML_CUDA_HOP_STREAM=0   (prompt hops on the compute stream too)
#      f32 and the staged arms copy the same bytes, so their identity hashes must agree; the cast arms may differ.
# Env: OUT (default ./out), BIN (default ./megakernel_p106), HOPS, LINES (1), ARMS (default "f32 q8 f16 staged
#      q8staged"), and for the lines BUILDS (folder of the builds), BUILD (the hops build's folder name), MODEL (first
#      shard), CTX (65536), ARGS (default '-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off'), PORT, SSD_DEV, USB_DEV;
#      FORCE=1 starts even with a compute process on the cards; ENV_ALL (default none) goes to every line's server before
#      the arm's switches (e.g. a kernel setting the comparison arms ran with)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$here/out}
BIN=${BIN:-$here/megakernel_p106}
HOPS=${HOPS-0,1 3,4 4,5 3,5}
LINES=${LINES:-1}
ARMS=${ARMS:-f32 q8 f16 staged q8staged}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off}
FORCE=${FORCE:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

declare -A ENV_OF=(
    [f32]="" [q8]="LLAMA_HOP_CAST=q8_0" [f16]="LLAMA_HOP_CAST=f16" [staged]="GGML_CUDA_HOP_STAGED=1"
    [q8staged]="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1" [staged1]="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_HOP_CHUNKS=1"
    [stageds0]="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_HOP_STREAM=0"
)

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$BIN" "$here"/fn_line.sh "$here"/fn_requests.py > "$OUT/md5.txt" 2>&1

log "1. hops between cards: ${HOPS:-none}"
for p in $HOPS; do
    tag=${p/,/-}
    "$BIN" --hop "$p" --hop-only --csv "$OUT/hop-$tag.csv" > "$OUT/hop-$tag.txt" 2>&1
    log "hop $p: exit $?; $(grep -E 'manual hop data|mailbox timeouts' "$OUT/hop-$tag.txt" | tr '\n' ' ')"
done

if [ "$LINES" = 1 ]; then
    : "${BUILDS:?BUILDS=<folder of the builds>}" "${BUILD:?BUILD=<the hops build>}" "${MODEL:?MODEL=<first shard>}"
    for arm in $ARMS; do
        [ -n "${ENV_OF[$arm]+x}" ] || { log "unknown arm $arm"; continue; }
        log "2. line $arm: ${ENV_OF[$arm]:-no switch}"
        LABEL=$arm BUILD=$BUILD BUILDS=$BUILDS MODEL=$MODEL CTX=$CTX ARGS="$ARGS" MODE=speed OUT=$OUT \
            ENVS="${ENV_ALL:+$ENV_ALL }${ENV_OF[$arm]}" bash "$here/fn_line.sh" > "$OUT/line-$arm.out" 2>&1
        # the switches' own log lines: the boundaries the cast found, each staged boundary's ring and its counts
        grep -E 'card hops in|card hop .* staged through|staged hops,' "$OUT/server-$arm.log" > "$OUT/hops-$arm.txt" 2>/dev/null
        log "line $arm: $(grep -E '^(decode|load|ident)' "$OUT/line-$arm.out" | cut -c1-120 | tr '\n' ' ')"
    done
fi
log "done"
touch "$OUT/DONE"
