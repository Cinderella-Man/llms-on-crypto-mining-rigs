#!/usr/bin/env bash
# rig_c811.sh: C811 (Q1417) on a P106 rig, the rig row R131. Runs ON the rig, detached, the rig idle; ends with
# $OUT/DONE (written by rig_loads.sh).
#   1. test-cuda-keepalive gpu on one card (CUDA_VISIBLE_DEVICES=$KA_CARD, card 0 by default), GGML_CUDA_KEEPALIVE=decode
#      and GGML_CUDA_KEEPALIVE_MS=2000, three processes: the switch unset (unset), GGML_CUDA_KEEPALIVE_EAGER=1 (eager1)
#      and CUDA_MODULE_LOADING=EAGER with the switch unset (loadeager). Each times a graph's first call of kernels not
#      launched before (sqr, sin), right after a graph whose keep-alive block runs, and a memset and a set at sizes new
#      to the card. eager1 must pass (no wait); unset shows whether a first launch waits for the block under lazy
#      module loading; loadeager whether loading every kernel at the start removes it. OUT/ka-gpu-<arm>.log.
#   2. R122's loads at rung f (docs/data/2026-11-12_r122_host_ladder/run/r122_loads.sh's F) through rig_loads.sh
#      (RIG_LOADS, the C615 driver R122 ran), with GGML_CUDA_KEEPALIVE_EAGER at 0 (the counts line only: f0, f0-2) and at
#      1 (f1, f1-2): the host profile, the server log and both requests per load, as R122's.
# Usage: as rig_loads.sh's (BIN, MODEL, DRAFT, TABLE, PRESET, RIG_SERVER, VM_SAMPLER, REQ_GREEDY, REQ_LONG, OUT), plus
#   [KA_CARD=0] [LOADS='f0 f1 f0-2 f1-2'] bash rig_c811.sh
# Read: FN85's run/tables.sh on OUT (the waits per load and request: none in generation with the switch at 1), the
# server logs' "keep-alive =" lines (the counts after each reply), S37's step split for the step.
set -uo pipefail
: "${BIN:?}" "${RIG_LOADS:?}" "${OUT:?}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
KA_CARD=${KA_CARD:-0}
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
log "C811 rig check: $BIN, test-cuda-keepalive md5 $(md5sum < "$BIN/test-cuda-keepalive" | cut -d' ' -f1), llama-server md5 $(md5sum < "$BIN/llama-server" | cut -d' ' -f1)"
for arm in "unset:" "eager1:GGML_CUDA_KEEPALIVE_EAGER=1" "loadeager:CUDA_MODULE_LOADING=EAGER"; do
    tag=${arm%%:*} envs=${arm#*:}
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$KA_CARD" GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_MS=2000 $envs \
        timeout 120 "$BIN/test-cuda-keepalive" gpu > "$OUT/ka-gpu-$tag.log" 2>&1
    log "test-cuda-keepalive gpu $tag: exit $?; $(grep -E '^gpu: graph B|^test-cuda-keepalive gpu' "$OUT/ka-gpu-$tag.log" | tr '\n' ' ')"
    sleep 3
done
F=LLAMA_DRAFT_PREFETCH=1,LLAMA_VERIFY_PREPARE=2,LLAMA_MTP_GRAPH_KEEP=1,LLAMA_KV_SEQ_RM_BOUND=1,LLAMA_SERVER_YIELD=2
F=$F,LLAMA_MTP_EMBD_FIRST=1,LLAMA_MTP_DRAFT_CAP=1,LLAMA_MTP_SKIP_LAST_ROW=1
export LOADS=${LOADS:-f0 f1 f0-2 f1-2}
export LOAD_ENV="f0:$F,GGML_CUDA_KEEPALIVE_EAGER=0 f1:$F,GGML_CUDA_KEEPALIVE_EAGER=1 f0-2:$F,GGML_CUDA_KEEPALIVE_EAGER=0 f1-2:$F,GGML_CUDA_KEEPALIVE_EAGER=1"
exec bash "$RIG_LOADS"
