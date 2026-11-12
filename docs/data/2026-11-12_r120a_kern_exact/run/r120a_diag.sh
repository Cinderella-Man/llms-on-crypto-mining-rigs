#!/usr/bin/env bash
# r120a_diag.sh: R120a's follow-up on the rig: is the preset's k-column budget (GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768)
# what makes Q3_K's (and Q2_0's) wide-row products of several tokens differ from their one-token products? With a budget
# set, pick_shape takes the shape of before Q823 within that budget and declines a product whose one iteration of
# activations does not fit; the declined product then runs on another kernel. The same harness and card as rig_kern.sh's
# bitid step, four arms: the k-column kernel and C726's and C757's kernels without the budget (the default shape rule,
# KCOL_TABLE=0 as in the preset), and the k-column kernel at the 48 KB budget of before Q823.
# Runs ON the rig, the rig idle, about a minute. Env: BITID (multitoken_bitid), OUT, DEV (card, default 0).
# Usage: BITID=... OUT=... bash r120a_diag.sh
set -uo pipefail
: "${BITID:?}" "${OUT:?}"
DEV=${DEV:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
P=GGML_CUDA_MMVQ_PASCAL
NT="${P}_KCOL_TABLE=0"                       # the preset's table switch, without its budget
RO="${P}_DMT=1 ${P}_Q5K_MT=1 ${P}_DMT_R=1"   # the read-once kernels C757's launch repeats
for a in "q3k:kcol-nobudget:$NT" "q3k:on-nobudget:$NT ${P}_Q3K_MT=1" "glu:on-nobudget:$NT $RO ${P}_GLU_MT=1" \
         "q3k:kcol-48k:${P}_KCOL_SMEM=49152 $NT"; do
    mode=${a%%:*} rest=${a#*:}
    n=${rest%%:*} sw=${rest#*:}
    log "bitid $mode $n start ($sw)"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $sw "$BITID" "$mode" > "$OUT/$mode-$n.txt" 2>&1
    log "bitid $mode $n exit $?: $(tail -n 1 "$OUT/$mode-$n.txt")"
done
