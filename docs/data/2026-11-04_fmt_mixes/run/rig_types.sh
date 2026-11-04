#!/usr/bin/env bash
# C323 (Q901) on one P106-090 of a rig: the dense types a verify-shaped mix may choose, timed at Qwen3.8-Flash-Next's
# dense shapes at 1 to 8 tokens by bench/mmvq_pascal's mmvq_kcol built with FMT=1 (Q8_0, Q6_K, Q5_K, Q5_0 and Q4_0 at
# every dense shape, beside the Swift Q2_0 census types at the same shapes, in the same run): each call is the
# activation quantizer and the product, as in the model; every variant's rows are checked against a double-precision
# reference (the `check` column). bench/pascal_mix/vmix.py times --bench reads the CSV. ~20 minutes; nothing else may
# run on the rig.
# Env: MMVQ (the FMT=1 mmvq_kcol on the rig), OUT (output folder on the rig), CARD (default 0), TYPES, SHAPES and TOKENS
# (defaults below).
# Usage: ssh $RIG "MMVQ=<dir>/mmvq_kcol OUT=<dir>/out setsid nohup bash -s > <dir>/rig_types.out 2>&1" < run/rig_types.sh
set -uo pipefail
MMVQ=${MMVQ:?MMVQ=<mmvq_kcol binary, built with FMT=1>}
OUT=${OUT:?OUT=<output folder>}
CARD=${CARD:-0}
TYPES=${TYPES:-q8_0,q6_K,q5_K,q5_0,q4_0,iq4_xs,q4_K,q3_K,q2_0,iq4_nl}
SHAPES=${SHAPES:-attn_qkv,attn_gate,attn_q,ssm_out,attn_output,shexp_gate,shexp_up,shexp_down,ple_key}
TOKENS=${TOKENS:-1,2,3,4,5,6,7,8}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
log() { echo "$*" | tee -a "$OUT/console.txt"; }
quiet() {
    log "== $(date -Is) $1: $(ps -eo comm | grep -cE '^(llama-|test-backend|mmvq_)' || true) GPU processes," \
        "load $(cut -d' ' -f1 /proc/loadavg), card $CARD" \
        "$(nvidia-smi --id="$CARD" --query-gpu=memory.used,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader)"
}
quiet "start"
grep -q KCOL_BENCH_FMT "$(dirname "$MMVQ")/BUILDINFO_kcol.txt" 2> /dev/null ||
    log "note: $(dirname "$MMVQ")/BUILDINFO_kcol.txt does not name KCOL_BENCH_FMT; the new shapes may be missing"
"$MMVQ" --types "$TYPES" --shapes "$SHAPES" --tokens "$TOKENS" --csv "$OUT/kcol_fmt.csv" > "$OUT/kcol_fmt.txt" 2>&1
rc=$?
log "mmvq_kcol: exit $rc, $(tail -n 1 "$OUT/kcol_fmt.txt")"
log "rows: $(($(wc -l < "$OUT/kcol_fmt.csv") - 1)), failed checks: $(grep -c ',FAIL' "$OUT/kcol_fmt.csv" || true)"
quiet "end"
date -Is > "$OUT/done"
exit $rc
