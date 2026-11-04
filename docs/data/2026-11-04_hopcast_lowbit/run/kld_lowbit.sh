#!/usr/bin/env bash
# C408 (Q963): the card-hop cast below 8 bits on the real model, on the GPUs of a build machine, with the RTX 30 build
# of pascal-p106-v3-b12076-hopcast4 (cpu_checks.sh STEPS=sm86). Three steps, each skipped when its output is complete:
#   gpucheck    test-backend-ops -b CUDA0 -o CPY (every copy case, the hop shapes of all nine types among them: the card
#               against the CPU) and test-hop-cast --unit-only (the round trips on the card and the CPU; the card's
#               against the CPU's, bit for bit or not)
#   boundaries  test-hop-cast on MODEL, the round trip at the rig's eight boundaries (AT) in each type of TYPES: per
#               boundary the residual's largest |x| (F16 stops at 65,504) and rms and the round trip's largest and rms
#               error (C204 / Q757 for F16, BF16 and Q8_0 too), every hop within the type's bound, and the logits' KL
#               divergence from the arm off over a short sequence
#   kld         C200k's KL divergence (../../2026-11-04_card_hops/run/kld_hops.sh, generalised to any type): the
#               quality chain's kld.sh, unchanged, on its bases (../../2026-11-03_flash_next_quality), each arm the round
#               trip at AT in one type, and the arm off (the switch unset) on the same cards; then the table paired by
#               chunk against off (../../2026-11-04_moe_tail_kld/run/table.sh) and the gate per arm and text
# The rig's boundaries: Flash-Next on nine cards with -ts 6,5,5,6,5,5,6,5,6 puts layers 0-5, 6-10, 11-15, 16-21,
# 22-26, 27-31, 32-37, 38-42 and 43-47 on cards 0-8, so the residual crosses before layers 6,11,16,22,27,32,38,43. The
# build machine's own split has other boundaries, which stay F32 (LLAMA_HOP_CAST_AT marks the casts' layers instead).
# Gate (C200k's): an arm passes a text when its paired change over off is within noise, d - 2 se <= 0.
#   (env: BIN (the sm86 build's bin) BASES (base.<text>.kld and gpu_args) MODEL OUT GPUS MOUNTS WIKI AGENT CODE TAG
#    [ARMS="off q6_K q5_1 q5_0 q4_1 iq4_nl q4_0"] [TYPES="f16 bf16 q8_0 q6_K q5_1 q5_0 q4_1 q4_0 iq4_nl"] [AT=6,11,16,22,27,32,38,43]
#    [STEPS="gpucheck boundaries kld"] [TEXTS] [IMAGE] [PREFLIGHT])
#   bash kld_lowbit.sh
# Output: OUT/gpucheck/{ops.log,unit.log}, OUT/boundaries/{hops.log,hops.tsv}, OUT/hops-<arm>.<text>.log,
# OUT/kld_lowbit.tsv (table.sh's columns), OUT/gate.tsv (arm, text, change, se, verdict), OUT/DONE.
set -uo pipefail
: "${BIN:?}" "${BASES:?}" "${MODEL:?}" "${OUT:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}" "${TAG:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
Q=$HERE/../../2026-11-03_flash_next_quality/run
KLD_SH=${KLD_SH:-$Q/kld.sh}
TABLE_SH=${TABLE_SH:-$HERE/../../2026-11-04_moe_tail_kld/run/table.sh}
ARMS=${ARMS:-off q6_K q5_1 q5_0 q4_1 iq4_nl q4_0}
TYPES=${TYPES:-f16 bf16 q8_0 q6_K q5_1 q5_0 q4_1 q4_0 iq4_nl}
AT=${AT:-6,11,16,22,27,32,38,43}
STEPS=${STEPS:-gpucheck boundaries kld}
export MOUNTS
. "$Q/common.sh"
step() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$OUT"
rm -f "$OUT/DONE"
failed=0
log "kld_lowbit: steps $STEPS, GPUs $GPUS, arms $ARMS, boundaries $AT"
preflight_gpus "$GPUS" || { log "CUDA does not work on $GPUS: stopped"; exit 1; }

gpu_run() { # <name> <script>: a container on GPUS with the build, the model's folders and OUT
    docker run --rm $DOCKER_BASE --name "pq-$TAG-$1" --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -v "$BIN:$BIN:ro" -v "$OUT:$OUT" $MOUNT_ARGS "$IMAGE" bash -c "$2"
}

if step gpucheck && ! grep -q "checks, 0 failed" "$OUT/gpucheck/unit.log" 2> /dev/null; then
    mkdir -p "$OUT/gpucheck"
    log "gpucheck: start"
    gpu_run check "nice -n 19 $BIN/test-backend-ops -b CUDA0 -o CPY > $OUT/gpucheck/ops.log 2>&1; echo \"ops rc=\$?\";
        nice -n 19 $BIN/test-hop-cast --unit-only > $OUT/gpucheck/unit.log 2>&1; echo \"unit rc=\$?\"" \
        > "$OUT/gpucheck/rc.txt" 2>&1
    log "gpucheck: $(tr '\n' ' ' < "$OUT/gpucheck/rc.txt") $(grep -E 'tests passed|backends passed' "$OUT/gpucheck/ops.log" | tr -s ' ' | tr '\n' ' ') $(tail -n 1 "$OUT/gpucheck/unit.log")"
    grep -q "ops rc=0" "$OUT/gpucheck/rc.txt" && grep -q "unit rc=0" "$OUT/gpucheck/rc.txt" || failed=1
fi

if step boundaries && ! grep -q "checks, " "$OUT/boundaries/hops.log" 2> /dev/null; then
    mkdir -p "$OUT/boundaries"
    log "boundaries: start"
    arms="--arm off"
    for t in $TYPES; do arms="$arms --arm '$t LLAMA_HOP_CAST=$t LLAMA_HOP_CAST_AT=$AT'"; done
    gpu_run hops "nice -n 19 $BIN/test-hop-cast -m $MODEL -ngl 99 --prompt 192 --ubatch 64 --tsv $OUT/boundaries/hops.tsv \
        $arms > $OUT/boundaries/hops.log 2>&1; echo \"rc=\$?\"" > "$OUT/boundaries/rc.txt" 2>&1
    log "boundaries: $(cat "$OUT/boundaries/rc.txt") $(tail -n 1 "$OUT/boundaries/hops.log")"
fi

if step kld; then
    # kld.sh reads the bases and the split from its OUT: link them in, the bases unchanged
    for t in ${TEXTS:-wiki agent code}; do
        [ -e "$OUT/base.$t.kld" ] || ln -s "$BASES/base.$t.kld" "$OUT/base.$t.kld"
    done
    cp "$BASES/gpu_args" "$OUT/gpu_args"
    PPL=$BIN/llama-perplexity
    for a in $ARMS; do
        # each arm a wrapper next to llama-perplexity that sets its variables, so kld.sh needs no change
        w=$BIN/ppl-hop-$a
        if [ "$a" = off ]; then e=""; else e="export LLAMA_HOP_CAST=$a LLAMA_HOP_CAST_AT=$AT"; fi
        printf '#!/usr/bin/env bash\n# kld_lowbit.sh arm %s\n%s\nexec "$(dirname "$0")/llama-perplexity" "$@"\n' "$a" "$e" > "$w"
        chmod +x "$w"
        PPL=$w OUT=$OUT GPUS=$GPUS TAG=$TAG MOUNTS=$MOUNTS WIKI=$WIKI AGENT=$AGENT CODE=$CODE TEXTS=${TEXTS:-} \
            bash "$KLD_SH" kld "$MODEL" "hops-$a" || failed=1
        # the boundaries the run cast at (none for off), from llama-perplexity's log
        for t in ${TEXTS:-wiki agent code}; do
            grep -m 1 'card hops in' "$OUT/hops-$a.$t.log" 2> /dev/null | sed "s/^/hops-$a $t: /"
        done
    done
    OUT=$OUT OFF=hops-off CTRL=none bash "$TABLE_SH" > "$OUT/kld_lowbit.tsv"
    # the gate: the paired change over off within noise (d - 2 se <= 0)
    awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; print "arm\ttext\td_kld_vs_off\tse\tverdict"; next }
        $(c["arm"]) != "hops-off" && $(c["d_kld_vs_off"]) != "-" {
            d = $(c["d_kld_vs_off"]); s = $(c["d_vs_off_se"])
            printf "%s\t%s\t%s\t%s\t%s\n", $(c["arm"]), $(c["text"]), d, s, (d - 2 * s <= 0 ? "within noise" : "ABOVE NOISE") }' \
        "$OUT/kld_lowbit.tsv" > "$OUT/gate.tsv"
    cat "$OUT/gate.tsv"
fi
date -Is > "$OUT/DONE"
log "kld_lowbit: done, failed=$failed"
exit $failed
