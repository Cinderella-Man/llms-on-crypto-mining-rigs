#!/usr/bin/env bash
# Q897's GPU pass: which routed experts Flash-Next picks on tool-call text, one importance matrix per token class.
#   text     template_text.py: FN4's calibration sessions through the chat template, tool results cut to
#            RESPONSE_MAX characters ($OUT/text.txt); FORMAT=plain takes TEXT (calib_text.py's file) as it is
#   mask     llama-imatrix --span-dump with the CPU build (the vocabulary only), then span_mask.py find: the classes
#            0 other, 1 call, 2 argument, 3 response, 4 assistant ($OUT/mask.txt, $OUT/mask_summary.tsv); the pieces
#            must rebuild the text byte for byte
#   gpu      llama-imatrix --span-mask on GPUS (sm_86 build, -fit on, -fa on, 512-token chunks, one ubatch a batch):
#            $OUT/imatrix.gguf (the usual matrix) and $OUT/imatrix.span{0,1,2,3,4}.gguf
#   analyze  span_check.py sum (the classes add up to the usual matrix), span_counts.py against EXPERTS (the ranking
#            the cut files were made by), force_keep.py for every N in KEEP: $OUT/tables/
# Env: WORK, OUT, MODEL (first shard), CPU_BIN (build.sh's cpu/bin), GPU_BIN (the sm_86 tarball's bin), REPO (a checkout
#      of this repository), EXPERTS (docs/data/2026-11-04_expert_usage/experts.tsv.gz), GPUS (e.g. 0,1,2), TAG (the
#      container's name, pq-<TAG>-imatrix), FORMAT (template), SESSIONS and TEST_ROWS (for text), TEXT (for plain),
#      RESPONSE_MAX (1500), CHUNKS (200), KEEP (464,448,416), MAX_PER_LAYER (8), STEPS (default "text mask gpu
#      analyze"), IMAGE, MOUNTS (folders the model and text sit in, mounted at their own paths), PREFLIGHT (the CUDA
#      pre-flight script, scripts/cuda_preflight.sh), PREFLIGHT_CACHE, EXTRA (more llama-imatrix arguments)
# Prints "gpu_pass: done" at the end, "gpu_pass: FAILED <step>" on a failure.
set -uo pipefail
: "${WORK:?}" "${OUT:?}" "${MODEL:?}" "${CPU_BIN:?}" "${REPO:?}" "${EXPERTS:?}"
FORMAT=${FORMAT:-template}
RESPONSE_MAX=${RESPONSE_MAX:-1500}
CHUNKS=${CHUNKS:-200}
KEEP=${KEEP:-464,448,416}
TAG=${TAG:-spancount}
STEPS=${STEPS:-text mask gpu analyze}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
R=$REPO/docs/data/2026-11-04_toolcall_experts/run
D="--rm --cpu-shares 256 --label pascal-quants=1"
mounts="-v $WORK:$WORK -v $OUT:$OUT -v $CPU_BIN:$CPU_BIN"
[ -n "${GPU_BIN:-}" ] && mounts="$mounts -v $GPU_BIN:$GPU_BIN"
for d in ${MOUNTS:-}; do mounts="$mounts -v $d:$d"; done
mkdir -p "$OUT/tables/call_only" "$OUT/tables/written"

log()  { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
py()   { nice -n 19 python3 -I "$@"; }
fail() { log "gpu_pass: FAILED $1"; exit 1; }
special=""; [ "$FORMAT" = template ] && special="--parse-special"
T=$OUT/text.txt

if step text; then
    if [ "$FORMAT" = template ]; then
        : "${SESSIONS:?}" "${TEST_ROWS:?}"
        py "$R/template_text.py" "$SESSIONS" "$TEST_ROWS" "$T" 1500000 12000 "$RESPONSE_MAX" || fail text
    else
        : "${TEXT:?}"
        cp "$TEXT" "$T" || fail text
    fi
    log "text: $(wc -c < "$T") bytes, md5 $(md5sum < "$T" | cut -d' ' -f1)"
fi

if step mask; then
    docker run $D $mounts "$IMAGE" "$CPU_BIN/llama-imatrix" -m "$MODEL" -f "$T" $special --span-dump "$OUT/dump.tsv" \
        > "$OUT/dump.log" 2>&1 || fail "mask (dump: $OUT/dump.log)"
    py "$R/span_mask.py" find "$OUT/dump.tsv" --format "$FORMAT" --text "$T" --out "$OUT/mask.txt" \
        --summary "$OUT/mask_summary.tsv" --chunks "$CHUNKS" --show 3 > "$OUT/mask.log" 2>&1 || fail "mask ($OUT/mask.log)"
    log "mask: $(grep -E '^(tokens|calls|parameters)\s' "$OUT/mask_summary.tsv" | tr '\t\n' ' ')"
fi

if step gpu; then
    : "${GPU_BIN:?}" "${GPUS:?}"
    if [ -n "${PREFLIGHT:-}" ]; then
        out=$(IMAGE=$IMAGE PREFLIGHT_DOCKER_ARGS="$D" PREFLIGHT_CACHE=${PREFLIGHT_CACHE:-} bash "$PREFLIGHT" -q -g "$GPUS" 2>&1) \
            || { log "CUDA pre-flight: $(printf '%s\n' "$out" | tail -n 1)"; fail "gpu (pre-flight)"; }
        log "CUDA pre-flight on GPUs $GPUS: $(printf '%s\n' "$out" | tail -n 1)"
    fi
    log "gpu: llama-imatrix on GPUs $GPUS, $CHUNKS chunks"
    docker run $D --name "pq-$TAG-imatrix" --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID $mounts "$IMAGE" \
        "$GPU_BIN/llama-imatrix" -m "$MODEL" -f "$T" $special --span-mask "$OUT/mask.txt" -o "$OUT/imatrix.gguf" \
        -c 512 -b 512 -ub 512 --chunks "$CHUNKS" -fit on -fa on -ofreq 1000 ${EXTRA:-} > "$OUT/imatrix.log" 2>&1 \
        || fail "gpu ($OUT/imatrix.log)"
    grep -E "Final estimate|save_span_imatrix" "$OUT/imatrix.log" | tail -n 6
fi

if step analyze; then
    spans=$(ls "$OUT"/imatrix.span[0-9].gguf | sort -V)
    py "$R/span_check.py" sum "$OUT/imatrix.gguf" $spans | tee "$OUT/tables/check_sum.txt"
    py "$R/span_counts.py" --full "$OUT/imatrix.gguf" --span $spans --experts "$EXPERTS" --keep "$KEEP" \
        --out "$OUT/tables" || fail "analyze (span_counts)"
    for n in ${KEEP//,/ }; do
        py "$R/force_keep.py" --span-experts "$OUT/tables/span_experts.tsv.gz" --experts "$EXPERTS" --keep "$n" \
            --max-per-layer "${MAX_PER_LAYER:-8}" --out "$OUT/tables" | tee "$OUT/tables/force_$n.txt" \
            || fail "analyze (force_keep $n)"
        py "$R/force_keep.py" --span-experts "$OUT/tables/span_experts.tsv.gz" --experts "$EXPERTS" --keep "$n" \
            --classes call --max-per-layer "${MAX_PER_LAYER:-8}" --out "$OUT/tables/call_only" \
            | tee "$OUT/tables/call_only/force_$n.txt" || fail "analyze (force_keep $n, call only)"
        py "$R/force_keep.py" --span-experts "$OUT/tables/span_experts.tsv.gz" --experts "$EXPERTS" --keep "$n" \
            --classes call,argument,assistant --max-per-layer "${MAX_PER_LAYER:-8}" --out "$OUT/tables/written" \
            | tee "$OUT/tables/written/force_$n.txt" || fail "analyze (force_keep $n, all the model writes)"
    done
fi
log "gpu_pass: done"
