#!/usr/bin/env bash
# The engine check without a GPU: one 512-token chunk of the code text's reference logits against each file, on the
# CPU (the Vulkan build finds no device in a container without GPUs). Shows the fork reads the reference logits and
# loads the file; the score of one chunk is not a result.
#   (env: BIN MODELS BASES OUT [FILES] [MOUNTS]) bash cpucheck.sh
set -uo pipefail
: "${BIN:?}" "${MODELS:?}" "${BASES:?}" "${OUT:?}"
mkdir -p "$OUT"
mounts="-v $BIN:$BIN -v $BASES:$BASES:ro -v $OUT:$OUT -v $MODELS:$MODELS"
for d in ${MOUNTS:-}; do mounts="$mounts -v $d:$d:ro"; done
for f in ${FILES:-Coder M}; do
    timeout 1800 docker run --rm --label pascal-quants=1 --cpu-shares 256 --name pq-gyro-cpucheck $mounts \
        pq-gyro-vulkan:24.04 "$BIN/vulkan/bin/llama-perplexity" -m "$MODELS/Qwen3.8-Flash-Next-Gyro-$f-TQ2_0.gguf" \
        -c 512 -b 512 -ub 512 --chunks 1 -ngl 0 -t 8 ${EXTRA:-} \
        --kl-divergence-base "$BASES/base.code.kld" --kl-divergence > "$OUT/$f.code1.log" 2>&1
    echo "$(date '+%F %T') $f exit $?" >> "$OUT/done.txt"
done
