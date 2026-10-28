#!/usr/bin/env bash
# Every Ornith-1.5-35B candidate in turn: run/quantize.sh makes the next one on the CPU while the previous one is
# scored on the GPUs against the Q8_0's three bases with ../../2026-10-27_ornith_kld/run/kld_texts.sh (wikitext-2
# 80 x 512, held-out agent sessions 12 x 8,192, Exercism code 80 x 512), so at most two candidates sit on disk at a
# time. A scored candidate is deleted unless its recipe is in KEEP; its scores stay in KLD/<file>.<text>.kld.log.
#   QUANT=<llama-quantize, CPU build> SRC=<BF16 .gguf> OUT=<folder for candidates> REF=<reference Q8_0 .gguf>
#   KLD=<folder holding the reference's bases <ref>.<text>.kld> BIN=<llama-perplexity, CUDA build> GPUS=<ids>
#   WIKI=<wiki.test.raw> AGENT=<agent text> CODE=<code text> RECIPES="P-Q5 Q6_K ..." [KEEP="P-Q5 ..."] \
#   [IMATRIX=<imatrix.gguf>, for -i recipes] bash candidates.sh
set -euo pipefail
: "${QUANT:?}" "${SRC:?}" "${OUT:?}" "${REF:?}" "${KLD:?}" "${BIN:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}"
: "${RECIPES:?}"
LABEL=${LABEL:-pascal-quants=1}
here=$(cd "$(dirname "$0")" && pwd)
KLDSH=${KLDSH:-$here/../../2026-10-27_ornith_kld/run/kld_texts.sh}
score() {
    echo "== $(date +%T) score $1"
    BIN=$BIN OUT=$KLD GPUS=$GPUS WIKI=$WIKI AGENT=$AGENT CODE=$CODE DOCKER_ARGS="--label $LABEL --cpu-shares 256" \
        bash "$KLDSH" "$REF" "$OUT/Ornith-1.5-35B-$1.gguf" || true
    for t in wiki agent code; do
        grep -q "Mean    KLD" "$KLD/Ornith-1.5-35B-$1.$t.kld.log" 2> /dev/null || { echo "kept $1: no $t score"; return; }
    done
    case " ${KEEP:-} " in *" $1 "*) ;; *) rm -f "$OUT/Ornith-1.5-35B-$1.gguf" ;; esac
}
prev=""
for r in $RECIPES; do
    echo "== $(date +%T) quantize $r"
    QUANT=$QUANT SRC=$SRC OUT=$OUT RECIPE=$r LABEL=$LABEL bash "$here/quantize.sh" &
    q=$!
    [ -n "$prev" ] && score "$prev"
    wait "$q" || { echo "quantize $r failed (see $OUT/quant-$r.log)"; prev=""; continue; }
    prev=$r
done
[ -n "$prev" ] && score "$prev"
echo "== $(date +%T) done"
