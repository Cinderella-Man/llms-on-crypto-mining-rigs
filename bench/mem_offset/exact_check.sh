#!/usr/bin/env bash
# exact_check.sh: is the model's output on the rig bit-identical to stock? Part of the memory-offset kit
# (bench/mem_offset/README.md, "The exact check").
#
#   exact_check.sh base         at offset 0: write the KLD base file with one-token steps, then rerun against it; the
#                               rerun's statistics block is the reference (identical logits give identical statistics)
#   exact_check.sh kld TAG      a one-token run against the base file; PASS when its whole statistics block equals the
#                               stock rerun's, digit for digit, and the same top token is 100 %
#   exact_check.sh greedy TAG   greedy completion of a fixed prompt, hashed; TAG "stock" is the reference, any other
#                               TAG passes when its hash equals it
#
# Why not "KLD = 0": llama-perplexity stores the base log-probabilities with 16 bits, so even bit-identical logits show
# a small KLD (Mean prints 0.000000, the maximum ~1e-5). A changed logit anywhere changes the block, so the comparison
# with a stock rerun is the exact test.
#
# Env (defaults in brackets): B llama.cpp bin dir [the v2 base build], M model [UD-Q4_K_XL], WIKI [wiki.test.raw],
#   DEVS cards [0,1,2,3,4,5; must include the card under test], E extra env [LLAMA_NO_MOE_DOWN_RESCALE=tg],
#   OUT results dir [out/ next to this script], CTX [512], CHUNKS [8] (CTX x CHUNKS one-token steps), N_GREEDY [128]
# Exit status: 0 PASS (or base written), 1 FAIL or a run failed, 2 usage.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
B=${B:-/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin}
M=${M:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
WIKI=${WIKI:-/opt/llama-builds/data/wiki.test.raw}
DEVS=${DEVS:-0,1,2,3,4,5}
E=${E:-LLAMA_NO_MOE_DOWN_RESCALE=tg}
OUT=${OUT:-$here/out}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-8}
N_GREEDY=${N_GREEDY:-128}
PROMPT="The memory controller of a graphics card checks every burst it reads. Explain, step by step, how a CRC on the data bus detects a transmission error and what the controller does next:"
mkdir -p "$OUT"
run() { env $E CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVS" "$@"; }
log() { echo "$(date '+%F %T') exact_check: $*" | tee -a "$OUT/exact_check.log"; }
ppl() { run "$B/llama-perplexity" -m "$M" -ngl 99 -fa on -f "$WIKI" -c "$CTX" --chunks "$CHUNKS" -b "$CTX" -ub 1 "$@"; }
# the statistics block of a KLD run, without timestamps and timings
stats() { sed -E 's/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [A-Z] //' "$1" | grep -E '^(Mean|Median|Maximum|Minimum|Cor|RMS|Same top|[0-9]+(\.[0-9]+)?%)'; }

kld_run() {  # kld_run TAG: one run against the base file, statistics in $OUT/kld-TAG.stats
    local tag=$1
    [ -s "$OUT/base.kld" ] || { log "no base file: run 'exact_check.sh base' at offset 0 first"; return 1; }
    log "kld $tag: $CHUNKS x $CTX one-token steps on cards $DEVS"
    if ! ppl --kl-divergence-base "$OUT/base.kld" --kl-divergence > "$OUT/kld-$tag.log" 2>&1; then
        log "kld $tag: llama-perplexity failed, see $OUT/kld-$tag.log"; return 1
    fi
    stats "$OUT/kld-$tag.log" > "$OUT/kld-$tag.stats"
    [ -s "$OUT/kld-$tag.stats" ] || { log "kld $tag: no statistics in the log"; return 1; }
    grep -E '^(Mean PPL\(Q\)|Mean +KLD|Maximum +KLD|Same top p)' "$OUT/kld-$tag.stats" | sed 's/^/    /' | tee -a "$OUT/exact_check.log"
}

case "${1:-}" in
base)
    log "base: writing $OUT/base.kld at offset 0 ($CHUNKS x $CTX one-token steps on cards $DEVS)"
    if ! ppl --kl-divergence-base "$OUT/base.kld" > "$OUT/base.log" 2>&1; then
        log "base: llama-perplexity failed, see $OUT/base.log"; exit 1
    fi
    grep -E 'Final estimate' "$OUT/base.log" | sed -E 's/^[0-9.]+ [A-Z] //; s/^/    /' | tee -a "$OUT/exact_check.log"
    kld_run stock || exit 1
    grep -q '^Same top p *: *100\.000' "$OUT/kld-stock.stats" \
        || { log "base: the stock rerun does not give the same top token everywhere: the run is not deterministic, stop"; exit 1; }
    log "base: reference statistics in $OUT/kld-stock.stats"
    ;;
kld)
    tag=${2:?tag}
    kld_run "$tag" || exit 1
    if cmp -s "$OUT/kld-stock.stats" "$OUT/kld-$tag.stats" && grep -q '^Same top p *: *100\.000' "$OUT/kld-$tag.stats"; then
        log "kld $tag: PASS (statistics identical to the stock rerun)"
    else
        log "kld $tag: FAIL (statistics differ from the stock rerun):"
        diff "$OUT/kld-stock.stats" "$OUT/kld-$tag.stats" | head -20 | tee -a "$OUT/exact_check.log"
        exit 1
    fi
    ;;
greedy)
    tag=${2:?tag}
    log "greedy $tag: $N_GREEDY tokens at temperature 0 on cards $DEVS"
    if ! run "$B/llama-completion" -m "$M" -ngl 99 -fa on -c 1024 -n "$N_GREEDY" --temp 0 -s 1 -no-cnv --no-display-prompt \
            --no-warmup -p "$PROMPT" > "$OUT/greedy-$tag.txt" 2> "$OUT/greedy-$tag.log"; then
        log "greedy $tag: llama-completion failed, see $OUT/greedy-$tag.log"; exit 1
    fi
    h=$(sha256sum < "$OUT/greedy-$tag.txt" | cut -c1-16)
    echo "$h" > "$OUT/greedy-$tag.sha"
    if [ "$tag" = stock ]; then
        log "greedy stock: hash $h ($(wc -c < "$OUT/greedy-stock.txt") bytes), the reference"
    elif [ "$h" = "$(cat "$OUT/greedy-stock.sha" 2>/dev/null)" ]; then
        log "greedy $tag: PASS (hash $h, as stock)"
    else
        log "greedy $tag: FAIL (hash $h, stock $(cat "$OUT/greedy-stock.sha" 2>/dev/null || echo missing))"
        exit 1
    fi
    ;;
*)
    sed -n '2,24p' "$0"; exit 2
    ;;
esac
