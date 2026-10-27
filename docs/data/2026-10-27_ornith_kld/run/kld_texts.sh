#!/usr/bin/env bash
# KL-divergence bases and scores of GGUF files on three texts, on CUDA GPUs in Docker (llama-perplexity):
#   wiki   the wikitext-2 test file, 80 chunks of 512 tokens (this repository's "within noise" test)
#   agent  12 chunks of 8,192 tokens of agent sessions (agent_text.py)
#   code   80 chunks of 512 tokens of the agent check's Exercism code (code_text.py)
# First the reference's base for each text (kept and reused when it exists), then each candidate against it.
#   BIN=<llama-perplexity> OUT=<folder> GPUS=<0,1,...> WIKI=<wiki.test.raw> AGENT=<agent text> CODE=<code text> \
#     [TS=<-ts split>] [TEXTS="wiki agent code"] bash kld_texts.sh <reference.gguf> [<candidate.gguf>[=<BIN>] ...]
# A candidate can name its own llama-perplexity (a build that knows its types). Every path is mounted at the same place
# in the container. Ornith-1.5-122B Q8_0 on seven cards (10, 12, 24, 32, 32, 24, 24 GB) needs TS=2.5,3,7,11,11,8,7.5:
# llama.cpp maps -ts onto 50 slots (49 layer blocks, the unused MTP block included, plus the output), so running totals
# at half layers give 3/3/7/11/11/8 layers and the rest with the output on the last card.
set -euo pipefail
: "${BIN:?}" "${OUT:?}" "${GPUS:?}"
REF=$1; shift
TEXTS=${TEXTS:-wiki agent code}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
run() {   # model, binary, text file, context, chunks, base file, mode (base|kld), log
    local extra="--kl-divergence-base $6" mounts=() d
    [ "$7" = kld ] && extra="$extra --kl-divergence"
    for d in "$(dirname "$1")" "$(dirname "$(dirname "$2")")" "$(dirname "$3")" "$OUT"; do mounts+=(-v "$d:$d"); done
    docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${mounts[@]}" "$IMAGE" \
        "$2" -m "$1" -f "$3" -c "$4" -b "$4" -ub 512 --chunks "$5" -ngl 99 ${TS:+-ts "$TS"} -fa on $extra > "$8" 2>&1
}
for t in $TEXTS; do
    case $t in
        wiki)  txt=${WIKI:?}; c=512; n=80 ;;
        agent) txt=${AGENT:?}; c=8192; n=12 ;;
        code)  txt=${CODE:?}; c=512; n=80 ;;
        *) echo "unknown text $t"; continue ;;
    esac
    r=$(basename "$REF" .gguf)
    base=$OUT/$r.$t.kld
    if [ ! -f "$base" ]; then
        run "$REF" "$BIN" "$txt" "$c" "$n" "$base" base "$OUT/$r.$t.base.log"
        grep "Final estimate" "$OUT/$r.$t.base.log"
    fi
    for spec in "$@"; do
        m=${spec%%=*}; b=$BIN
        [ "$spec" != "$m" ] && b=${spec#*=}
        log=$OUT/$(basename "$m" .gguf).$t.kld.log
        run "$m" "$b" "$txt" "$c" "$n" "$base" kld "$log" || { echo "FAILED: $log"; continue; }
        grep -E "^(Mean    KLD|Same top p)" "$log"
    done
done
