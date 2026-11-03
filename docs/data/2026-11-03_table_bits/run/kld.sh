#!/usr/bin/env bash
# KL divergence of n-gram table variants against the unmodified file, on the three texts of
# ../../2026-10-27_ornith_kld (wiki 80 x 512, agent 12 x 8,192, code 80 x 512), with llama-perplexity in Docker on CUDA
# GPUs and the same flags as that folder's run/kld_texts.sh. The unmodified file's base logits come first (kept and
# reused when they exist), then each variant against them. A variant is a folder holding the model's shards (here the
# two unchanged shards as symlinks and the rewritten table shard); its name names the logs. A file `args` in it holds
# extra llama-perplexity arguments for its runs (the `-ub 256` reference: the unmodified shards in another summation
# order).
#   BIN=<llama-perplexity> OUT=<folder> GPUS=<0,1,...> WIKI=<wiki.test.raw> AGENT=<agent text> CODE=<code text> \
#     MOUNTS="<dir> ..." [TEXTS="wiki agent code"] [DOCKER_ARGS="--label ... --cpu-shares 256"] \
#     bash kld.sh <reference first shard> [<variant folder> ...]
# MOUNTS lists every folder the shards, their symlinks' targets, the texts and OUT live in; each is mounted at the
# same path.
set -euo pipefail
: "${BIN:?}" "${OUT:?}" "${GPUS:?}" "${MOUNTS:?}"
REF=$1; shift
TEXTS=${TEXTS:-wiki agent code}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
mounts=(-v "$(dirname "$(dirname "$BIN")"):$(dirname "$(dirname "$BIN")")")
for d in $MOUNTS; do mounts+=(-v "$d:$d"); done
run() {   # model, text file, context, chunks, base file, mode (base|kld), log, extra arguments
    local extra="--kl-divergence-base $5"
    [ "$6" = kld ] && extra="$extra --kl-divergence"
    docker run --rm ${DOCKER_ARGS:-} --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${mounts[@]}" "$IMAGE" \
        "$BIN" -m "$1" -f "$2" -c "$3" -b "$3" -ub 512 --chunks "$4" -ngl 99 -fa on $extra ${8:-} > "$7" 2>&1
}
for t in $TEXTS; do
    case $t in
        wiki)  txt=${WIKI:?}; c=512; n=80 ;;
        agent) txt=${AGENT:?}; c=8192; n=12 ;;
        code)  txt=${CODE:?}; c=512; n=80 ;;
        *) echo "unknown text $t"; continue ;;
    esac
    base=$OUT/base.$t.kld
    if [ ! -f "$base" ]; then
        run "$REF" "$txt" "$c" "$n" "$base" base "$OUT/base.$t.log"
        grep "Final estimate" "$OUT/base.$t.log"
    fi
    for v in "$@"; do
        m=$(ls "$v"/*-00001-of-*.gguf)
        log=$OUT/$(basename "$v").$t.log
        [ -f "$log" ] && grep -q "^Mean    KLD" "$log" && continue
        run "$m" "$txt" "$c" "$n" "$base" kld "$log" "$(cat "$v/args" 2> /dev/null || true)" ||
            { echo "FAILED: $log"; continue; }
        echo "$(basename "$v") $t: $(grep -E "^(Mean    KLD|Same top p)" "$log" | tr -s ' ' | tr '\n' ' ')"
    done
done
