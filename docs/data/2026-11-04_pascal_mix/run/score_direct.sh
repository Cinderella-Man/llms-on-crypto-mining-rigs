#!/usr/bin/env bash
# The KL divergence of built files read where they lie, with nothing copied: ../../2026-11-04_gguf_compose/run/score.sh
# without its staging, for a fast disk that must not take writes. Each file gets a folder of symbolic links in LINKS:
# its shards 1..N-1 to the built shards, the last (the n-gram table, read at random) to TABLE_FAST, a copy of the same
# table on the fast disk (its size and first 64 KiB must match); then FN2's fn2.sh scores it from there with the
# bases' split (FAST=LINKS, so fn2.sh copies nothing). The containers must mount LINKS and every target (MOUNTS in the
# environment file). The same interface as score.sh, so run/score_all.sh can drive it (SCORE=<this file>).
#   ENV=<environment file> QUALITY=<2026-11-03_flash_next_quality's run/> BASES=<its kld folder> OUT=<score folder>
#   HOT=<scratch> TABLE_FAST=<table shard> LINKS=<folder for the links> [SCORE_TAG=pmix] FILES="<name>=<first shard>
#   ..." bash score_direct.sh
set -euo pipefail
: "${ENV:?}" "${QUALITY:?}" "${BASES:?}" "${OUT:?}" "${HOT:?}" "${TABLE_FAST:?}" "${LINKS:?}" "${FILES:?}"
mine=("$OUT" "$HOT" "$LINKS")
. "$ENV"
OUT=${mine[0]} HOT=${mine[1]} LINKS=${mine[2]}
export TAG=${SCORE_TAG:-pmix}
. "$QUALITY/common.sh"
mkdir -p "$OUT" "$HOT" "$LINKS"
for f in "$BASES"/base.{wiki,agent,code}.kld "$BASES"/gpu_args "$BASES"/q8_0-ub256.*.log; do
    [ -e "$f" ] && [ ! -e "$OUT/$(basename "$f")" ] && ln -sfn "$f" "$OUT/"
done
for t in wiki agent code; do [ -f "$OUT/base.$t.kld" ] || { log "no base logits $BASES/base.$t.kld"; exit 1; }; done
head_fast=$(head -c 65536 "$TABLE_FAST" | md5sum)
rc=0
while IFS='=' read -r name first; do
    [ -z "$name" ] && continue
    d=$LINKS/$name
    mkdir -p "$d"
    mapfile -t S < <(ls "${first%-00001-of-*}"-0000?-of-0000?.gguf)
    last=${S[${#S[@]} - 1]}
    for s in "${S[@]}"; do
        if [ "$s" = "$last" ]; then
            [ "$(stat -Lc %s "$s")" -eq "$(stat -Lc %s "$TABLE_FAST")" ] && [ "$(head -c 65536 "$s" | md5sum)" = "$head_fast" ] ||
                { log "$s: not the table of $TABLE_FAST"; exit 1; }
            ln -sfn "$TABLE_FAST" "$d/$(basename "$s")"
        else
            ln -sfn "$(readlink -f "$s")" "$d/$(basename "$s")"
        fi
    done
    log "$name: scoring $first in place (links in $d)"
    FN2_FILES="$name=$d/$(basename "$first")" KLD=$OUT HOT=$HOT FAST=$LINKS ENV= bash "$QUALITY/fn2.sh" || rc=1
done <<< "$FILES"
exit $rc
