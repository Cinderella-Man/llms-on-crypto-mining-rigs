#!/usr/bin/env bash
# The scores: each built file's KL divergence against the Swift-1.5 Q8_0 reference's logits on the three texts, with
# the scripts, cards and tensor split of ../../2026-11-03_flash_next_quality (its run/kld.sh through its run/fn2.sh),
# so the numbers sit beside that folder's scores of the unmodified and hcq8 files. Run it once that folder's chain has
# ended (it waits for the GPUs anyway).
# That folder's kld folder (BASES) is used read-only: the base logits and the split (gpu_args) are linked into OUT,
# where the scores go, with its q8_0-ub256 (the floor) and swift-* logs for the table. One file at a time is staged on
# the fast disk: its shards copied to HOT/<name>/ and its table shard linked to TABLE_FAST, a copy of the same table
# on the fast disk (the n-gram table is read at random; 28.8 GB need not be copied again). The md5 of TABLE_FAST and
# of every file's own table must be TABLE_MD5. The staged copy is removed after the file's three texts, and a file
# already scored is skipped. Containers are named pq-$SCORE_TAG-*.
#   ENV=<that folder's environment file> QUALITY=<that folder's run/> BASES=<its kld folder> OUT=<score folder> \
#   HOT=<fast scratch> TABLE_FAST=<table shard on the fast disk> [TABLE_MD5=98ce140e641bc5d68af9f31df0f1c048] \
#   [SCORE_TAG=c74] FILES="<name>=<first shard>
#   ..." setsid nohup bash score.sh >> <log> 2>&1 < /dev/null &
# Then: python3 -I $QUALITY/table.py $OUT q8_0-ub256 swift-q2_0-hcq8 <names>.
set -euo pipefail
: "${ENV:?}" "${QUALITY:?}" "${BASES:?}" "${OUT:?}" "${HOT:?}" "${TABLE_FAST:?}" "${FILES:?}"
mine=("$OUT" "$HOT")
. "$ENV"  # it sets that chain's own HOT and KLD, which this script must not write to
OUT=${mine[0]} HOT=${mine[1]}
export TAG=${SCORE_TAG:-c74}
TABLE_MD5=${TABLE_MD5:-98ce140e641bc5d68af9f31df0f1c048}
. "$QUALITY/common.sh"
mkdir -p "$OUT" "$HOT"
for f in "$BASES"/base.{wiki,agent,code}.kld "$BASES"/gpu_args "$BASES"/q8_0-ub256.*.log "$BASES"/swift-*.log; do
    [ -e "$f" ] && ln -sfn "$f" "$OUT/"
done
for t in wiki agent code; do [ -f "$OUT/base.$t.kld" ] || { log "no base logits $BASES/base.$t.kld"; exit 1; }; done
[ -s "$OUT/gpu_args" ] || { log "no split $BASES/gpu_args"; exit 1; }
declare -A MD5
md5() {  # once per file
    local r
    r=$(readlink -f "$1")
    [ -n "${MD5[$r]:-}" ] || MD5[$r]=$(nice -n 19 md5sum < "$r" | cut -d' ' -f1)
    echo "${MD5[$r]}"
}
[ "$(md5 "$TABLE_FAST")" = "$TABLE_MD5" ] || { log "$TABLE_FAST: md5 is not $TABLE_MD5"; exit 1; }
head_fast=$(head -c 65536 "$TABLE_FAST" | md5sum)

rc=0
while IFS='=' read -r name first; do
    [ -z "$name" ] && continue
    n=0
    for t in wiki agent code; do grep -q "^Mean    KLD" "$OUT/$name.$t.log" 2> /dev/null && n=$((n + 1)); done
    if [ $n -eq 3 ]; then log "$name: scored before"; continue; fi
    d=$HOT/$name
    mkdir -p "$d"
    for s in "${first%-00001-of-*}"-0000?-of-0000?.gguf; do
        dst=$d/$(basename "$s")
        if [ "$(stat -Lc %s "$s")" -eq "$(stat -Lc %s "$TABLE_FAST")" ] && [ "$(head -c 65536 "$s" | md5sum)" = "$head_fast" ]; then
            [ "$(md5 "$s")" = "$TABLE_MD5" ] || { log "$s: md5 is not $TABLE_MD5"; exit 1; }
            ln -sfn "$TABLE_FAST" "$dst"  # the table shard
            continue
        fi
        [ -f "$dst" ] && [ ! -L "$dst" ] && [ "$(stat -c %s "$dst")" -eq "$(stat -Lc %s "$s")" ] && continue
        log "$name: copying $(basename "$s") ($(stat -Lc %s "$s") bytes)"
        nice -n 19 cp -L "$s" "$dst"
    done
    # FN2's loop for this one file: all seven cards, the bases' split; FAST covers HOT, so nothing is copied again
    FN2_FILES="$name=$d/$(basename "$first")" KLD=$OUT HOT=$HOT FAST=$HOT ENV= bash "$QUALITY/fn2.sh" || rc=1
    rm -rf "${d:?}"
done <<< "$FILES"
exit $rc
