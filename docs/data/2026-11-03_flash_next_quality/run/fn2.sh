#!/usr/bin/env bash
# FN2: each local Flash-Next file's KL divergence against the reference's bases, one file at a time on all seven GPUs
# with the bases' split. FN2_FILES lists "<name>=<first shard>" lines. A file outside FAST (the fast disk's path
# prefix; read from the environment file ENV when unset) is first copied to $HOT/fn2/<name>: one sequential read
# instead of three model loads and the n-gram table's random reads; the copy is removed after its three texts. Files
# whose three logs are complete are skipped.
#   (env: the chain's) bash fn2.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
[ -z "${FAST:-}" ] && [ -n "${ENV:-}" ] && . "$ENV"
: "${FN2_FILES:?}" "${KLD:?}" "${HOT:?}" "${GPUS_ALL:?}" "${FAST:?}"
export GPUS=$GPUS_ALL OUT=$KLD
failed=0
while IFS='=' read -r name first; do
    [ -z "$name" ] && continue
    n=0
    for t in wiki agent code; do grep -q "^Mean    KLD" "$KLD/$name.$t.log" 2> /dev/null && n=$((n + 1)); done
    if [ $n -eq 3 ]; then log "$name: scored before"; continue; fi
    model=$first
    case $first in
        "$FAST"/*) ;;
        *)
            d=$HOT/fn2/$name
            mkdir -p "$d"
            for f in "${first%-00001-of-*}"-0000?-of-0000?.gguf; do
                [ -f "$d/$(basename "$f")" ] && [ "$(stat -c %s "$d/$(basename "$f")")" -eq "$(stat -c %s "$f")" ] && continue
                log "$name: copying $(basename "$f") ($(stat -c %s "$f") bytes)"
                nice -n 19 cp "$f" "$d/" || { log "$name: copy failed"; failed=1; continue 2; }
            done
            model=$d/$(basename "$first")
            ;;
    esac
    wait_gpus "$GPUS"
    bash "$HERE/kld.sh" kld "$model" "$name" || failed=1
    [ "$model" != "$first" ] && rm -rf "${HOT:?}/fn2/$name"
done <<< "$FN2_FILES"
exit $failed
