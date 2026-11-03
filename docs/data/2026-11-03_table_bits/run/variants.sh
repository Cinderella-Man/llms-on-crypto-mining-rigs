#!/usr/bin/env bash
# The table variants: for each format, the table shard round-tripped by table_bits into $WORK/variants/<format>/
# (per-head errors in roundtrip.tsv, fit log in roundtrip.log), with the model's other shards linked beside it.
#   TB=<table_bits> INFO=<table_info.txt> MODEL=<first shard of the unmodified model> WORK=<folder> MOUNTS="<dirs>" \
#     [THREADS=16] [RT_DOCKER_ARGS="--label ... --cpu-shares 256"] bash variants.sh FORMAT...
# RT_DOCKER_ARGS are the round trips' docker arguments (DOCKER_ARGS stays kld.sh's when all.sh runs both).
set -euo pipefail
: "${TB:?}" "${INFO:?}" "${MODEL:?}" "${WORK:?}" "${MOUNTS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
src=$(dirname "$MODEL")
name=$(basename "$MODEL" | sed 's/-00001-of-[0-9]*\.gguf$//')
table=$(awk '$1 == "file" { print $2 }' "$INFO")
for f in "$@"; do
    d=$WORK/variants/$f
    [ -f "$d/done" ] && continue
    mkdir -p "$d"
    for s in "$src/$name"-*.gguf; do [ "$s" = "$table" ] || ln -sf "$s" "$d/"; done
    BIN=$TB DOCKER_ARGS=${RT_DOCKER_ARGS:-} bash "$HERE/tb.sh" roundtrip "$INFO" "$f" "$d/$(basename "$table")" > "$d/roundtrip.tsv" 2> "$d/roundtrip.log"
    touch "$d/done"
    echo "$f: $(tail -1 "$d/roundtrip.tsv")"
done
