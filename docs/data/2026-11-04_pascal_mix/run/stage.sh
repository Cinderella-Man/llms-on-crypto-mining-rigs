#!/usr/bin/env bash
# Stage a built mix on the fast disk before its KLD run, so the cards don't wait on the copy: shards 1..N-1 copied to
# HOT/<name>/ and checked against the build's .md5, the last shard (the n-gram table) linked to TABLE_FAST, a copy of
# the same table on the fast disk (its size and first 64 KiB must match). score.sh then finds every shard in place
# when the file's first shard is given as HOT/<name>/<first shard>, and removes the folder after the file's texts.
# Low priority (nice 19, idle I/O).
#   HOT=<fast scratch> TABLE_FAST=<table shard> bash stage.sh <name> <first shard of the built file>
set -euo pipefail
: "${HOT:?}" "${TABLE_FAST:?}"
name=$1 first=$2
src=$(dirname "$first")
d=$HOT/$name
mkdir -p "$d"
mapfile -t S < <(ls "${first%-00001-of-*}"-0000?-of-0000?.gguf)
last=${S[${#S[@]} - 1]}
for s in "${S[@]}"; do
    dst=$d/$(basename "$s")
    if [ "$s" = "$last" ]; then
        [ "$(stat -Lc %s "$s")" -eq "$(stat -Lc %s "$TABLE_FAST")" ] &&
            [ "$(head -c 65536 "$s" | md5sum)" = "$(head -c 65536 "$TABLE_FAST" | md5sum)" ] ||
            { echo "$s: not the table of $TABLE_FAST" >&2; exit 1; }
        ln -sfn "$TABLE_FAST" "$dst"
        continue
    fi
    [ -f "$dst" ] && [ "$(stat -c %s "$dst")" -eq "$(stat -Lc %s "$s")" ] || nice -n 19 ionice -c3 cp -L "$s" "$dst"
done
(cd "$d" && nice -n 19 ionice -c3 md5sum -c "$src"/*.md5)
echo "staged $name in $d"
