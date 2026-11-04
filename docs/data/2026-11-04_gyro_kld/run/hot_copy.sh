#!/usr/bin/env bash
# Copies a checked file to a fast disk for its KL-divergence runs (the n-gram table is read at random) and checks the
# copy against its SHA-256: the one given, or the one in <file>.ok next to the original. The copy gets its own .ok.
#   (env: HOT) bash hot_copy.sh <file> [sha256]
set -euo pipefail
: "${HOT:?}"
src=$(readlink -f "$1") name=$(basename "$1")
want=${2:-$(cut -d' ' -f1 "$1.ok")}
mkdir -p "$HOT"
[ -f "$HOT/$name.ok" ] && { echo "$name: copied and checked before"; exit 0; }
echo "$(date '+%F %T') copy $name ($(df -h --output=avail "$HOT" | tail -n 1 | tr -d ' ') free)"
ionice -c3 nice -n 19 cp "$src" "$HOT/$name.part"
got=$(ionice -c3 nice -n 19 sha256sum "$HOT/$name.part" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "BAD copy $name: $got"; exit 1; }
mv "$HOT/$name.part" "$HOT/$name" && echo "$want  $name" > "$HOT/$name.ok"
echo "$(date '+%F %T') ok $name"
