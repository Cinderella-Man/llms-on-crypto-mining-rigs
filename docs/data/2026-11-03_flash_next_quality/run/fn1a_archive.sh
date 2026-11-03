#!/usr/bin/env bash
# FN1a's archive copy: the checked Q8_0 from the fast disk the GPU steps read (Q8_HOT) to the bulk disk (Q8_ARCHIVE),
# each copy checked against the published SHA-256 again. Runs beside the GPU steps, at idle-class I/O.
#   (env: Q8_HOT Q8_ARCHIVE Q8_LIST) bash fn1a_archive.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"
: "${Q8_HOT:?}" "${Q8_ARCHIVE:?}" "${Q8_LIST:?}"
mkdir -p "$Q8_ARCHIVE"
while read -r sha size name; do
    [ -z "$sha" ] && continue
    if [ -f "$Q8_ARCHIVE/$name.ok" ]; then log "$name: archived before"; continue; fi
    nice -n 19 ionice -c 3 cp "$Q8_HOT/$name" "$Q8_ARCHIVE/$name.partial"
    got=$(nice -n 19 ionice -c 3 sha256sum "$Q8_ARCHIVE/$name.partial" | cut -d' ' -f1)
    [ "$got" = "$sha" ] || { log "BAD copy of $name: $got"; exit 1; }
    mv "$Q8_ARCHIVE/$name.partial" "$Q8_ARCHIVE/$name"
    echo "$sha  $name" > "$Q8_ARCHIVE/$name.ok"
    log "archived $name"
done < "$Q8_LIST"
log "archive complete: $(du -sh "$Q8_ARCHIVE" | cut -f1) in $Q8_ARCHIVE"
