#!/usr/bin/env bash
# The two Gyro files, each from its Hugging Face repository at a pinned revision, downloaded with resumable curl at the
# lowest CPU and I/O priority and checked against the SHA-256 Hugging Face publishes for it (its LFS oid). LIST holds
# one "<repo> <revision> <sha256> <bytes> <file>" line per file (files.list next to this script). A checked file gets
# <file>.ok and is not fetched again; a file with the wrong hash is renamed <file>.bad and the script stops.
#   (env: OUT [LIST]) bash download.sh
set -euo pipefail
: "${OUT:?}"
LIST=$(realpath "${LIST:-$(dirname "$0")/files.list}")
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
mkdir -p "$OUT"
cd "$OUT"
log "into $OUT: $(df -h --output=avail . | tail -n 1 | tr -d ' ') free"
while read -r repo rev sha size name; do
    case $repo in ''|'#'*) continue ;; esac
    if [ -f "$name.ok" ]; then log "$name: checked before"; continue; fi
    url=https://huggingface.co/$repo/resolve/$rev/$name
    for try in $(seq 1 60); do
        [ -f "$name" ] && [ "$(stat -c %s "$name")" -eq "$size" ] && break
        ionice -c3 nice -n 19 curl -fsSL -C - --retry 5 --retry-delay 10 -o "$name" "$url" && break
        log "retry $try for $name (curl exit $?)"
        sleep 30
    done
    [ "$(stat -c %s "$name")" -eq "$size" ] || { log "$name: $(stat -c %s "$name") bytes, not $size"; exit 1; }
    log "$name: $size bytes, hashing"
    got=$(ionice -c3 nice -n 19 sha256sum "$name" | cut -d' ' -f1)
    if [ "$got" != "$sha" ]; then mv "$name" "$name.bad"; log "BAD $name: sha256 $got"; exit 1; fi
    echo "$sha  $name" > "$name.ok"
    log "ok $name"
done < "$LIST"
log "all files match their sha256"
