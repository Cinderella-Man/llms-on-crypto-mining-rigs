#!/usr/bin/env bash
# FN1a: the reference file, ukisai's Swift-1.5 Q8_0 GGUF at a pinned revision, downloaded with resumable curl and each
# file checked against its SHA-256 (Hugging Face's LFS oid, listed in Q8_LIST as "<sha256> <bytes> <name>" lines). A
# checked file gets <name>.ok and is not fetched again; a file with the wrong hash is renamed <name>.bad and the step
# fails.
#   (env: Q8_REPO Q8_REV [Q8_SUBDIR] Q8_HOT Q8_LIST) bash fn1a_download.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"
: "${Q8_REPO:?}" "${Q8_REV:?}" "${Q8_HOT:?}" "${Q8_LIST:?}"
mkdir -p "$Q8_HOT"
cd "$Q8_HOT"
log "into $Q8_HOT: $(df -h --output=avail . | tail -n 1 | tr -d ' ') free"
while read -r sha size name; do
    [ -z "$sha" ] && continue
    if [ -f "$name.ok" ]; then log "$name: checked before"; continue; fi
    url=https://huggingface.co/$Q8_REPO/resolve/$Q8_REV/${Q8_SUBDIR:+$Q8_SUBDIR/}$name
    for try in $(seq 1 30); do
        [ -f "$name" ] && [ "$(stat -c %s "$name")" -eq "$size" ] && break
        nice -n 10 curl -fsSL -C - --retry 5 --retry-delay 10 -o "$name" "$url" && break
        log "retry $try for $name (curl exit $?)"
        sleep 30
    done
    [ "$(stat -c %s "$name")" -eq "$size" ] || { log "$name: $(stat -c %s "$name") bytes, not $size"; exit 1; }
    got=$(nice -n 10 sha256sum "$name" | cut -d' ' -f1)
    if [ "$got" != "$sha" ]; then mv "$name" "$name.bad"; log "BAD $name: sha256 $got"; exit 1; fi
    echo "$sha  $name" > "$name.ok"
    log "ok $name ($size bytes)"
done < "$Q8_LIST"
log "all files match their sha256"
