#!/usr/bin/env bash
# C1038 (Q1771): records the P106 check's two fork-kernel sets, v4kern and ro8kt, again with a newer build's own
# test-backend-ops, as R153 recorded them (docs/data/2026-11-13_c831_upstream, "On a P106"): v4kern first, then ro8kt
# linking v4kern's cases (ro8kt's cases are all in v4kern). scripts/record_checks.sh does the work, in a container
# without GPUs at --cpu-shares=256, test-backend-ops under nice -n 19. Then one md5 per file of both sets (files.md5),
# for the copy to the rigs.
# Env: OUT (a new folder; it receives v4kern/ and ro8kt/), BIN (bin/ of the unpacked rig tarball), REPO (a copy of this
#   repository), JOBS (test-backend-ops -j, default 8)
# Output: OUT/v4kern/, OUT/ro8kt/ (the sets: cases/, manifest.tsv, runs.tsv, record.txt), OUT/files.md5,
#   OUT/record_sets.log, OUT/rec-<set>.out and OUT/<set>.log (record_checks.sh's console and log), OUT/record.done
# Usage: OUT=<dir> BIN=<tarball>/bin REPO=<repository> nice -n 19 bash record_sets.sh
set -uo pipefail
OUT=${OUT:?OUT=<new folder for the sets>}
BIN=${BIN:?BIN=<bin/ of the rig tarball>}
REPO=${REPO:?REPO=<a copy of this repository>}
JOBS=${JOBS:-8}
for s in v4kern ro8kt; do
    [ -e "$OUT/$s" ] && { echo "$OUT/$s exists: the sets go into a new folder" >&2; exit 1; }
done
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/record.done"
log() { echo "$(date -Is) $*" | tee -a "$OUT/record_sets.log"; }
log "test-backend-ops md5 $(md5sum < "$BIN/test-backend-ops" | cut -d' ' -f1)"
log "v4kern: record"
OUT=$OUT SUITE=v4kern BIN=$BIN STEPS=record JOBS=$JOBS bash "$REPO/scripts/record_checks.sh" > "$OUT/rec-v4kern.out" 2>&1
log "v4kern: exit $?, $(wc -l < "$OUT/v4kern/manifest.tsv") cases"
log "ro8kt: record, linking v4kern's cases"
OUT=$OUT SUITE=ro8kt BIN=$BIN STEPS=record JOBS=$JOBS LINK_FROM=$OUT/v4kern bash "$REPO/scripts/record_checks.sh" \
    > "$OUT/rec-ro8kt.out" 2>&1
log "ro8kt: exit $?, $(wc -l < "$OUT/ro8kt/manifest.tsv") cases"
(cd "$OUT" && find v4kern ro8kt -type f | LC_ALL=C sort | xargs -d '\n' md5sum) > "$OUT/files.md5"
log "files.md5: $(wc -l < "$OUT/files.md5") files, $(du -sb "$OUT/v4kern" "$OUT/ro8kt" | awk '{ s += $1 } END { print s }') bytes (hard links counted once), md5 of the list $(md5sum < "$OUT/files.md5" | cut -d' ' -f1)"
date -Is > "$OUT/record.done"
