#!/usr/bin/env bash
# Moves data folders off the rig's root SSD onto a larger disk, verified, and leaves a symlink at each old path, so
# that scripts and commands that use the old paths keep working. For each folder: rsync -aHAX to DEST plus the old
# path, then a checksum dry run (rsync -c -n --itemize-changes) that must print nothing; only then is the original
# removed and replaced by a symlink to the copy, owned like the original. A folder whose check prints anything is left
# where it is. Run as root on the rig, with nothing using the folders.
#   ssh $RIG "DEST=<disk>/from-ssd PATHS='/opt/llama-builds <home>/models' bash -s" < move_to_disk.sh
set -euo pipefail
: "${DEST:?folder on the larger disk}" "${PATHS:?folders to move}"
avail() { df -B1 --output=avail / | tail -1 | tr -d ' '; }
echo "root free before: $(avail) bytes"
for p in $PATHS; do
  if [ -L "$p" ] || [ ! -d "$p" ]; then echo "$p: not a real folder, skipped"; continue; fi
  d=$DEST$p
  mkdir -p "$(dirname "$d")"
  echo "== $p -> $d: $(du -xsb "$p" | cut -f1) bytes, $(find "$p" -xdev -type f | wc -l) files"
  rsync -aHAX "$p/" "$d/"
  check=$(rsync -aHAX -c -n --itemize-changes "$p/" "$d/")
  if [ -n "$check" ]; then
    echo "$p: the checksum dry run is not empty, left in place:"; echo "$check" | head -20; continue
  fi
  owner=$(stat -c %U:%G "$p")
  rm -rf -- "$p"
  ln -s "$d" "$p"
  chown -h "$owner" "$p"
  echo "$p: verified, removed, symlink -> $d"
done
echo "root free after: $(avail) bytes"
