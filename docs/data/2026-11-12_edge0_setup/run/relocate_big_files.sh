#!/usr/bin/env bash
# Move named files of a model folder to other disks and leave symlinks in their place, for a folder whose own disk
# cannot take a fast large write. A file that is not there yet gets a link to where it will be downloaded.
#
#   MODEL_DIR=/path/Edge0-145B-q3k bash relocate_big_files.sh experts.bin=/mnt/a/dir ple-fp8.gguf=/mnt/b/dir
set -eu
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
for pair in "$@"; do
  f=${pair%%=*}; d=${pair#*=}
  mkdir -p "$d"
  if [ -L "$MODEL_DIR/$f" ]; then echo "$(date '+%F %T') $f is already a link"; continue; fi
  if [ -f "$MODEL_DIR/$f" ]; then
    nice -n 19 ionice -c3 cp "$MODEL_DIR/$f" "$d/$f.moving"
    [ "$(stat -c %s "$d/$f.moving")" = "$(stat -c %s "$MODEL_DIR/$f")" ]
    mv "$d/$f.moving" "$d/$f"
    rm "$MODEL_DIR/$f"
  fi
  ln -s "$d/$f" "$MODEL_DIR/$f"
  echo "$(date '+%F %T') $f -> $d/$f ($(stat -c %s "$d/$f" 2>/dev/null || echo 0) bytes so far)"
done
