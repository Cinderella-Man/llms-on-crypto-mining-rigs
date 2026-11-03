#!/usr/bin/env bash
# Puts a GGUF table shard (scripts/gguf_reshard.py's isolated shard) on the internal SSD and links it into every model
# folder that uses it: the shard is copied with its md5 taken on the way, the SSD copy is read back and checked again,
# the drive's copy moves to BACKUP, and each model folder gets a symlink of the shard's name there pointing at the SSD
# copy. Swift-1.5 GSQ-RCO Q2_0 and IQ3_XXS share the same table bytes, so one SSD copy serves both. Run as root on
# the rig, with nothing loading the models.
#   ssh $RIG "SRC=<drive>/<model>/<name>-00003-of-00003.gguf MD5=<md5> SSD=<ssd dir>/<table name>.gguf \
#     BACKUP=<drive dir> LINKS='<drive>/<model>/<name>-00003-of-00003.gguf <drive>/<model2>/<name2>-00003-of-00003.gguf' \
#     bash -s" < table_to_ssd.sh
set -euo pipefail
: "${SRC:?}" "${MD5:?}" "${SSD:?}" "${BACKUP:?}" "${LINKS:?}"
[ -L "$SRC" ] && { echo "$SRC is already a symlink"; exit 1; }
mkdir -p "$(dirname "$SSD")" "$BACKUP"
echo "root free before: $(df -B1 --output=avail / | tail -1 | tr -d ' ') bytes"
got=$(tee "$SSD" < "$SRC" | md5sum | cut -d' ' -f1)
sync "$SSD"
echo 3 > /proc/sys/vm/drop_caches
back=$(md5sum < "$SSD" | cut -d' ' -f1)
echo "read from the drive $got, read back from the SSD $back, expected $MD5"
[ "$got" = "$MD5" ] && [ "$back" = "$MD5" ] || { echo "md5 mismatch: the drive copy stays, nothing linked"; exit 1; }
chown --reference="$SRC" "$SSD"
mv "$SRC" "$BACKUP/"
for l in $LINKS; do
  [ -e "$l" ] && ! [ -L "$l" ] && { echo "$l exists and is not a symlink, skipped"; continue; }
  ln -sfn "$SSD" "$l"
  chown -h --reference="$(dirname "$l")" "$l"
  echo "$l -> $SSD"
done
echo "root free after: $(df -B1 --output=avail / | tail -1 | tr -d ' ') bytes"
