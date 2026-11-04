#!/usr/bin/env bash
# header_copies.sh: header-only copies of GGUF shards for scripts/fit_probe.sh on a rig that lacks the files. Each copy
# holds the shard's header (its key-values and tensor table, up to the aligned start of the data, read by
# scripts/split_plan.py's reader) and is extended to the shard's full size with truncate, so llama.cpp's loader sees a
# complete file and its projection needs no weights; packed sparse (tar -S), a few MB per file.
#   FILES="<name>=<shard1>,<shard2>,... ..." OUT=<folder> bash header_copies.sh   # writes OUT/<name>.tar.gz
# Unpack on the rig with tar -xSzf; the shards keep their names. Runs on any CPU (reads headers only).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPTS=${SCRIPTS:-$HERE/../../../../scripts}
: "${FILES:?}" "${OUT:?}"
mkdir -p "$OUT"
for spec in $FILES; do
  name=${spec%%=*}; IFS=, read -ra shards <<< "${spec#*=}"
  d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"
  for link in "${shards[@]}"; do
    s=$(readlink -f "$link")  # a linked shard is read from its target and keeps the link's name
    n=$(python3 -I - "$SCRIPTS" "$s" <<'EOF'
# the header's length: the key-values and tensor table walked with split_plan.py's type table, up to the aligned
# start of the data
import struct, sys
sys.path.insert(0, sys.argv[1])
from split_plan import SCALAR
f = open(sys.argv[2], "rb")
rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def value(t):
    if t in SCALAR:
        return rd(SCALAR[t])
    if t == 8:
        return f.read(rd("<Q"))
    et, k = rd("<I"), rd("<Q")
    return [value(et) for _ in range(k)]


assert f.read(4) == b"GGUF"
_ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
align = 32
for _ in range(nkv):
    key = f.read(rd("<Q"))
    v = value(rd("<I"))
    align = v if key == b"general.alignment" else align
for _ in range(nt):
    f.read(rd("<Q"))
    [rd("<Q") for _ in range(rd("<I"))]
    rd("<I"), rd("<Q")
print((f.tell() + align - 1) // align * align)
EOF
)
    [ -n "$n" ] || { echo "$name: cannot read $s"; exit 1; }
    head -c "$n" "$s" > "$d/$(basename "$link")" && truncate -s "$(stat -c %s "$s")" "$d/$(basename "$link")"
    echo "$name: $(basename "$link") header $n bytes of $(stat -c %s "$s")"
  done
  tar -C "$OUT" -cSzf "$OUT/$name.tar.gz" "$name" && rm -rf "$d"
  echo "$name: $OUT/$name.tar.gz $(stat -c %s "$OUT/$name.tar.gz") bytes, md5 $(md5sum < "$OUT/$name.tar.gz" | cut -c1-32)"
done
