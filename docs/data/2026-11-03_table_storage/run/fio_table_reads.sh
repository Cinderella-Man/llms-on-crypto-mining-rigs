#!/usr/bin/env bash
# 4 KiB random reads at queue depth 1 and 32 on existing files, read-only, with O_DIRECT (Q214).
# The question: where can an on-disk n-gram table live when every generated token makes 11-14 random 4 KiB reads of it?
#
# Usage on the rig, as root (fio must be installed):
#   FILES="ssd=/path/on/the/ssd.gguf usbhdd=/path/on/the/usb/disk.gguf" OUT=/tmp/table_storage bash fio_table_reads.sh
# Pick files larger than the host's RAM; O_DIRECT bypasses the page cache in any case. Nothing is written.
set -euo pipefail
: "${FILES:?set FILES to label=path pairs}"
OUT=${OUT:-/tmp/table_storage}
RUNTIME=${RUNTIME:-30}
DEPTHS=${DEPTHS:-"1 32"}

mkdir -p "$OUT"
printf 'device\tfile_gib\tqd\tiops\tmib_s\tmean_us\tp50_us\tp99_us\n' > "$OUT/summary.tsv"
for pair in $FILES; do
  label=${pair%%=*}
  f=${pair#*=}
  for qd in $DEPTHS; do
    fio --name="$label-qd$qd" --filename="$f" --readonly --direct=1 --rw=randread --bs=4k \
        --ioengine=libaio --iodepth="$qd" --numjobs=1 --runtime="$RUNTIME" --time_based --ramp_time=5 \
        --norandommap --randrepeat=0 --output-format=json --output="$OUT/$label-qd$qd.json"
    python3 - "$OUT/$label-qd$qd.json" "$label" "$qd" "$(stat -c %s "$f")" >> "$OUT/summary.tsv" <<'PY'
import json, sys
path, label, qd, size = sys.argv[1:5]
r = json.load(open(path))["jobs"][0]["read"]
c = r["clat_ns"]
p = c.get("percentile", {})
print("\t".join([label, f"{int(size) / 2**30:.1f}", qd, f"{r['iops']:.0f}", f"{r['bw'] / 1024:.1f}",
                 f"{c['mean'] / 1000:.0f}", f"{p.get('50.000000', 0) / 1000:.0f}", f"{p.get('99.000000', 0) / 1000:.0f}"]))
PY
  done
done
cat "$OUT/summary.tsv"
