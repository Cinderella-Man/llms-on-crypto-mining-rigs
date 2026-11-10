#!/usr/bin/env bash
# The rig's side of relay_split.sh: one pass of the suite's tar off the LAN (nc), its md5 taken on the way, and
# three tar readers that each unpack their own member list onto their own disk. No copy of the tar on the rig.
# Usage: bash recv_split.sh <listen address> <port> <lists dir> <dest 1> <dest 2> <dest 3>
# Writes in <dest 1>: recv.md5 (the stream's), recv.{1,2,3}.rc (each tar's exit status), recv.done
set -uo pipefail
A=${1:?address}; PORT=${2:?port}; L=${3:?lists dir}; D1=${4:?}; D2=${5:?}; D3=${6:?}
mkdir -p "$D1" "$D2" "$D3"
rm -f "$D1"/recv.md5 "$D1"/recv.[123].rc "$D1"/recv.done
timeout 3600 nc -l "$A" "$PORT" \
    | tee >(md5sum > "$D1/recv.md5.tmp"; mv "$D1/recv.md5.tmp" "$D1/recv.md5") \
          >(tar xf - -C "$D2" -T "$L/list2.txt"; echo $? > "$D1/recv.2.rc") \
          >(tar xf - -C "$D3" -T "$L/list3.txt"; echo $? > "$D1/recv.3.rc") \
    | tar xf - -C "$D1" -T "$L/list1.txt"
echo $? > "$D1/recv.1.rc"
for _ in $(seq 1 300); do
    [ -s "$D1/recv.md5" ] && [ -s "$D1/recv.2.rc" ] && [ -s "$D1/recv.3.rc" ] && break
    sleep 1
done
sync
date -Is > "$D1/recv.done"
