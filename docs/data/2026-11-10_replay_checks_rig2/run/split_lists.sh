#!/usr/bin/env bash
# A replay suite for a rig without one disk that holds it (here: 13.2 GB against 6.5, 8.6 and 4.1 GB free): the suite
# goes onto three disks and is joined by symlinks (relay_split.sh). This writes the three member lists from the tar's
# listing: a case's three files (.in, .ref, .meta) stay together, cases in id order, list 1 up to C1 bytes, list 2 up
# to C2 bytes, list 3 the rest; the suite's own small files go to list 1.
# Runs on the build machine (awk only). Usage: bash split_lists.sh <tar> <out dir> [C1 bytes] [C2 bytes]
set -euo pipefail
TAR=${1:?the set tar}; O=${2:?out dir}; C1=${3:-5600000000}; C2=${4:-5200000000}
mkdir -p "$O"
[ -s "$O/listing.txt" ] || nice -n 19 tar -tvf "$TAR" > "$O/listing.txt"
rm -f "$O"/list[123].txt
grep '^-' "$O/listing.txt" | awk '{print $3 "\t" $6}' | sort -t "$(printf '\t')" -k2,2 | awk -F'\t' -v c1="$C1" -v c2="$C2" -v o="$O" '
    $2 !~ /\/cases\// { print $2 > (o "/list1.txt"); s[1] += $1; n[1]++; next }
    {
        id = $2; sub(/.*\/cases\//, "", id); sub(/\..*/, "", id)
        if (id != last) {
            # the bin of the case that starts here: decided on the bytes placed so far
            if (s[1] < c1) b = 1; else if (s[2] < c2) b = 2; else b = 3
            last = id; cases[b]++
        }
        print $2 > (o "/list" b ".txt"); s[b] += $1; n[b]++
    }
    END { for (b = 1; b <= 3; b++) printf "list%d: %d files, %d cases, %d bytes\n", b, n[b], cases[b], s[b] }'
wc -l "$O"/list[123].txt
