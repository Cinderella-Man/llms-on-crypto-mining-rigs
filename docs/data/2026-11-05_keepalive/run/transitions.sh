#!/usr/bin/env bash
# transitions.sh: from C532's nvidia-smi log (docs/data/2026-11-05_card_clocks: a sample per card every ~137 ms) and its
# marks, every change of a card's P-state, SM clock or clock-event reasons from the server's start to the log's end,
# with the time in seconds from the server's start, and the marks themselves. It shows how long each card stays in P0
# after its last work, and which cards the driver never marks idle (reason bit 0x1).
# Usage: transitions.sh <smi.csv or smi.csv.gz> <marks.tsv> > transitions.tsv
set -euo pipefail
smi=$1
marks=$2
cat_smi() { case $smi in *.gz) zcat "$smi" ;; *) cat "$smi" ;; esac; }
t0=$(awk -F'\t' '$2 == "server_start" { print $1 }' "$marks")
# the log's time of day is local time; its date and the marks' epoch seconds give the offset
cat_smi | awk -F', ' -v t0="$t0" -v marks="$marks" '
BEGIN {
    while ((getline line < marks) > 0) {
        split(line, f, "\t")
        nm++; mt[nm] = f[1]; mn[nm] = f[2]
    }
    print "s_from_server_start\twhat\tcard\tpstate\tsm_mhz\tpower_w\treason"
    # the first sample of a day at midnight local time, from the first mark (smi_start) taken as the first sample
}
NR == 1 {
    split($1, d, " "); split(d[2], h, ":")
    tod0 = h[1] * 3600 + h[2] * 60 + h[3]
    base = mt[1] - tod0          # epoch seconds of local midnight (smi_start is the log'"'"'s first sample)
}
{
    split($1, d, " "); split(d[2], h, ":")
    t = base + h[1] * 3600 + h[2] * 60 + h[3] - t0
    if (t < -2) next
    while (im < nm && mt[im + 1] - t0 <= t) { im++; printf "%.1f\tmark %s\t\t\t\t\t\n", mt[im] - t0, mn[im] }
    c = $2; k = $3 " " $4 " " $7
    if (k != last[c]) { printf "%.1f\tchange\t%s\t%s\t%s\t%s\t%s\n", t, c, $3, $4, $6, $7; last[c] = k }
}
END { while (im < nm) { im++; printf "%.1f\tmark %s\t\t\t\t\t\n", mt[im] - t0, mn[im] } }'
