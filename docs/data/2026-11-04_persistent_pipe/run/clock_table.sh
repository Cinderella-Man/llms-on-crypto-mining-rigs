#!/usr/bin/env bash
# clock_table.sh: run_rig.sh's nvidia-smi log (smi.csv, a sample per card every 500 ms) as a table of 10-second
# windows: per window and step in progress (from run.log), how many card samples sat in P8, at the base clock (P0 below
# 1,400 MHz; the P106's base is 1,354), between, and at boost (1,800 MHz and up), and the mean power of a card.
# Usage: clock_table.sh <smi.csv> <run.log> > clocks_10s.tsv
set -euo pipefail
smi=$1
runlog=$2
awk -F', ' -v runlog="$runlog" '
BEGIN {
    # run.log: "<ISO time> <step>: <options>", "<ISO time> <step>: exit <rc>", "<ISO time> done, <n> failed"
    while ((getline line < runlog) > 0) {
        split(line, f, " ")
        s = f[2]
        sub(/:$/, "", s)
        if (line ~ /: exit [0-9]+$/ || s == "done,") continue
        n++
        st[n] = substr(f[1], 12, 8)
        sn[n] = s
    }
    print "window\tstep\tsamples\tP8\tbase\tbetween\tboost\tW_a_card"
}
NR > 1 {
    t = substr($1, 12, 8)
    step = "-"
    for (i = 1; i <= n; i++) if (t >= st[i]) step = sn[i]
    key = substr($1, 12, 7) "0\t" step
    if (!(key in cnt)) order[++m] = key
    cnt[key]++
    c = $4 + 0
    if ($3 == "P8") p8[key]++
    else if (c < 1400) lo[key]++
    else if (c >= 1800) hi[key]++
    else mid[key]++
    w[key] += $6
}
END {
    for (i = 1; i <= m; i++) {
        k = order[i]
        printf "%s\t%d\t%d\t%d\t%d\t%d\t%.1f\n", k, cnt[k], p8[k], lo[k], mid[k], hi[k], w[k] / cnt[k]
    }
}
' "$smi"
