#!/usr/bin/env bash
# bench_cards.sh: bench/persistent_pipe run_rig.sh's nvidia-smi log (smi.csv, a sample per card every 500 ms) per step
# and card: the samples from SKIP seconds after the step's start (default 10: the set-up, the calibration and the
# clock's ramp) to its exit, in P8, at base (P0 below 1,400 MHz), between and at boost (1,800 MHz or more), the median
# SM clock and the mean power. Steps from run.log ("<ISO time> <step>: <options>" ... "<ISO time> <step>: exit <rc>").
# Usage: SKIP=10 bench_cards.sh <smi.csv> <run.log> <label> > cards.tsv
set -euo pipefail
smi=$1
runlog=$2
label=${3:-}
awk -F', ' -v runlog="$runlog" -v label="$label" -v skip="${SKIP:-10}" '
function sec(t) { split(t, h, ":"); return h[1] * 3600 + h[2] * 60 + h[3] }
BEGIN {
    while ((getline line < runlog) > 0) {
        split(line, f, " ")
        s = f[2]
        sub(/:$/, "", s)
        t = sec(substr(f[1], 12, 8))
        if (line ~ /: exit [0-9]+$/) { if (s in cur) t1[cur[s]] = t; continue }
        if (s == "done," || s == "pause") continue
        # a step that ran twice under one name is <step>-2 the second time, as run_rig.sh now names its files
        occ[s]++
        nm = occ[s] > 1 ? s "-" occ[s] : s
        cur[s] = nm; n++; name[n] = nm; t0[nm] = t
    }
    print "run\tstep\tcard\tsamples\tP8\tbase\tbetween\tboost\tmedian_MHz\tW_mean"
}
NR > 1 && $1 ~ /^[0-9]/ {
    t = sec(substr($1, 12, 8))
    st = ""
    for (i = 1; i <= n; i++) if (t >= t0[name[i]] + skip && t <= t1[name[i]]) st = name[i]
    if (st == "") next
    k = st SUBSEP ($2 + 0)
    if (!(k in cnt)) order[++m] = k
    cnt[k]++
    c = $4 + 0
    if ($3 == "P8") p8[k]++
    else if (c < 1400) lo[k]++
    else if (c >= 1800) hi[k]++
    else mid[k]++
    w[k] += $6
    hist[k, c]++
    if (!((k, c) in seen)) { seen[k, c] = 1; nv[k]++; v[k, nv[k]] = c }
}
END {
    for (i = 1; i <= m; i++) {
        k = order[i]
        for (a = 1; a <= nv[k]; a++) for (b = a + 1; b <= nv[k]; b++)
            if (v[k, b] < v[k, a]) { x = v[k, a]; v[k, a] = v[k, b]; v[k, b] = x }
        acc = 0; med = 0
        for (a = 1; a <= nv[k]; a++) { acc += hist[k, v[k, a]]; if (acc * 2 >= cnt[k]) { med = v[k, a]; break } }
        split(k, kk, SUBSEP)
        printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\n", label, kk[1], kk[2], cnt[k], p8[k], lo[k], mid[k], hi[k], med, w[k] / cnt[k]
    }
}
' "$smi"
