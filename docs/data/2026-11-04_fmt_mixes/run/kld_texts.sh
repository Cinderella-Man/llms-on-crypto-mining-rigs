#!/usr/bin/env bash
# The three texts side by side, from kld_agent.tsv, kld_wiki.tsv and kld_code.tsv (run/kld_table.py's tables): each
# file's change from the start (pmix-q2-fast) and its class on every text, and on how many of the three texts it
# gains and loses. The rows, their order and the first six columns are the agent table's.
#   bash kld_texts.sh <folder holding the three tables> > kld_texts.tsv
set -euo pipefail
d=${1:?}
awk -F'\t' -v OFS='\t' '
    FNR == 1 { t++; for (i = 1; i <= NF; i++) col[t, $i] = i; next }
    {
        n = $1
        if (t == 1) { order[++rows] = n; meta[n] = $2 OFS $3 OFS $4 OFS $5 OFS $6 }
        d[t, n] = $(col[t, "d_kld"]); v[t, n] = $(col[t, "verdict"])
    }
    END {
        print "name", "class", "layers", "donor", "delta_mib", "delta_ms_timed", "d_agent", "d_wiki", "d_code",
            "verdict_agent", "verdict_wiki", "verdict_code", "gains", "losses"
        for (r = 1; r <= rows; r++) {
            n = order[r]; g = l = 0
            for (k = 1; k <= 3; k++) { if (v[k, n] == "gain") g++; if (v[k, n] == "loss") l++ }
            if (v[1, n] == "start" || v[1, n] == "reference") g = l = "-"
            print n, meta[n], d[1, n], d[2, n], d[3, n], v[1, n], v[2, n], v[3, n], g, l
        }
    }' "$d/kld_agent.tsv" "$d/kld_wiki.tsv" "$d/kld_code.tsv"
