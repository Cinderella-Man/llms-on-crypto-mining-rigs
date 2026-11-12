#!/usr/bin/env bash
# r134_ro8.sh: R134 (C809, Q1415). _RO8's 32 perf cases (kern against mt, the cases perf_table.py tags RO8) side by
# side: R120b on the -kern build (before C809) and R134 on the -c809 build, each pass by value, with each run's change,
# spread and class. Both verdict.tsv files are perf_table.py's (2026-11-12_r120b_kern_speed/run/perf_table.py).
# Usage: bash r134_ro8.sh <R120b verdict.tsv> <R134 verdict.tsv> > ro8_r120b_r134.tsv
# Columns: type m k n, then for R120b: mt_p1 mt_p2 kern_p1 kern_p2 class, then for R134: mt_p1 mt_p2 kern_p1 kern_p2
# change_p1 change_p2 spread class.
set -euo pipefail
awk -F'\t' -v OFS='\t' '
  FNR == 1 { f++; next }
  $1 != "kern-mt" || $2 != "RO8" { next }
  f == 1 { a[$3 OFS $4 OFS $5 OFS $6] = $7 OFS $8 OFS $9 OFS $10 OFS $14; next }
  f == 2 { b[$3 OFS $4 OFS $5 OFS $6] = $7 OFS $8 OFS $9 OFS $10 OFS $11 OFS $12 OFS $13 OFS $14; keys[++n] = $3 OFS $4 OFS $5 OFS $6 }
  END {
    print "type", "m", "k", "n", "r120b_mt_p1", "r120b_mt_p2", "r120b_kern_p1", "r120b_kern_p2", "r120b_class",
          "r134_mt_p1", "r134_mt_p2", "r134_kern_p1", "r134_kern_p2", "r134_change_p1", "r134_change_p2", "r134_spread", "r134_class"
    for (i = 1; i <= n; i++) print keys[i], (keys[i] in a ? a[keys[i]] : "-\t-\t-\t-\t-"), b[keys[i]]
  }' "$1" "$2"
