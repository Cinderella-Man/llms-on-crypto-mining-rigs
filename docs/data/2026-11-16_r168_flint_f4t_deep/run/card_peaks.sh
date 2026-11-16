#!/usr/bin/env bash
# card_peaks.sh <run folder>: each card's highest memory over each load of a deep-lane run (bench/deep_lane), from the
# lane's GPU monitor (gpu.csv, every card every 10 s) between the console's "LOAD <name> (<arm>)" and
# "LOAD <name> done" lines. Prints a TSV: load, arm, one column per card (MiB), and their sum (the cards' peaks need not
# fall in the same sample). Runs anywhere with bash and awk; the run folder is the lane's OUT (with console.txt beside
# it, as remote.sh collect leaves it, or the rig's WORK/console.txt given as CONSOLE).
set -euo pipefail
d=${1:?usage: card_peaks.sh <run folder>}
console=${CONSOLE:-$d/console.txt}
[ -r "$console" ] && [ -r "$d/gpu.csv" ] || { echo "need $console and $d/gpu.csv"; exit 1; }
awk -F', ' '
  FNR == NR {
    if (match($0, /^[0-9:]+ LOAD [^ ]+ \([^)]*\)$/)) { split($0, a, " "); st[a[3]] = a[1]; arm[a[3]] = substr(a[4], 2, length(a[4]) - 2); ord[++n] = a[3] }
    else if (match($0, /^[0-9:]+ LOAD [^ ]+ done$/)) { split($0, a, " "); en[a[3]] = a[1] }
    next
  }
  {
    split($1, t, " "); tm = substr(t[2], 1, 8); c = $2 + 0; m = $6; sub(/ MiB/, "", m)
    if (c > nc) nc = c
    for (i = 1; i <= n; i++) { l = ord[i]; if ((l in en) && tm >= st[l] && tm <= en[l] && m + 0 > pk[l, c]) pk[l, c] = m + 0 }
  }
  END {
    printf "load\tarm"; for (c = 0; c <= nc; c++) printf "\tcard%d_mib", c; printf "\tsum_mib\n"
    for (i = 1; i <= n; i++) {
      l = ord[i]; if (!(l in en)) continue
      s = 0; printf "%s\t%s", l, arm[l]
      for (c = 0; c <= nc; c++) { printf "\t%d", pk[l, c]; s += pk[l, c] }
      printf "\t%d\n", s
    }
  }' "$console" "$d/gpu.csv"
