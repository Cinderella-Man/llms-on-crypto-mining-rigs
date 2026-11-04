#!/usr/bin/env bash
# summarize.sh LABEL=FILE ...: one TSV row per watchdog log (scripts/ram_watchdog.sh's TSV, or the older mem_guard.sh logs,
# whose first nine columns are the same: epoch avail cma usable swap-used swapout-in-window psi-some psi-full pid), with
# what each limit of the watchdog looks at, to set or check the limits against runs:
#   samples (rows with a guarded process), lowest usable (MemAvailable minus CmaFree, MiB), the longest run of seconds
#   (consecutive samples, one a second) with usable below 20 / 48 / 100 MiB, most swapped out in the window (MiB),
#   highest pressure (some, full, avg10 %), the longest run with full above 30 / 40 / 50 % and some above 40 %, and the
#   KILL lines. Runs where the files are (awk only); the logs of the first rows below are in the data folder's README.
#   bash summarize.sh s15_mtp=mem_guard_timed_arm.log s18_dflash2b=mem_guard_timed_dflash2b.log > history.tsv
set -u
printf 'label\tsamples\tlowest_usable_mib\tusable_below_20_s\tusable_below_48_s\tusable_below_100_s\tmax_swapout_mib\tmax_psi_some\tmax_psi_full\tfull_over_30_s\tfull_over_40_s\tfull_over_50_s\tsome_over_40_s\tkills\n'
for spec in "$@"; do
  label=${spec%%=*}
  file=${spec#*=}
  awk -v label="$label" '
    function run(k, cond) { if (cond) { r[k]++; if (r[k] > best[k]) best[k] = r[k] } else r[k] = 0 }
    /^#/ { if ($0 ~ / KILL /) kills++; next }
    NF != 9 || $9 == "-" { next }
    {
      n++
      if (prev && $1 - prev > 3) for (k in r) r[k] = 0
      prev = $1
      if (n == 1 || $4 < minu) minu = $4
      if ($6 > maxso) maxso = $6
      if ($7 + 0 > maxsome) maxsome = $7 + 0
      if ($8 + 0 > maxfull) maxfull = $8 + 0
      run("u20", $4 < 20); run("u48", $4 < 48); run("u100", $4 < 100)
      run("f30", $8 > 30); run("f40", $8 > 40); run("f50", $8 > 50); run("s40", $7 > 40)
    }
    END {
      printf "%s\t%d\t%s\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\n", label, n, minu, best["u20"], best["u48"], best["u100"],
        maxso, maxsome, maxfull, best["f30"], best["f40"], best["f50"], best["s40"], kills
    }' "$file"
done
