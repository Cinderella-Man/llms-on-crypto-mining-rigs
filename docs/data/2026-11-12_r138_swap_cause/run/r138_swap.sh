#!/usr/bin/env bash
# r138_swap.sh: R138's (Q1430) swap tables beside R123b's reader, R123a's ladder and R135's r135_extra.sh: the host just
# before each load, its swap after loading, after the replay and over time, and its lowest MemAvailable.
#   DATA=<folder with the load folders> OUT=<dir> bash r138_swap.sh embd=r138_embd pp1=r138_pp1 r135=r135_pp1 ...
# Each load folder is rig_taper.sh's output: quiet_check.txt (uptime and free -m just before the server starts),
# load-A/memory_loaded.txt and memory_after.txt (free -m after loading and after the replay), and load-A/memavail.txt
# (the watchdog every 2 s: epoch seconds, MemAvailable MiB and, since R138, SwapFree MiB). A load from before R138 has no
# SwapFree column: its over-time cells are empty.
# Swap used over time is SwapTotal (free -m's, after loading) minus SwapFree, both in whole MiB, so it can be 1 MiB off
# free -m's own figure.
# Writes
#   $OUT/swap_summary.tsv  per load: before the start (buff/cache, available, swap used, 1-minute load average), swap
#                          used after loading and after the replay, the first and the highest swap used over time and
#                          the second of the highest, the second swap first rose 16 MiB or more above the first sample,
#                          the lowest MemAvailable and its second (seconds from the watchdog's first sample)
#   $OUT/swap_time.tsv     per load and sample: seconds, MemAvailable MiB, swap used MiB
set -euo pipefail
: "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
printf 'load\tstart_buff_cache_mib\tstart_available_mib\tstart_swap_used_mib\tstart_load_avg_1m\tswap_used_loaded_mib\tswap_used_after_mib\tswap_used_first_sample_mib\tswap_used_max_mib\tswap_max_at_s\tswap_rise_16_at_s\tlowest_memavailable_mib\tlowest_at_s\n' > "$OUT/swap_summary.tsv"
printf 'load\ts\tmemavailable_mib\tswap_used_mib\n' > "$OUT/swap_time.tsv"
for spec in "$@"; do
  n=${spec%%=*}; f=$DATA/${spec#*=}; d=$f/load-A
  q=$f/quiet_check.txt
  bc=$(awk '/^Mem:/ { print $6 }' "$q"); av=$(awk '/^Mem:/ { print $7 }' "$q"); su=$(awk '/^Swap:/ { print $3 }' "$q")
  la=$(sed -nE 's/.*load average: ([0-9.]+),.*/\1/p' "$q" | head -n 1)
  tot=$(awk '/^Swap:/ { print $2 }' "$d/memory_loaded.txt")
  sl=$(awk '/^Swap:/ { print $3 }' "$d/memory_loaded.txt"); sa=$(awk '/^Swap:/ { print $3 }' "$d/memory_after.txt")
  awk -v n="$n" -v tot="$tot" 'NR == 1 { t0 = $1 } { printf "%s\t%d\t%d\t%s\n", n, $1 - t0, $2, (NF >= 3 ? tot - $3 : "") }' \
    "$d/memavail.txt" >> "$OUT/swap_time.tsv"
  stats=$(awk -v tot="$tot" 'NR == 1 { t0 = $1 } { s = $1 - t0
      if (NR == 1 || $2 < lo) { lo = $2; lo_s = s }
      if (NF >= 3) { u = tot - $3; if (!seen) { first = u; seen = 1 }
        if (u > mx || mx == "") { mx = u; mx_s = s }
        if (rise == "" && u - first >= 16) rise = s } }
    END { printf "%s\t%s\t%s\t%s\t%d\t%d", (seen ? first : ""), mx, mx_s, rise, lo, lo_s }' "$d/memavail.txt")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$bc" "$av" "$su" "$la" "$sl" "$sa" "$stats" >> "$OUT/swap_summary.tsv"
done
