#!/usr/bin/env bash
# r135_extra.sh: R135's (Q1421) three tables beside R123b's reader and R123a's ladder: the reserves, the swap, and the
# host's MemAvailable over time for two loads side by side.
#   DATA=<folder with the load folders> OUT=<dir> bash r135_extra.sh base=r123b_base embd=r123b_embd pp1=r135_pp1 ...
# Each load folder is rig_taper.sh's output (load-A/server.log at -lv 5, load-A/memory_loaded.txt and memory_after.txt
# from free -m, load-A/memavail.txt from the watchdog: epoch seconds and MemAvailable MiB every 2 s).
# Writes
#   $OUT/reserves.tsv  one row per `sched_reserve: reserving ...` block: the load, the context (a graph of fewer than
#                      1,000 nodes is the MTP draft's), nodes, splits, `sched copies`, every compute buffer, and whether
#                      LLAMA_PP_MULTI_SPLIT's line is in the block (C812's reader with the switch's column)
#   $OUT/swap.tsv      per load: swap used (MiB) after loading and after the replay, from free -m
#   $OUT/memavail.tsv  the first two loads given, side by side: seconds from each load's first sample, MemAvailable of
#                      each and the second minus the first (pairs by sample number; the watchdog samples every 2 s)
set -euo pipefail
: "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
printf 'load\tcontext\tblock\tnodes\tsplits\tsched_copies\tcompute_buffers_mib\tpp_multi_split_line\n' > "$OUT/reserves.tsv"
printf 'load\tswap_used_loaded_mib\tswap_used_after_mib\n' > "$OUT/swap.tsv"
names=()
for spec in "$@"; do
  n=${spec%%=*}; d=$DATA/${spec#*=}/load-A
  names+=("$n=$d")
  sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | awk -v load="$n" '
    /sched_reserve: reserving \.\.\./ { open = 1; bufs = ""; sw = "no"; k++ }
    open && /LLAMA_PP_MULTI_SPLIT: / { sw = "yes" }
    open && /compute buffer size = / { b = $0; sub(/.*sched_reserve: +/, "", b); sub(/ compute buffer size = +/, "=", b); sub(/ MiB.*/, "", b); bufs = bufs (bufs == "" ? "" : ", ") b }
    open && /graph: nodes = / { x = $0; sub(/.*nodes = /, "", x); sub(/,.*/, "", x); nodes = x + 0; y = $0; sub(/.*splits = /, "", y); sub(/,.*/, "", y); splits = y }
    open && /sched copies = / { z = $0; sub(/.*sched copies = /, "", z); ctx = (nodes < 1000) ? "draft" : "target"
      print load "\t" ctx "\t" k "\t" nodes "\t" splits "\t" z + 0 "\t" bufs "\t" sw; open = 0 }' >> "$OUT/reserves.tsv"
  printf '%s\t%s\t%s\n' "$n" "$(awk '/^Swap:/ { print $3 }' "$d/memory_loaded.txt")" \
    "$(awk '/^Swap:/ { print $3 }' "$d/memory_after.txt")" >> "$OUT/swap.tsv"
done
a=${names[0]#*=}; b=${names[1]#*=}
printf 's_%s\tmemavail_%s_mib\ts_%s\tmemavail_%s_mib\t%s_minus_%s_mib\n' "${names[0]%%=*}" "${names[0]%%=*}" \
  "${names[1]%%=*}" "${names[1]%%=*}" "${names[1]%%=*}" "${names[0]%%=*}" > "$OUT/memavail.tsv"
paste <(awk 'NR == 1 { t0 = $1 } { printf "%d\t%d\n", $1 - t0, $2 }' "$a/memavail.txt") \
      <(awk 'NR == 1 { t0 = $1 } { printf "%d\t%d\n", $1 - t0, $2 }' "$b/memavail.txt") \
  | awk -F'\t' 'NF == 4 && $1 != "" && $3 != "" { printf "%s\t%s\t%s\t%s\t%d\n", $1, $2, $3, $4, $4 - $2 }' >> "$OUT/memavail.tsv"
