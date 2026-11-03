#!/usr/bin/env bash
# s12_tables.sh: Q226's memory tables from the rig's monitor logs (bench/agent_check/rig_server.sh's vmstat.log,
# meminfo.log, gpu.csv and server-<arm>.log). Run where the logs are, e.g. on the agent check's client:
#   NODRAFT=<folder with the no-draft arm's rig logs> DRAFT=<folder with the draft line's rig logs> OUT=<folder> \
#     bash s12_tables.sh
# Writes OUT/host_memory.csv (per line: MemAvailable and swap at the start, once loaded, at their lowest and highest),
# OUT/draft_load.csv (the draft line's vmstat rows from its start, with the latest MemAvailable sample and its age) and
# OUT/vram.csv (MiB used per card once loaded and the highest during the run).
set -euo pipefail
: "${NODRAFT:?}" "${DRAFT:?}" "${OUT:?}"
mkdir -p "$OUT"

start_of() { date -d "$(sed -n '1s/^=== \([^ ]*\) .*/\1/p' "$1")" +%s; }  # the server log's first line
offset_s() {  # offset_s LOG PATTERN: seconds after the start of the first line matching PATTERN (m.ss.mmm.uuu)
  grep -m1 -E "$2" "$1" | awk '{split($1, p, "."); print p[1] * 60 + p[2] + p[3] / 1000}'
}

nd_log=$(ls "$NODRAFT"/server-*.log | head -1); md_log=$(ls "$DRAFT"/server-*.log | head -1)
nd_t0=$(start_of "$nd_log"); md_t0=$(start_of "$md_log")
nd_up=$(awk -v a="$nd_t0" -v b="$(offset_s "$nd_log" 'llama_server: model loaded')" 'BEGIN {printf "%d", a + b}')
md_main=$(awk -v a="$md_t0" -v b="$(offset_s "$md_log" 'loading draft model')" 'BEGIN {printf "%d", a + b}')

{
  echo "line,phase,epoch,seconds_from_start,mem_available_mib,swap_used_mib"
  for spec in "nodraft $NODRAFT $nd_t0 $nd_up" "mtp $DRAFT $md_t0 $md_main"; do
    set -- $spec
    awk -v line="$1" -v t0="$3" -v t1="$4" '
      function row(ph, i) { printf "%s,%s,%d,%d,%d,%d\n", line, ph, t[i], t[i] - t0, a[i] / 1024, s[i] / 1024 }
      { n++; t[n] = $1; a[n] = $2; s[n] = $3 }
      END {
        row("start", 1)
        for (i = 1; i <= n && t[i] < t1; i++) ; if (i <= n) row(line == "mtp" ? "main model loaded, draft loading" : "loaded", i)
        lo = 0; hi = 0
        for (i = 1; i <= n; i++) if (t[i] >= t1) { if (!lo || a[i] < a[lo]) lo = i; if (!hi || s[i] > s[hi]) hi = i }
        if (lo) row("lowest MemAvailable after loading", lo); if (hi) row("most swap after loading", hi)
        row("last sample", n)
      }' "$2/meminfo.log"
  done
} > "$OUT/host_memory.csv"

# vmstat -t rows: r b swpd free buff cache si so bi bo in cs us sy id wa st gu date time
awk -v t0="$md_t0" 'FNR == NR { n++; mt[n] = $1; ma[n] = $2; next }
  FNR == 1 { print "epoch,seconds_from_start,runnable,blocked,swap_used_mib,free_mib,swap_in_kib_s,swap_out_kib_s," \
                   "read_mib_s,cpu_user_pct,cpu_system_pct,cpu_idle_pct,cpu_iowait_pct,mem_available_mib,mem_available_age_s" }
  $1 ~ /^[0-9]+$/ {
    cmd = "date -d \"" $(NF - 1) " " $NF "\" +%s"; cmd | getline t; close(cmd)
    if (t < t0) next
    k = 0; for (i = 1; i <= n && mt[i] <= t; i++) k = i
    printf "%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%d,%d,%d,%s,%s\n", t, t - t0, $1, $2, $3 / 1024, $4 / 1024, $7, $8,
      $9 / 1024, $13, $14, $15, $16, k ? int(ma[k] / 1024) : "", k ? t - mt[k] : ""
  }' "$DRAFT/meminfo.log" "$DRAFT/vmstat.log" > "$OUT/draft_load.csv"

{
  echo "line,card,mib_used_once_loaded,mib_used_highest"
  for spec in "nodraft $NODRAFT $nd_up" "mtp $DRAFT $md_main"; do
    set -- $spec
    awk -F', ' -v line="$1" -v t1="$3" '{
        cmd = "date -d \"" substr($1, 1, 19) "\" +%s"; cmd | getline t; close(cmd)
        c = $2; m = $6 + 0
        if (t >= t1 && !(c in first)) first[c] = m
        if (m > hi[c]) hi[c] = m
      } END { for (c = 0; c <= 8; c++) printf "%s,%d,%s,%d\n", line, c, (c in first) ? first[c] : "", hi[c] }' \
      "$2/gpu.csv"
  done
} > "$OUT/vram.csv"
echo "tables in $OUT: $(ls "$OUT" | tr '\n' ' ')"
