#!/usr/bin/env bash
# Samples the host's memory and paging every INTERVAL seconds until the file STOP exists, one TSV line a sample: the wall
# time ($EPOCHREALTIME, s), MemAvailable, Cached, SwapTotal and SwapFree (kB, /proc/meminfo), and pswpin, pswpout
# (pages) and pgmajfault (cumulative, /proc/vmstat). Bash builtins only: it forks nothing per sample.
# Usage: OUT=<file.tsv> STOP=<file> [INTERVAL=0.5] bash vm_sampler.sh &
set -u
: "${OUT:?OUT=<file.tsv>}" "${STOP:?STOP=<file whose existence ends the sampling>}"
INTERVAL=${INTERVAL:-0.5}
exec {sleep_fd}<> <(:)                # a pipe nobody writes: read -t on it sleeps without a fork

printf 'wall\tmem_available_kb\tcached_kb\tswap_total_kb\tswap_free_kb\tpswpin\tpswpout\tpgmajfault\n' > "$OUT"
while [[ ! -e $STOP ]]; do
    ma=- ca=- st=- sf=-
    while read -r k v _; do
        case $k in MemAvailable:) ma=$v ;; Cached:) ca=$v ;; SwapTotal:) st=$v ;; SwapFree:) sf=$v ;; esac
    done < /proc/meminfo
    pin=- pout=- pmaj=-
    while read -r k v; do
        case $k in pswpin) pin=$v ;; pswpout) pout=$v ;; pgmajfault) pmaj=$v ;; esac
    done < /proc/vmstat
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$EPOCHREALTIME" "$ma" "$ca" "$st" "$sf" "$pin" "$pout" "$pmaj" >> "$OUT"
    read -r -t "$INTERVAL" -u "$sleep_fd" _ || true
done
