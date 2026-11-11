#!/usr/bin/env bash
# Samples one process and the host's paging every INTERVAL seconds until the process exits, one TSV line a sample:
# what the process's main thread waits on (state, kernel wait channel), its page faults, its resident and swapped
# memory, and the host's swap and fault counters. For C752 (Q1328): the ~117 ms host waits before card 0's launch
# that come with swapping on a 4 GB rig.
#
# Usage: PID=<llama-server pid> OUT=<file.tsv> [INTERVAL=0.05] bash proc_sampler.sh
# Run it as root (the wait channel of another user's thread needs it). It forks nothing per sample (bash builtins
# only), so it costs a few per cent of one core at 0.05 s.
#
# Columns: wall (s, $EPOCHREALTIME), uptime (s, /proc/uptime: boot time, 10 ms resolution), the main thread's state
# and wait channel (/proc/PID/task/PID), its major and minor faults (cumulative), the process's VmRSS and VmSwap (kB),
# and from /proc/vmstat (cumulative, pages): pswpin, pswpout, pgmajfault, workingset_refault_anon,
# workingset_refault_file; then MemAvailable and SwapFree (kB) from /proc/meminfo.
set -u
: "${PID:?PID of the process to sample}" "${OUT:?output TSV}"
INTERVAL=${INTERVAL:-0.05}
T=/proc/$PID/task/$PID
exec {sleep_fd}<> <(:)                # a pipe nobody writes: read -t on it sleeps without a fork

printf 'wall\tuptime\tstate\twchan\tmajflt\tminflt\tvm_rss_kb\tvm_swap_kb\tpswpin\tpswpout\tpgmajfault\twsr_anon\twsr_file\tmem_available_kb\tswap_free_kb\n' > "$OUT"
while [[ -d /proc/$PID ]]; do
    read -r up _ < /proc/uptime || break
    read -r stat < "$T/stat" 2>/dev/null || break
    rest=${stat##*) }                     # the fields after the command name, from field 3 (state)
    read -r -a f <<< "$rest"
    state=${f[0]} minflt=${f[7]} majflt=${f[9]}
    wchan=-
    read -r wchan < "$T/wchan" 2>/dev/null || true
    rss=- swp=-
    while read -r k v _; do
        case $k in VmRSS:) rss=$v ;; VmSwap:) swp=$v ;; esac
    done < "/proc/$PID/status"
    pin=- pout=- pmaj=- wra=- wrf=-
    while read -r k v; do
        case $k in
            pswpin) pin=$v ;; pswpout) pout=$v ;; pgmajfault) pmaj=$v ;;
            workingset_refault_anon) wra=$v ;; workingset_refault_file) wrf=$v ;;
        esac
    done < /proc/vmstat
    ma=- sf=-
    while read -r k v _; do
        case $k in MemAvailable:) ma=$v ;; SwapFree:) sf=$v ;; esac
    done < /proc/meminfo
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$EPOCHREALTIME" "$up" "$state" "${wchan:-0}" \
        "$majflt" "$minflt" "$rss" "$swp" "$pin" "$pout" "$pmaj" "$wra" "$wrf" "$ma" "$sf" >> "$OUT"
    read -r -t "$INTERVAL" -u "$sleep_fd" _ || true
done
