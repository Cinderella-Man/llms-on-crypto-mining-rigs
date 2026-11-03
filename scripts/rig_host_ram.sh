printf '%s\n' "$2" > "$1" ||#!/usr/bin/env bash
# rig_host_ram.sh: where the host RAM of a 4 GB rig goes, and the reversible settings that give some back.
# Background and the expected numbers: docs/research/notes/host_ram_4gb.md; the run order for a rig:
# docs/data/2026-11-04_rig_host_ram/README.md. Run it as root on the rig, read from stdin over ssh:
#
#   ssh "$RIG" 'bash -s -- census before' < scripts/rig_host_ram.sh        # read-only; one TSV block
#   ssh "$RIG" 'bash -s -- delta before after' < scripts/rig_host_ram.sh   # what changed between two censuses
#   ssh "$RIG" 'bash -s -- apply thp' < scripts/rig_host_ram.sh            # a runtime-only setting; prints the delta
#   ssh "$RIG" 'bash -s -- revert thp' < scripts/rig_host_ram.sh
#   ssh "$RIG" 'bash -s -- boot-howto' < scripts/rig_host_ram.sh           # the GRUB steps, as text; edits nothing
#   ssh "$RIG" "BIN=/path/to/bin MODEL=/path/to/small.gguf bash -s -- contexts" < scripts/rig_host_ram.sh
#
# Modes:
#   census [NAME]            Read-only. One "key<TAB>value" row per fact: memory (MemTotal, MemAvailable, MemFree, CmaTotal,
#                            CmaFree, Shmem, AnonPages, ...), the CMA pageblocks of /proc/pagetypeinfo, the kernel command line,
#                            what could mark pageblocks CMA without a CMA area (the kernel's KHO configuration, debugfs and
#                            log), THP, the swiotlb pool from the kernel log, zswap and zram, swap, MGLRU, vm sysctls, memory
#                            pressure, journald's disk and RAM use, the system and user slices' cgroup memory, and, for a
#                            running llama-server, RSS and PSS, VmLck and VmPin, Shmem and anonymous pages, its cgroup and the
#                            CUDA variables of its environment. With NAME the block is also saved as $OUT/NAME.tsv.
#                            Keys have a fixed order, so two runs diff cleanly; values in kB end in _kB.
#   delta BEFORE AFTER       The rows that differ between two census files (a path, or a NAME saved by census), with the change
#                            in the unit of the row and in MiB for _kB rows. Plain text processing; no rig access needed.
#   apply SETTING [VALUE]    Set one runtime-only setting, then print the census delta. Nothing here survives a reboot.
#   revert SETTING           Put back what apply saved (kept in $STATE, a tmpfs); prints the census delta.
#                            SETTING (default VALUE):
#                              thp        transparent huge pages "enabled" (madvise)
#                              mglru      /sys/kernel/mm/lru_gen/min_ttl_ms (1000)
#                              watermark  vm.watermark_scale_factor (50)
#                              cgroup     MemoryHigh of system.slice and user.slice, systemctl set-property --runtime (192M);
#                                         refused while llama-server runs inside either slice (it would be throttled): start it
#                                         in its own slice, e.g. systemd-run --slice=llama.slice
#                              journald   RuntimeMaxUse as a drop-in under /run/systemd/journald.conf.d (16M)
#                              zswap      max_pool_percent and compressor, then enabled (5, lz4); a latency tool, not memory
#   boot-howto               Prints the one-time GRUB steps (kho=off, and swiotlb=4096 when the pool is 64 MiB) and the permanent
#                            version. Text only: this script never edits GRUB and never reboots.
#   contexts                 Q565's protocol: an idle llama-server per card count and per arm, with the host memory after its
#                            load. Needs BIN (llama.cpp's bin/) and MODEL (a small file that fits one card). Refuses to run while
#                            a llama-server is running.
#
# Environment: OUT (census files and contexts results, default /var/tmp/rig_host_ram), STATE (apply's saved values, default
# /run/rig_host_ram), SETTLE (seconds to wait after a change, default 5, or after a load, default 10), SERVER_NAME (the
# process to look for, default llama-server), FORCE=1 (overrides the refusals of apply mglru when MGLRU is off, and of apply
# cgroup when llama-server runs inside a limited slice). contexts: COUNTS (card counts, default "1 3 5 9"), ARMS (default
# "unset CUDA_SCALE_LAUNCH_QUEUES=0.5x CUDA_SCALE_LAUNCH_QUEUES=0.25x": "unset" is none of the variables, any other word is
# one VAR=value; 4x is refused), SERVER_ARGS (default "-ngl 99 -c 2048 -np 1 --cache-ram 0"), PORT (18080), LOAD_TIMEOUT
# (600 s), SETTLE_AFTER (5 s, after each stop).
set -u
export LC_ALL=C

OUT=${OUT:-/var/tmp/rig_host_ram}
STATE=${STATE:-/run/rig_host_ram}
SERVER_NAME=${SERVER_NAME:-llama-server}
CGROOT=/sys/fs/cgroup
SLICES="system.slice user.slice"
ENV_VARS="CUDA_SCALE_LAUNCH_QUEUES GGML_CUDA_NO_PINNED CUDA_MODULE_LOADING CUDA_DEVICE_MAX_CONNECTIONS CUDA_VISIBLE_DEVICES CUDA_FORCE_PRELOAD_LIBRARIES LLAMA_INPUT_ON_GPU LLAMA_DIO_STAGING"

die() { echo "rig_host_ram: $*" >&2; exit 1; }
need_root() { [ "$(id -u)" = 0 ] || die "run as root"; }
kv() { printf '%s\t%s\n' "$1" "${2:-NA}"; }
# The first line of a file, printable characters only, or nothing if it cannot be read.
one() { head -n 1 "$1" 2>/dev/null | head -c 200 | tr -d '\n' | tr -c '[:print:]' ' ' | sed 's/ *$//'; }
# The text on stdin, or "none" when it is empty.
nz() { local v; v=$(cat); printf '%s' "${v:-none}"; }
# The selected word of a kernel switch file such as "always [madvise] never".
bracket() { sed -n 's/.*\[\(.*\)\].*/\1/p' "$1" 2>/dev/null | head -n 1; }
# du in kB, NA when the directory is missing.
dukb() { local v; v=$(du -sk "$1" 2>/dev/null | awk '{print $1}'); printf '%s' "${v:-NA}"; }
# A cgroup file holding bytes, as kB.
cgkb() { local v; v=$(head -n 1 "$1" 2>/dev/null); case $v in ''|*[!0-9]*) printf NA;; *) printf '%s' $((v / 1024));; esac; }
cgstat() { awk -v k="$2" '$1 == k { printf "%d", $2 / 1024; f = 1 } END { if (!f) printf "NA" }' "$1/memory.stat" 2>/dev/null; }
# Lines of the kernel log matching an extended regex (the ring buffer first, then the journal of this boot).
klog() {
    local o
    o=$(dmesg 2>/dev/null | grep -i -E "$1")
    [ -n "$o" ] || o=$(journalctl -k -b --no-pager 2>/dev/null | grep -i -E "$1")
    printf '%s\n' "$o" | grep -v '^$'
}
llama_pids() { ps -eo pid=,comm= 2>/dev/null | awk -v n="$SERVER_NAME" '$2 == n { print $1 }'; }
cgroup_of() { awk -F: '$1 == "0" { print $3 }' "/proc/$1/cgroup" 2>/dev/null; }
# A saved census: a path, or a name under $OUT.
path_of() { case $1 in */*) printf '%s' "$1";; *) printf '%s/%s.tsv' "$OUT" "${1%.tsv}";; esac; }

# ---------------------------------------------------------------------------------------------------------------- census

census_meta() {
    kv meta.format 1
    kv meta.date "$(date +%Y-%m-%dT%H:%M:%S%z)"
    kv meta.kernel "$(uname -r)"
    kv meta.uptime_s "$(awk '{ printf "%d", $1 }' /proc/uptime 2>/dev/null)"
    kv meta.uid "$(id -u)"
    kv gpu.count "$(ls /proc/driver/nvidia/gpus 2>/dev/null | grep -c .)"
    kv nvidia_drm.modeset "$(one /sys/module/nvidia_drm/parameters/modeset)"
    kv services.running "$(systemctl list-units --type=service --state=running --no-legend --no-pager 2>/dev/null | grep -c .)"
    kv systemd-oomd.active "$(systemctl is-active systemd-oomd 2>/dev/null)"
}

census_mem() {
    awk 'BEGIN {
            n = split("MemTotal MemAvailable MemFree Cached Buffers Shmem AnonPages Mapped Mlocked Unevictable Slab SUnreclaim KernelStack PageTables Percpu VmallocUsed CmaTotal CmaFree SwapTotal SwapFree Zswap Zswapped AnonHugePages", w, " ")
        }
        { k = $1; sub(":", "", k); v[k] = $2 }
        END {
            for (i = 1; i <= n; i++) printf "mem.%s_kB\t%s\n", w[i], (w[i] in v) ? v[w[i]] : "NA"
            if (("MemAvailable" in v) && ("CmaFree" in v)) printf "mem.usable_kB\t%.0f\n", v["MemAvailable"] - v["CmaFree"]
            else print "mem.usable_kB\tNA"
            if (("SwapTotal" in v) && ("SwapFree" in v)) printf "mem.swap_used_kB\t%.0f\n", v["SwapTotal"] - v["SwapFree"]
            else print "mem.swap_used_kB\tNA"
        }' /proc/meminfo
}

# Pageblocks by migrate type from /proc/pagetypeinfo (root only). The CMA column is found from the table's header.
census_pageblocks() {
    local psz
    psz=$(getconf PAGESIZE 2>/dev/null)
    if ! [ -r /proc/pagetypeinfo ] || ! head -c 1 /proc/pagetypeinfo > /dev/null 2>&1; then
        kv pageblocks.CMA NA
        kv pageblocks.CMA_MiB NA
        return
    fi
    awk -v ps="${psz:-4096}" '
        /^Pages per block:/ { ppb = $4 }
        /^Number of blocks type/ { inblk = 1; for (i = 5; i <= NF; i++) name[i] = $i; nf = NF; next }
        inblk && /^Node/ { zone = $4; for (i = 5; i <= NF; i++) { tot[name[i]] += $i; zt[name[i] "." zone] += $i }; zs[zone] = 1; next }
        inblk { inblk = 0 }
        END {
            mib = ppb * ps / 1048576
            nt = split("Unmovable Movable Reclaimable HighAtomic CMA Isolate", t, " ")
            for (i = 1; i <= nt; i++) printf "pageblocks.%s\t%s\n", t[i], (t[i] in tot) ? sprintf("%.0f", tot[t[i]]) : "NA"
            printf "pageblocks.size_MiB\t%s\n", (ppb > 0) ? mib : "NA"
            printf "pageblocks.CMA_MiB\t%s\n", (("CMA" in tot) && ppb > 0) ? sprintf("%.0f", tot["CMA"] * mib) : "NA"
            nz = split("DMA DMA32 Normal Movable", z, " ")
            for (i = 1; i <= nz; i++) if (z[i] in zs) printf "pageblocks.CMA.%s\t%s\n", z[i], (("CMA." z[i]) in zt) ? sprintf("%.0f", zt["CMA." z[i]]) : "NA"
        }' /proc/pagetypeinfo
}

# The kernel command line, the parameters that matter here, KHO's configuration, and the swiotlb pool.
KCFG='^CONFIG_(KEXEC_HANDOVER|KEXEC_HANDOVER_DEBUGFS|KEXEC_HANDOVER_ENABLE_DEFAULT|MEMBLOCK_KHO_SCRATCH|CMA|SWIOTLB_DYNAMIC)='
census_boot() {
    local k v cfg f pool line sl tot
    kv cmdline "$(tr '\n' ' ' < /proc/cmdline 2>/dev/null | sed -E 's/(UUID|PARTUUID)=[0-9A-Fa-f-]+/\1=.../g; s/ +$//')"
    for k in kho kho_bootmem kho_scratch swiotlb cma crashkernel transparent_hugepage; do
        v=$(tr ' ' '\n' < /proc/cmdline 2>/dev/null | awk -F= -v k="$k" '
            $1 == k { v = $0; sub(/^[^=]*=?/, "", v); print (v == "" ? "set" : v); f = 1; exit }
            END { if (!f) print "unset" }')
        kv "cmdline.$k" "$v"
    done
    f=/boot/config-$(uname -r)
    cfg=NA
    if [ -r "$f" ]; then
        cfg=$(grep -E "$KCFG" "$f" | sed 's/^CONFIG_//' | tr '\n' ',' | sed 's/,$//' | nz)
    elif [ -r /proc/config.gz ] && command -v zcat > /dev/null; then
        cfg=$(zcat /proc/config.gz | grep -E "$KCFG" | sed 's/^CONFIG_//' | tr '\n' ',' | sed 's/,$//' | nz)
    fi
    kv kho.config "$cfg"
    case ",$cfg," in
        *,NA,*) kv kho.in_kernel NA ;;
        *,KEXEC_HANDOVER=y,*) kv kho.in_kernel yes ;;
        *) kv kho.in_kernel no ;;
    esac
    # debugfs lists one scratch region per line (low, global, one per node), in hex bytes; their sum is the CMA-marked memory
    sl=$(tr -c '[:print:]' ' ' < /sys/kernel/debug/kho/out/scratch_len 2> /dev/null | head -c 400 | sed 's/ *$//')
    kv kho.scratch_len "${sl:-NA}"
    tot=NA
    if [ -n "$sl" ]; then
        tot=0
        for v in $sl; do
            case $v in 0x[0-9a-fA-F]*|[0-9]*) tot=$((tot + v)) ;; *) tot=NA; break ;; esac
        done
        [ "$tot" = NA ] || tot=$((tot / 1048576))
    fi
    kv kho.scratch_MiB "$tot"
    kv cma.areas "$(ls /sys/kernel/mm/cma 2>/dev/null | tr '\n' ',' | sed 's/,$//' | nz)"
    line=$(klog 'software IO TLB: .*mapped' | head -n 1)
    pool=$(printf '%s' "$line" | sed -n 's/.*(\([0-9][0-9]*\)MB).*/\1/p')
    kv swiotlb.pool_MiB "${pool:-none-in-log}"
    kv swiotlb.nslabs "$(one /sys/kernel/debug/swiotlb/io_tlb_nslabs)"
    kv swiotlb.used_slabs "$(one /sys/kernel/debug/swiotlb/io_tlb_used)"
}

census_policy() {
    local p
    kv thp.enabled "$(bracket /sys/kernel/mm/transparent_hugepage/enabled)"
    kv thp.defrag "$(bracket /sys/kernel/mm/transparent_hugepage/defrag)"
    kv thp.max_ptes_none "$(one /sys/kernel/mm/transparent_hugepage/khugepaged/max_ptes_none)"
    kv lru_gen.enabled "$(one /sys/kernel/mm/lru_gen/enabled)"
    kv lru_gen.min_ttl_ms "$(one /sys/kernel/mm/lru_gen/min_ttl_ms)"
    for p in watermark_scale_factor min_free_kbytes swappiness overcommit_memory; do kv "vm.$p" "$(one "/proc/sys/vm/$p")"; done
    for p in enabled compressor zpool max_pool_percent; do kv "zswap.$p" "$(one "/sys/module/zswap/parameters/$p")"; done
    kv zswap.pool_bytes "$(one /sys/kernel/debug/zswap/pool_total_size)"
    kv zswap.stored_pages "$(one /sys/kernel/debug/zswap/stored_pages)"
    kv zswap.written_back_pages "$(one /sys/kernel/debug/zswap/written_back_pages)"
    kv zram.devices "$(ls /sys/block 2>/dev/null | grep -c '^zram')"
    kv swap.devices "$(swapon --noheadings --show=NAME 2>/dev/null | grep -c .)"
    kv swap.types "$(swapon --noheadings --show=TYPE 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//' | nz)"
}

# Reclaim and pressure counters: their deltas show stalls, swapping and kills between two censuses.
census_pressure() {
    awk 'BEGIN { n = split("pswpin pswpout oom_kill pgmajfault pgscan_direct pgscan_kswapd", w, " ") }
         { v[$1] = $2 }
         /^allocstall/ { as += $2 }
         END {
            for (i = 1; i <= n; i++) printf "vmstat.%s\t%s\n", w[i], (w[i] in v) ? v[w[i]] : "NA"
            printf "vmstat.allocstall\t%.0f\n", as
         }' /proc/vmstat
    if [ -r /proc/pressure/memory ]; then
        awk '{ k = $1; for (i = 2; i <= NF; i++) { split($i, a, "="); if (a[1] == "avg10") print "psi.mem." k ".avg10\t" a[2]; if (a[1] == "total") print "psi.mem." k ".total_us\t" a[2] } }' /proc/pressure/memory
    else
        kv psi.mem.some.avg10 NA
        kv psi.mem.full.avg10 NA
    fi
}

census_journald() {
    local v
    kv journald.disk_usage "$(journalctl --disk-usage 2>/dev/null | sed -n 's/.*take up \([^ ]*\) in.*/\1/p')"
    kv journald.run_kB "$(dukb /run/log/journal)"
    kv journald.var_kB "$(dukb /var/log/journal)"
    v=$(systemd-analyze cat-config systemd/journald.conf 2>/dev/null | awk -F= '/^[ \t]*RuntimeMaxUse[ \t]*=/ { v = $2; sub(/[ \t]+$/, "", v) } END { print v }')
    kv journald.RuntimeMaxUse "${v:-default}"
}

census_cgroups() {
    local s d p
    for s in $SLICES; do
        d=$CGROOT/$s
        kv "cg.$s.MemoryHigh" "$(systemctl show -p MemoryHigh --value "$s" 2>/dev/null)"
        kv "cg.$s.current_kB" "$(cgkb "$d/memory.current")"
        kv "cg.$s.swap_kB" "$(cgkb "$d/memory.swap.current")"
        for p in anon file shmem; do kv "cg.$s.${p}_kB" "$(cgstat "$d" "$p")"; done
    done
}

census_env() {
    local v
    for v in $ENV_VARS; do kv "shell.env.$v" "$(printenv "$v" || echo unset)"; done
    kv shell.etc_environment "$(grep -s -E '^(export )?(CUDA_|GGML_|LLAMA_)' /etc/environment | tr '\n' ' ' | sed 's/ $//' | nz)"
}

# The first llama-server is llama0, the second llama1, and so on.
census_llama() {
    local pids n i=0 pid p cg d v envtxt
    pids=$(llama_pids)
    n=$(printf '%s\n' "$pids" | grep -c .)
    kv llama.n "$n"
    for pid in $pids; do
        p=llama$i
        i=$((i + 1))
        [ -d "/proc/$pid" ] || continue
        kv "$p.pid" "$pid"
        awk -v p="$p." 'BEGIN { n = split("VmRSS VmHWM VmLck VmPin VmSwap RssAnon RssFile RssShmem", w, " ") }
            { k = $1; sub(":", "", k); v[k] = $2 }
            END { for (i = 1; i <= n; i++) printf "%s%s_kB\t%s\n", p, w[i], (w[i] in v) ? v[w[i]] : "NA" }' "/proc/$pid/status" 2>/dev/null
        awk -v p="$p." 'BEGIN { n = split("Rss Pss Pss_Anon Pss_File Pss_Shmem Swap SwapPss Locked", w, " ") }
            { k = $1; sub(":", "", k); v[k] = $2 }
            END { for (i = 1; i <= n; i++) printf "%ssmaps_%s_kB\t%s\n", p, w[i], (w[i] in v) ? v[w[i]] : "NA" }' "/proc/$pid/smaps_rollup" 2>/dev/null
        envtxt=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
        for v in $ENV_VARS; do
            if [ -z "$envtxt" ]; then kv "$p.env.$v" NA; continue; fi
            kv "$p.env.$v" "$(printf '%s\n' "$envtxt" | awk -F= -v k="$v" '$1 == k { sub(/^[^=]*=/, ""); print; f = 1; exit } END { if (!f) print "unset" }')"
        done
        cg=$(cgroup_of "$pid")
        kv "$p.cgroup" "$cg"
        d=$CGROOT$cg
        kv "$p.cg.current_kB" "$(cgkb "$d/memory.current")"
        kv "$p.cg.swap_kB" "$(cgkb "$d/memory.swap.current")"
        for v in anon file shmem; do kv "$p.cg.${v}_kB" "$(cgstat "$d" "$v")"; done
    done
}

census() {
    census_meta
    census_mem
    census_pageblocks
    census_boot
    census_policy
    census_pressure
    census_journald
    census_cgroups
    census_env
    census_llama
}

# --------------------------------------------------------------------------------------------------------------- delta

# delta BEFORE AFTER (two files of census rows): the rows that differ, with the change.
delta() {
    printf 'key\tbefore\tafter\tchange\tMiB\n'
    awk -F'\t' '
        function isnum(s) { return s ~ /^-?[0-9]+$/ }
        function row(k, b, a,    c, m) {
            if (k == "meta.date" || k == "meta.uptime_s" || b == a) return
            c = "changed"; m = ""
            if (isnum(a) && isnum(b)) { c = sprintf("%+.0f", a - b); if (k ~ /_kB$/) m = sprintf("%+.1f", (a - b) / 1024) }
            printf "%s\t%s\t%s\t%s\t%s\n", k, b, a, c, m
            shown++
        }
        FNR == NR { bv[$1] = $2; bk[++nb] = $1; next }
        { av[$1] = $2; ak[++na] = $1 }
        END {
            for (i = 1; i <= na; i++) row(ak[i], (ak[i] in bv) ? bv[ak[i]] : "-", av[ak[i]])
            for (i = 1; i <= nb; i++) if (!(bk[i] in av)) row(bk[i], bv[bk[i]], "-")
            if (!shown) print "# no change"
        }' "$1" "$2"
}

cmd_census() {
    if [ -n "${1:-}" ]; then
        mkdir -p "$OUT" || die "cannot create $OUT"
        census | tee "$(path_of "$1")"
    else
        census
    fi
}

cmd_delta() {
    [ $# -eq 2 ] || die "usage: delta BEFORE AFTER"
    local a b
    a=$(path_of "$1")
    b=$(path_of "$2")
    [ -s "$a" ] || die "no census file $a"
    [ -s "$b" ] || die "no census file $b"
    delta "$a" "$b"
}

# -------------------------------------------------------------------------------------------------- apply and revert

state_save() { mkdir -p "$STATE" && { [ -e "$STATE/$1" ] || printf '%s\n' "$2" > "$STATE/$1"; }; }  # the value from before the first apply
state_get() { [ -e "$STATE/$1" ] && head -n 1 "$STATE/$1"; }
state_drop() { rm -f "$STATE/$1"; }
# Write a sysfs or procfs value and show what the file holds afterwards.
put() {
    printf '%s\n' "$2" > "$1" || { echo "cannot write '$2' to $1" >&2; return 1; }
    echo "set $1 -> $(one "$1")"
}
revert_file() {  # revert_file STATE_KEY FILE
    local old
    if old=$(state_get "$1"); then put "$2" "$old" && state_drop "$1"; else echo "$1: nothing saved since boot; left as it is"; fi
}

THP=/sys/kernel/mm/transparent_hugepage/enabled
apply_thp() {
    [ -w "$THP" ] || { echo "no THP switch at $THP" >&2; return 1; }
    echo "THP was [$(bracket "$THP")]"
    state_save thp "$(bracket "$THP")" || return 1
    put "$THP" "${1:-madvise}"
}
revert_thp() { revert_file thp "$THP"; }

TTL=/sys/kernel/mm/lru_gen/min_ttl_ms
apply_mglru() {
    [ -w "$TTL" ] || { echo "no MGLRU switch at $TTL (kernel without lru_gen)" >&2; return 1; }
    case $(one /sys/kernel/mm/lru_gen/enabled) in
        0|0x0|0x0000)
            if [ "${FORCE:-0}" != 1 ]; then echo "MGLRU is off here (lru_gen/enabled is 0): min_ttl_ms would do nothing; FORCE=1 sets it anyway" >&2; return 1; fi ;;
    esac
    echo "min_ttl_ms was $(one "$TTL"); the kernel kills llama-server (oom_score_adj 1000) instead of thrashing once the last ${1:-1000} ms of pages cannot stay"
    state_save mglru "$(one "$TTL")" || return 1
    put "$TTL" "${1:-1000}"
}
revert_mglru() { revert_file mglru "$TTL"; }

WM=/proc/sys/vm/watermark_scale_factor
apply_watermark() {
    [ -w "$WM" ] || { echo "no $WM" >&2; return 1; }
    echo "watermark_scale_factor was $(one "$WM")"
    state_save watermark "$(one "$WM")" || return 1
    put "$WM" "${1:-50}"
}
revert_watermark() { revert_file watermark "$WM"; }

# MemoryHigh throttles a cgroup and reclaims its pages first; a llama-server inside the slice would be throttled too.
apply_cgroup() {
    local val=${1:-192M} s pid cg bad=""
    for pid in $(llama_pids); do
        cg=$(cgroup_of "$pid")
        case $cg in /system.slice/*|/user.slice/*) bad="$bad $pid:$cg";; esac
    done
    if [ -n "$bad" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "refused: $SERVER_NAME runs inside a slice this would throttle ($bad)." >&2
        echo "Start it in its own slice (systemd-run --slice=llama.slice ...), or set FORCE=1." >&2
        return 1
    fi
    for s in $SLICES; do
        state_save "cgroup.$s" "$(systemctl show -p MemoryHigh --value "$s" 2>/dev/null)" || return 1
        systemctl set-property --runtime "$s" "MemoryHigh=$val" || return 1
        echo "$s: MemoryHigh $(state_get "cgroup.$s") -> $val ($(systemctl show -p MemoryHigh --value "$s"))"
    done
}
revert_cgroup() {
    local s old
    for s in $SLICES; do
        if old=$(state_get "cgroup.$s"); then
            systemctl set-property --runtime "$s" "MemoryHigh=$old" && state_drop "cgroup.$s" && echo "$s: MemoryHigh back to ${old:-infinity}"
        else
            echo "$s: nothing saved since boot; left as it is"
        fi
    done
}

JDROP=/run/systemd/journald.conf.d/zz-rig-host-ram.conf
apply_journald() {
    local val=${1:-16M}
    mkdir -p "$(dirname "$JDROP")" || return 1
    printf '[Journal]\nRuntimeMaxUse=%s\n' "$val" > "$JDROP" || return 1
    systemctl restart systemd-journald || return 1
    journalctl --rotate > /dev/null 2>&1
    echo "RuntimeMaxUse=$val in $JDROP (gone at reboot; to keep it, copy it to /etc/systemd/journald.conf.d/)"
}
revert_journald() {
    [ -e "$JDROP" ] || { echo "journald: no drop-in from apply; left as it is"; return 0; }
    rm -f "$JDROP" && systemctl restart systemd-journald && echo "journald: drop-in removed"
}

ZP=/sys/module/zswap/parameters
apply_zswap() {
    local pct=${1:-5} comp=${2:-lz4} p
    [ -d "$ZP" ] || { echo "no zswap in this kernel ($ZP missing)" >&2; return 1; }
    [ "$(swapon --noheadings --show=NAME 2>/dev/null | grep -c .)" -gt 0 ] || echo "warning: no swap device is active; zswap does nothing without one" >&2
    for p in enabled compressor max_pool_percent; do state_save "zswap.$p" "$(one "$ZP/$p")" || return 1; done
    put "$ZP/max_pool_percent" "$pct" && put "$ZP/compressor" "$comp" && put "$ZP/enabled" Y
}
revert_zswap() {
    local p
    for p in enabled compressor max_pool_percent; do revert_file "zswap.$p" "$ZP/$p"; done
}

SETTINGS="thp mglru watermark cgroup journald zswap"
cmd_change() {  # cmd_change apply|revert SETTING [VALUE]
    local verb=$1 name=${2:-} rc
    case " $SETTINGS " in
        *" $name "*) ;;
        *) die "usage: $verb SETTING [VALUE]; SETTING is one of: $SETTINGS" ;;
    esac
    need_root
    mkdir -p "$OUT" "$STATE" || die "cannot create $OUT and $STATE"
    census > "$OUT/$name.$verb.before.tsv"
    "${verb}_$name" "${@:3}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "$verb $name failed (exit $rc); a partial change can be undone with: revert $name" >&2
        exit $rc
    fi
    sleep "${SETTLE:-5}"
    census > "$OUT/$name.$verb.after.tsv"
    echo
    echo "== census delta, $verb $name (files: $OUT/$name.$verb.{before,after}.tsv)"
    delta "$OUT/$name.$verb.before.tsv" "$OUT/$name.$verb.after.tsv"
}

# ------------------------------------------------------------------------------------------------------------ boot-howto

cmd_boot_howto() {
    local c pool cfg kho cma cmaf
    c=$(census)
    get() { printf '%s\n' "$c" | awk -F'\t' -v k="$1" '$1 == k { print $2; exit }'; }
    pool=$(get swiotlb.pool_MiB)
    cfg=$(get kho.config)
    kho=$(get cmdline.kho)
    cma=$(get pageblocks.CMA_MiB)
    cmaf=$(get mem.CmaFree_kB)
    local extra="kho=off"
    if [ "$pool" = 64 ]; then extra="kho=off swiotlb=4096"; fi
    cat << EOF
Host RAM, the boot-time settings. This script never edits GRUB and never reboots: these steps are yours.

What this machine shows now
  kernel                       $(get meta.kernel)
  kernel command line          $(get cmdline)
  kho= on the command line     $kho
  KHO and CMA in the kernel    $cfg
  CMA-marked pageblocks        ${cma} MiB, CmaTotal $(get mem.CmaTotal_kB) kB, CmaFree $cmaf kB
  KHO scratch (debugfs)        $(get kho.scratch_MiB) MiB (equal to the CMA-marked memory when KHO is the cause)
  swiotlb pool (kernel log)    $pool MiB
EOF
    case $(get kho.in_kernel) in
        no) echo; echo "STOP: this kernel has no KHO (no CONFIG_KEXEC_HANDOVER=y), so kho=off changes nothing here and the CMA-marked memory has another cause (Q560)." ;;
        NA) echo; echo "Note: the kernel configuration could not be read, so KHO is not confirmed; the census's kho.scratch_len (debugfs) and cma.areas rows say more." ;;
    esac
    cat << EOF

1. Save the numbers first, on the running system (they are kept under $OUT, which survives a reboot):
     ssh "\$RIG" 'bash -s -- census before_boot' < scripts/rig_host_ram.sh

2. One boot with the new line, nothing saved (the next boot goes back to the old line)
   a. Reboot with someone at the rig's console: the disk passphrase is typed there.
   b. As the firmware logo goes by, hold Shift (BIOS boot) or press Esc repeatedly (UEFI boot) until the GRUB menu appears.
   c. With the first entry (the normal Ubuntu one) selected, press e. GRUB's editor uses a US keyboard layout.
   d. Find the line that starts with "linux". Move to its end (Ctrl-E or End) and append, after the words already there
      (quiet splash and \$vt_handoff):
         $extra
EOF
    if [ "$pool" = 64 ]; then
        echo "      (swiotlb=4096 is a pool of 8 MiB instead of 64: the log shows 64 MiB here. It saves up to 56 MiB.)"
    else
        echo "      (Not swiotlb=4096: the log does not show the default 64 MiB pool, so there is nothing to shrink.)"
    fi
    cat << EOF
   e. Press Ctrl-X (or F10) to boot with it, and type the passphrase when asked.
   f. After the boot, on the rig:
         ssh "\$RIG" 'bash -s -- census after_boot' < scripts/rig_host_ram.sh
         ssh "\$RIG" 'bash -s -- delta before_boot after_boot' < scripts/rig_host_ram.sh
      Expected: cmdline.kho = off; mem.CmaFree_kB 0 and pageblocks.CMA 0; mem.MemTotal_kB and mem.MemAvailable_kB about the
      same (MemAvailable already counted the free CMA pages); mem.usable_kB (MemAvailable minus CmaFree) higher by est. 410 to
      970 MiB, the memory pinned buffers can now use; with swiotlb=4096, a swiotlb pool of 8 MiB in the log and est. up to
      56 MiB more MemTotal.

3. The permanent version, only after step 2 showed the gain (still by hand, as root, with the console at hand for the next boot)
   a. Look for another file that sets the same variable, since a file in grub.d overrides /etc/default/grub:
         grep -r GRUB_CMDLINE_LINUX /etc/default/grub /etc/default/grub.d/
   b. In /etc/default/grub change the line to keep the words already in it and add yours:
         GRUB_CMDLINE_LINUX_DEFAULT="quiet splash $extra"
   c. Write the menu: update-grub
   d. Reboot once more (passphrase again) and check: grep -o 'kho=[a-z0-9]*' /proc/cmdline
   Undo: take the words out again, update-grub, reboot.
EOF
}

# ---------------------------------------------------------------------------------------------------------------- contexts

health_ok() { curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; }
ctx_stop() {
    local p i
    for p in $(llama_pids); do kill "$p" 2> /dev/null; done
    for i in $(seq 30); do [ -z "$(llama_pids)" ] && break; sleep 1; done
    for p in $(llama_pids); do kill -9 "$p" 2> /dev/null; done
    sleep "${SETTLE_AFTER:-5}"
}

# The columns of contexts.tsv: the host's change over the load (after minus before, "more" or "less"), then the server's own
# figures, then the environment the server really had.
CTX_HOST="mem.MemAvailable_kB:less mem.MemFree_kB:less mem.usable_kB:less mem.Shmem_kB:more mem.AnonPages_kB:more mem.Slab_kB:more mem.SUnreclaim_kB:more mem.VmallocUsed_kB:more mem.PageTables_kB:more mem.KernelStack_kB:more mem.Percpu_kB:more mem.Cached_kB:more mem.swap_used_kB:more"
CTX_PROC="llama0.VmRSS_kB llama0.RssAnon_kB llama0.RssShmem_kB llama0.VmLck_kB llama0.VmPin_kB llama0.smaps_Pss_kB llama0.smaps_Pss_Anon_kB llama0.smaps_Pss_File_kB llama0.smaps_Pss_Shmem_kB"
CTX_ENV="CUDA_SCALE_LAUNCH_QUEUES CUDA_MODULE_LOADING CUDA_DEVICE_MAX_CONNECTIONS GGML_CUDA_NO_PINNED"

# ctx_table header | ctx_table ARM CARDS BEFORE AFTER: one row of contexts.tsv.
ctx_table() {
    if [ "$1" = header ]; then
        awk -v hl="$CTX_HOST" -v pl="$CTX_PROC" -v el="$CTX_ENV" 'BEGIN {
            printf "arm\tcards"
            n = split(hl, h, " "); for (i = 1; i <= n; i++) { split(h[i], a, ":"); k = a[1]; sub(/^mem\./, "", k); printf "\t%s_%s", a[2], k }
            n = split(pl, p, " "); for (i = 1; i <= n; i++) { k = p[i]; sub(/^llama0\./, "", k); printf "\t%s", k }
            n = split(el, e, " "); for (i = 1; i <= n; i++) printf "\tenv_%s", e[i]
            printf "\n"
        }'
        return
    fi
    awk -F'\t' -v arm="$1" -v cards="$2" -v hl="$CTX_HOST" -v pl="$CTX_PROC" -v el="$CTX_ENV" '
        FNR == NR { b[$1] = $2; next }
        { a[$1] = $2 }
        END {
            printf "%s\t%s", arm, cards
            n = split(hl, h, " ")
            for (i = 1; i <= n; i++) {
                split(h[i], x, ":"); k = x[1]
                if ((k in a) && (k in b) && a[k] ~ /^-?[0-9]+$/ && b[k] ~ /^-?[0-9]+$/) printf "\t%.0f", (x[2] == "less") ? b[k] - a[k] : a[k] - b[k]
                else printf "\tNA"
            }
            n = split(pl, p, " "); for (i = 1; i <= n; i++) printf "\t%s", (p[i] in a) ? a[p[i]] : "NA"
            n = split(el, e, " "); for (i = 1; i <= n; i++) printf "\t%s", (("llama0.env." e[i]) in a) ? a["llama0.env." e[i]] : "NA"
            printf "\n"
        }' "$3" "$4"
}

# ctx_slope < contexts.tsv: the least-squares line of each host and server column against the card count, per arm.
ctx_slope() {
    local nm
    nm=$(( $(printf '%s\n' $CTX_HOST | wc -l) + $(printf '%s\n' $CTX_PROC | wc -l) ))
    awk -F'\t' -v nm="$nm" '
        NR == 1 { for (j = 3; j <= NF; j++) h[j] = $j; next }
        {
            arm = $1; x = $2
            if (!(arm in seen)) { seen[arm] = 1; order[++na] = arm }
            for (j = 3; j < 3 + nm; j++) if ($j ~ /^-?[0-9]+$/) { k = arm SUBSEP j; n[k]++; sx[k] += x; sy[k] += $j; sxx[k] += x * x; sxy[k] += x * $j }
        }
        END {
            print "arm\tcolumn\tMiB_per_card\tMiB_offset\tpoints"
            for (i = 1; i <= na; i++) for (j = 3; j < 3 + nm; j++) {
                k = order[i] SUBSEP j
                if (!(k in n) || n[k] < 2) continue
                den = n[k] * sxx[k] - sx[k] * sx[k]
                if (den == 0) continue
                s = (n[k] * sxy[k] - sx[k] * sy[k]) / den
                o = (sy[k] - s * sx[k]) / n[k]
                printf "%s\t%s\t%.1f\t%.1f\t%d\n", order[i], h[j], s / 1024, o / 1024, n[k]
            }
        }'
}

# ctx_run ARM CARDS: one idle server, its census before and after the load.
ctx_run() {
    local arm=$1 n=$2 tag t0 seen envset=""
    tag=$(printf '%s_%s' "$arm" "$n" | tr -c 'A-Za-z0-9._\n' '_')
    [ "$arm" = unset ] || envset=$arm
    census > "$dir/$tag.before.tsv"
    echo "== $(date +%T) arm $arm, $n card(s)"
    # shellcheck disable=SC2086
    env -u CUDA_SCALE_LAUNCH_QUEUES -u CUDA_MODULE_LOADING -u CUDA_DEVICE_MAX_CONNECTIONS -u GGML_CUDA_NO_PINNED \
        ${envset:+"$envset"} CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n - 1)))" \
        setsid nohup "$BIN/llama-server" $args -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        > "$dir/$tag.log" 2>&1 < /dev/null &
    t0=$(date +%s)
    seen=0  # env, setsid and nohup exec into llama-server, so the name appears a moment after the start
    until health_ok; do
        if [ -n "$(llama_pids)" ]; then
            seen=1
        elif [ "$seen" = 1 ] || [ $(($(date +%s) - t0)) -gt 15 ]; then
            echo "$tag: the server exited while loading:" >&2; tail -n 5 "$dir/$tag.log" >&2; return 1
        fi
        if [ $(($(date +%s) - t0)) -gt "${LOAD_TIMEOUT:-600}" ]; then echo "$tag: no health after ${LOAD_TIMEOUT:-600} s" >&2; ctx_stop; return 1; fi
        sleep 2
    done
    echo "   up after $(($(date +%s) - t0)) s; waiting ${SETTLE:-10} s"
    sleep "${SETTLE:-10}"
    census > "$dir/$tag.after.tsv"
    ctx_stop
    ctx_table "$arm" "$n" "$dir/$tag.before.tsv" "$dir/$tag.after.tsv" >> "$dir/contexts.tsv"
}

cmd_contexts() {
    : "${BIN:?set BIN to the bin/ directory of the llama.cpp build}" "${MODEL:?set MODEL to a small model file that fits one card}"
    need_root
    [ -x "$BIN/llama-server" ] || die "no executable $BIN/llama-server"
    [ -r "$MODEL" ] || die "cannot read $MODEL"
    command -v curl > /dev/null || die "curl is needed for the health check"
    local counts=${COUNTS:-"1 3 5 9"} arms=${ARMS:-"unset CUDA_SCALE_LAUNCH_QUEUES=0.5x CUDA_SCALE_LAUNCH_QUEUES=0.25x"} ngpu arm n failed=""
    PORT=${PORT:-18080}
    args=${SERVER_ARGS:-"-ngl 99 -c 2048 -np 1 --cache-ram 0"}
    dir=$OUT/contexts
    for arm in $arms; do
        case $arm in
            CUDA_SCALE_LAUNCH_QUEUES=4x*) die "arm $arm refused: 4x costs about 120 MB of host RAM per card (llama.cpp PR 19042)" ;;
            unset|[A-Z]*_*=*) ;;
            *) die "arm '$arm' is neither 'unset' nor VAR=value" ;;
        esac
    done
    if [ -n "$(llama_pids)" ]; then die "a $SERVER_NAME is running; stop it first (the host must be idle)"; fi
    ngpu=$(ls /proc/driver/nvidia/gpus 2>/dev/null | grep -c .)
    [ "$ngpu" -gt 0 ] || die "no NVIDIA GPU under /proc/driver/nvidia/gpus"
    mkdir -p "$dir" || die "cannot create $dir"
    ctx_table header > "$dir/contexts.tsv"
    trap ctx_stop EXIT
    for arm in $arms; do
        for n in $counts; do
            if [ "$n" -gt "$ngpu" ]; then echo "skip $n cards: only $ngpu here"; continue; fi
            ctx_run "$arm" "$n" || failed="$failed $arm/$n"
        done
    done
    trap - EXIT
    ctx_slope < "$dir/contexts.tsv" > "$dir/slope.tsv"
    echo; echo "== $dir/contexts.tsv (kB; host columns are the change over the load, server columns its own figures)"
    cat "$dir/contexts.tsv"
    echo; echo "== $dir/slope.tsv (the per-card slope is the host cost of one more CUDA context; the offset is the process's)"
    cat "$dir/slope.tsv"
    [ -z "$failed" ] || { echo "failed runs:$failed" >&2; return 1; }
}

# ------------------------------------------------------------------------------------------------------------------ main

main() {
    local mode=${1:-}
    [ $# -gt 0 ] && shift
    case $mode in
        census) cmd_census "$@" ;;
        delta) cmd_delta "$@" ;;
        apply|revert) cmd_change "$mode" "$@" ;;
        boot-howto) cmd_boot_howto ;;
        contexts) cmd_contexts ;;
        *) echo "usage: rig_host_ram.sh census [NAME] | delta BEFORE AFTER | apply SETTING [VALUE] | revert SETTING | boot-howto | contexts" >&2
           echo "SETTING: $SETTINGS" >&2
           return 2 ;;
    esac
}

main "$@"
exit $?
