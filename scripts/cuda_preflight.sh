#!/usr/bin/env bash
# cuda_preflight.sh: before a long GPU job, check in seconds that CUDA really works on the GPUs the job will use, and
# if it does not, say which card and why, in one line. A driver that needs a reboot still lets `docker run` start a
# container and lets `nvidia-smi` answer while every CUDA call fails; the job then starts and crashes minutes or hours
# in. This script runs a small probe (cuda_preflight/cuprobe.c) in the job's own image with the job's own GPU request:
# cuInit, then for every visible device a context, an allocation and a copy to the card and back. On the host it also
# reads the kernel log of this boot for Xid 79 ("fallen off the bus"), Xid 154 (a reboot or reset is required) and UVM
# fatal errors, which name the card and say what clears it.
#
#   cuda_preflight.sh [-i IMAGE] [-q] -g 0,1,2            the cards by NVML index (docker's --gpus "device=0,1,2")
#   cuda_preflight.sh [-i IMAGE] [-q] -g all              every card (--gpus all)
#   cuda_preflight.sh [-i IMAGE] [-q] -- <docker GPU request>   any form docker takes, for example
#       -- --device nvidia.com/gpu=0 --device nvidia.com/gpu=2                       (CDI)
#       -- --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=GPU-<uuid>,GPU-<uuid>          (the runtime's own CDI mode)
#
# Output: one line per device, then a last line that is the verdict, "PREFLIGHT OK: ..." or "PREFLIGHT FAIL: ...".
# With -q only that last line. Exit status: 0 CUDA works on every requested card, 1 it does not (or the container
# cannot start, or the probe hangs), 2 the check itself cannot run (usage, no docker, the probe will not compile).
#
# Environment (all optional):
#   IMAGE                      the job's image (default nvidia/cuda:12.9.2-devel-ubuntu24.04); the probe runs in it
#   PREFLIGHT_BUILD_IMAGE      image to compile the probe in when IMAGE has no gcc (default IMAGE)
#   PREFLIGHT_CACHE            where the compiled probe is kept (default ~/.cache/cuda_preflight); the first run
#                              compiles it in a container, later runs reuse it, so nothing is installed on the host
#   PREFLIGHT_DOCKER_ARGS      extra arguments for every docker run, for example "--label owner=me --cpu-shares 256"
#   PREFLIGHT_TIMEOUT          seconds the probe may take (default 60); a hung driver is reported, not waited for
#   PREFLIGHT_KERNEL_LOG       read the kernel log from this file instead of the host's (journalctl -k, then dmesg)
#   PREFLIGHT_KERNEL_STRICT=1  fail on kernel-log events even when the probe passes (default: only a warning)
#   PREFLIGHT_SKIP=1           do nothing and succeed (an override for a person who knows better)
#   PROBE_MIB                  size of the probe's allocation per card, MiB (default 16)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/cuda_preflight/cuprobe.c
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CACHE=${PREFLIGHT_CACHE:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/cuda_preflight}
TIMEOUT=${PREFLIGHT_TIMEOUT:-60}
read -r -a EXTRA <<< "${PREFLIGHT_DOCKER_ARGS:-}"
GPU_ARGS=()
REQ=""
QUIET=0

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit "${1:-2}"; }

while [ $# -gt 0 ]; do
    case $1 in
        -g) [ $# -ge 2 ] || usage
            if [ "$2" = all ]; then GPU_ARGS+=(--gpus all); REQ="$REQ --gpus all"
            else GPU_ARGS+=(--gpus "\"device=$2\""); REQ="$REQ --gpus device=$2"; fi
            shift 2 ;;
        -i) [ $# -ge 2 ] || usage; IMAGE=$2; shift 2 ;;
        -q) QUIET=1; shift ;;
        -h|--help) usage 0 ;;
        --) shift; GPU_ARGS+=("$@"); REQ="$REQ $*"; break ;;
        *) usage ;;
    esac
done
[ ${#GPU_ARGS[@]} -gt 0 ] || usage
REQ=${REQ# }
BUILD_IMAGE=${PREFLIGHT_BUILD_IMAGE:-$IMAGE}

say() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }
cannot() { printf 'PREFLIGHT CANNOT RUN: %s\n' "$*"; exit 2; }

if [ "${PREFLIGHT_SKIP:-0}" = 1 ]; then echo "PREFLIGHT SKIPPED: PREFLIGHT_SKIP=1"; exit 0; fi
command -v docker > /dev/null 2>&1 || cannot "docker is not installed here"
[ -f "$SRC" ] || cannot "$SRC is missing"

# ---- the probe: compiled once per source version and image, in a container without GPUs, and kept in $CACHE ----
build_probe() {
    local key img tmp
    key=$(sha1sum "$SRC" | cut -c1-10)
    img=${BUILD_IMAGE//[^A-Za-z0-9._-]/_}
    BIN=$CACHE/cuprobe-$key-$img
    [ -x "$BIN" ] && return 0
    mkdir -p "$CACHE" 2> /dev/null || cannot "cannot create $CACHE"
    tmp=$BIN.tmp$$
    if ! docker run --rm ${EXTRA[@]+"${EXTRA[@]}"} -e NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" \
            -v "$HERE/cuda_preflight:$HERE/cuda_preflight:ro" -v "$CACHE:$CACHE" "$BUILD_IMAGE" \
            gcc -O1 -o "$tmp" "$SRC" -ldl > "$tmp.log" 2>&1; then
        local why; why=$(grep -iE 'error|not found|unable' "$tmp.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)
        rm -f "$tmp" "$tmp.log"
        cannot "the probe will not compile in $BUILD_IMAGE (needs gcc; set PREFLIGHT_BUILD_IMAGE to an image that has it): $why"
    fi
    mv "$tmp" "$BIN" && rm -f "$tmp.log"
}

# ---- the host's view: nvidia-smi's index of each bus id, and what the kernel log of this boot says ----
declare -A HOSTIDX=()
SMI_OK=0
busKey() { local b=${1,,}; b=${b%.*}; printf '%s' "${b: -5}"; }   # 0000:09:00.0, 00000000:09:00.0 and 0000:09:00 -> 09:00

load_hostmap() {
    local idx bus
    command -v nvidia-smi > /dev/null 2>&1 || return 0
    while IFS=', ' read -r idx bus _; do
        case $idx in ''|*[!0-9]*) continue ;; esac
        HOSTIDX[$(busKey "$bus")]=$idx; SMI_OK=1
    done < <(timeout 15 nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2> /dev/null)
}

gpu_label() {
    local k; k=$(busKey "$1")
    if [ -n "${HOSTIDX[$k]:-}" ]; then echo "host GPU ${HOSTIDX[$k]}"
    elif [ "$SMI_OK" = 1 ]; then echo "missing from nvidia-smi, the lost card"
    else echo "host GPU index unknown"; fi
}

# put ", host GPU <n>" after every "bus <id>" in the lines it reads
map_hosts() {
    local k args=()
    for k in "${!HOSTIDX[@]}"; do args+=(-e "s/bus [0-9a-fA-F]{4}:${k}\\.[0-9]/&, host GPU ${HOSTIDX[$k]}/I"); done
    if [ ${#args[@]} -gt 0 ]; then sed -E "${args[@]}"; else cat; fi
}

klog() {
    local t
    if [ -n "${PREFLIGHT_KERNEL_LOG:-}" ]; then cat "$PREFLIGHT_KERNEL_LOG" 2> /dev/null; return 0; fi
    if command -v journalctl > /dev/null 2>&1; then
        t=$(journalctl -k -b --no-pager -o short-iso 2> /dev/null)
        if [ -n "$t" ]; then printf '%s\n' "$t"; return 0; fi
    fi
    if command -v dmesg > /dev/null 2>&1; then dmesg -T 2> /dev/null; fi
    return 0
}

KAWK='
function when() { return match($0, /[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/) ? substr($0, RSTART, RLENGTH) : "?" }
/NVRM: Xid \(PCI:[0-9A-Fa-f:]+\): (79|154),/ {
    match($0, /Xid \(PCI:[0-9A-Fa-f:]+\)/)
    bus = tolower(substr($0, RSTART + 9, RLENGTH - 10))
    if ($0 ~ /\): 79,/) { if (!(bus in lost)) { lost[bus] = when(); order[++nl] = bus } }
    else if ($0 ~ /Node Reboot Required/) {
        if (!(bus in reb)) { reb[bus] = 1; nreb++; if (!treb) treb = when() }
        if (bus in rst) { delete rst[bus]; nrst-- }      # a reset that became a reboot counts once
    }
    else if ($0 ~ /GPU Reset Required/) { if (!(bus in rst) && !(bus in reb)) { rst[bus] = 1; nrst++; if (!trst) trst = when() } }
}
/uvm encountered global fatal error/ { if (!uvm) uvm = when() }
END {
    for (i = 1; i <= nl; i++) print "lost", order[i], lost[order[i]]
    if (nreb) print "reboot", nreb, treb
    if (nrst) print "reset", nrst, trst
    if (uvm) print "uvm", uvm
}'

# kernel_text: sets KTEXT to what the log of this boot says, in one sentence ("" when it says nothing), from the lines
# the awk script printed (KRAW); KFATAL=1 when it says a reboot or a reset is needed
kernel_text() {
    local kind a b p sep=""
    KTEXT=""; KFATAL=0
    while read -r kind a b; do
        case $kind in
            lost)   p="Xid 79 (fell off the bus) on PCI $a [$(gpu_label "$a")] at $b" ;;
            reboot) p="Xid 154 (Node Reboot Required) on $a GPU(s) at $b" ;;
            reset)  p="Xid 154 (GPU Reset Required) on $a GPU(s) at $b" ;;
            uvm)    p="a UVM global fatal error at $a" ;;
            *) continue ;;
        esac
        KTEXT="$KTEXT$sep$p"; sep="; "; KFATAL=1
    done <<< "$KRAW"
}

# ---- run the probe ----
build_probe
START=$SECONDS
CNAME=cuda-preflight-$$
# on a timeout the docker client is stopped (the second signal, KILL, comes 3 s later: the client traps TERM) and the
# container is killed by name; its init would ignore a forwarded signal, so --sig-proxy=false
out=$(timeout -k 3 "$TIMEOUT" docker run --rm --sig-proxy=false --name "$CNAME" ${EXTRA[@]+"${EXTRA[@]}"} "${GPU_ARGS[@]}" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -v "$CACHE:$CACHE:ro" "$IMAGE" "$BIN" 2>&1)
rc=$?
hung=0
if [ $rc -eq 124 ] || { [ $rc -eq 137 ] && [ $((SECONDS - START)) -ge "$TIMEOUT" ]; }; then
    hung=1; docker kill "$CNAME" > /dev/null 2>&1; docker rm -f "$CNAME" > /dev/null 2>&1
fi

verdict=$(printf '%s\n' "$out" | grep -E '^PREFLIGHT (OK|FAIL):' | tail -n 1)
details=$(printf '%s\n' "$out" | grep -E '^cuda [0-9]+ bus ' || true)
if [ $hung = 1 ]; then
    verdict="PREFLIGHT FAIL: the probe did not finish within $TIMEOUT s on [$REQ]; likely cause: a CUDA call hangs, which a driver stuck after a GPU error does"
elif [ -z "$verdict" ]; then
    err=$(printf '%s\n' "$out" | grep -iE 'nvidia-container-cli|nvml' | head -n 1 | cut -c1-300)
    [ -n "$err" ] || err=$(printf '%s\n' "$out" | grep -iE 'rror' | head -n 1 | cut -c1-300)
    if printf '%s' "$err" | grep -qiE 'nvml|nvidia-container-cli|detection error'; then
        cause="the NVIDIA container runtime cannot enumerate the GPUs, which is what a card lost from the bus does to every 'docker run --gpus'"
    else
        cause="docker refused the GPU request or the image; the message above says which"
    fi
    verdict="PREFLIGHT FAIL: docker could not start the probe container with [$REQ] (exit $rc: ${err:-no message}); likely cause: $cause"
elif [ $rc -ne 0 ] && [ "${verdict#PREFLIGHT OK}" != "$verdict" ]; then
    verdict="PREFLIGHT FAIL: the probe exited $rc after reporting OK"
fi
ok=0; [ $rc -eq 0 ] && [ "${verdict#PREFLIGHT OK}" != "$verdict" ] && ok=1

# kernel log (this boot): always read on a failure, and on success to warn
KRAW=$(klog | awk "$KAWK")
KTEXT=""; KFATAL=0
if [ "$ok" = 0 ] || [ -n "$KRAW" ]; then load_hostmap; kernel_text; fi
[ -z "$details" ] || say "$(printf '%s\n' "$details" | map_hosts)"
verdict=$(printf '%s' "$verdict" | map_hosts)

if [ "$ok" = 1 ]; then
    if [ -n "$KTEXT" ] && [ "${PREFLIGHT_KERNEL_STRICT:-0}" = 1 ]; then
        echo "PREFLIGHT FAIL: the probe passed on [$REQ] in $((SECONDS - START)) s, but the kernel log of this boot says: $KTEXT; PREFLIGHT_KERNEL_STRICT=1"
        exit 1
    fi
    msg="$verdict (in $((SECONDS - START)) s)"
    [ -n "$KTEXT" ] && msg="$msg. WARNING, the kernel log of this boot says: $KTEXT"
    echo "$msg"
    exit 0
fi
msg="$verdict"
[ -n "$KTEXT" ] && msg="$msg. Kernel log of this boot: $KTEXT"
if [ "$KFATAL" = 1 ]; then msg="$msg. This needs a reboot or power cycle by whoever runs the machine; do not start the job."; fi
echo "$msg"
exit 1
