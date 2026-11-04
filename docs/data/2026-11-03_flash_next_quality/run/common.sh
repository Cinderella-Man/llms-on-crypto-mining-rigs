# Helpers shared by the chain's steps; sourced by chain.sh and by each step, after the chain's environment file.
# Needs TAG (the containers are named pq-$TAG-*), MOUNTS (folders mounted at the same path in every container) and IMAGE.

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MOUNT_ARGS=""
for _d in $MOUNTS; do MOUNT_ARGS="$MOUNT_ARGS -v $_d:$_d"; done
# every container: the shared label, a low CPU share
DOCKER_BASE="--label pascal-quants=1 --cpu-shares 256"

# The GPUs other sessions' containers hold: one line "<name>:<ids>" per running pq-* or acrel-* container that is not
# ours, from its device request ("all" for --gpus all).
busy_gpus() {
    local c ids
    for c in $(docker ps --format '{{.Names}}'); do
        case $c in pq-"$TAG"-*) continue ;; pq-*|acrel-*) ;; *) continue ;; esac
        ids=$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{if eq .Count -1}}all{{end}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' "$c" 2> /dev/null || true)
        [ -n "${ids// /}" ] && echo "$c:$ids"
    done
}

# preflight_gpus 0,1,2: CUDA works on these GPUs, checked in $IMAGE with this chain's Docker arguments by the script at
# $PREFLIGHT (scripts/cuda_preflight.sh of this repository; set it, and PREFLIGHT_CACHE for the compiled probe, in the
# chain's environment file). Logs the verdict line and returns non-zero when CUDA does not work there, so a driver that
# needs a reboot stops the step in seconds with the card and the cause named. Nothing is checked when PREFLIGHT is unset.
preflight_gpus() {
    local out rc=0
    [ -n "${1:-}" ] || return 0
    if [ -z "${PREFLIGHT:-}" ]; then log "CUDA pre-flight skipped: PREFLIGHT is not set"; return 0; fi
    out=$(IMAGE=$IMAGE PREFLIGHT_DOCKER_ARGS="$DOCKER_BASE" bash "$PREFLIGHT" -q -g "$1" 2>&1) || rc=$?
    log "CUDA pre-flight on GPUs $1: $(printf '%s\n' "$out" | tail -n 1)"
    return $rc
}

# wait_gpus 0,1,2: returns once no other session's container holds any of these GPUs and each has less than FREE_MIB
# (default 1000) MiB in use, then runs preflight_gpus on them; logs what it waits for every ten minutes. Returns 1 when
# a card cannot be read or the pre-flight fails.
wait_gpus() {
    local want=${1//,/ } n=0 hit line ids g d used
    while :; do
        hit=""
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            ids=${line#*:}
            for d in ${ids//,/ }; do
                for g in $want; do [ "$d" = all ] || [ "$d" = "$g" ] && hit="$hit ${line%%:*}(gpu $d)"; done
            done
        done < <(busy_gpus)
        for g in $want; do
            used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr -d ' ') || used=""
            case $used in
                ''|*[!0-9]*) log "GPU $g: nvidia-smi cannot read its memory use"; preflight_gpus "$1" || true; return 1 ;;
            esac
            [ "$used" -lt "${FREE_MIB:-1000}" ] || hit="$hit gpu$g:${used}MiB"
        done
        if [ -z "$hit" ]; then log "GPUs ${1} free"; preflight_gpus "$1" || return 1; return 0; fi
        [ $((n % 10)) -eq 0 ] && log "waiting for GPUs ${1}:$hit"
        n=$((n + 1))
        sleep 60
    done
}
