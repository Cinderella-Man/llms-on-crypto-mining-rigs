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

# wait_gpus 0,1,2: returns once no other session's container holds any of these GPUs and each has less than FREE_MIB
# (default 1000) MiB in use; logs what it waits for every ten minutes.
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
            used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
            [ "$used" -lt "${FREE_MIB:-1000}" ] || hit="$hit gpu$g:${used}MiB"
        done
        if [ -z "$hit" ]; then log "GPUs ${1} free"; return 0; fi
        [ $((n % 10)) -eq 0 ] && log "waiting for GPUs ${1}:$hit"
        n=$((n + 1))
        sleep 60
    done
}
