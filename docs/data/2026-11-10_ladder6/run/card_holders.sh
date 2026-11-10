#!/usr/bin/env bash
# C662 (Q1150): who else holds one card of the build machine, by its PCI bus id. Prints one line per foreign holder: a running
# container that was given the card (its device request names the card's UUID or index, or every card) and does not
# carry the label LABEL=1 (default c662), or a compute process on the card that is not inside such a container. Nothing printed = the
# card is free for this job. Read-only.
# Usage (on the machine): [LABEL=<label>] bash card_holders.sh <PCI bus id>   (also prints "uuid index" of the card on stderr)
set -uo pipefail
BUS=$(echo "${1:?PCI bus id, e.g. 0b:00.0}" | tr 'a-z' 'A-Z')
read -r UUID IDX < <(nvidia-smi --query-gpu=pci.bus_id,uuid,index --format=csv,noheader | awk -F', ' -v b="$BUS" 'toupper($1) ~ ":" b "$" {print $2, $3}')
[ -n "${UUID:-}" ] || { echo "no card at $BUS"; exit 2; }
echo "$UUID $IDX" >&2
LABEL=${LABEL:-c662}
mine() { [ "$(docker inspect --format "{{index .Config.Labels \"$LABEL\"}}" "$1" 2>/dev/null)" = 1 ]; }
for c in $(docker ps -q); do
    mine "$c" && continue
    req=$(docker inspect --format '{{json .HostConfig.DeviceRequests}}' "$c")
    rt=$(docker inspect --format '{{.HostConfig.Runtime}}' "$c")
    vis=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$c" | grep '^NVIDIA_VISIBLE_DEVICES=' | cut -d= -f2)
    held=0
    echo "$req" | grep -q -E "$UUID|\"Count\":-1|\"DeviceIDs\":\[[^]]*\"$IDX\"" && held=1
    [ "$rt" = nvidia ] && echo ",$vis," | grep -q -E ",(all|$UUID|$IDX)," && held=1
    [ $held = 1 ] && echo "container $(docker inspect --format '{{.Name}}' "$c")"
done
for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1 == u {print $2}'); do
    cid=$(grep -o -E '[0-9a-f]{64}' "/proc/$p/cgroup" 2>/dev/null | head -n 1)
    [ -n "$cid" ] && mine "$cid" && continue
    echo "process $p ($(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-60))"
done
exit 0
