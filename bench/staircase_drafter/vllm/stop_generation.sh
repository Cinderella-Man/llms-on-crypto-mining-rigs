#!/usr/bin/env bash
# Stop run_generation.sh cleanly: no new requests, wait for the running answers (at most WAIT seconds, default 300),
# then stop the client and the server. Answers still running at the deadline are dropped; their prompts stay claimed.
# usage: DATA=/data/q20 OUT=gen-V1 bash stop_generation.sh
set -uo pipefail
: "${DATA:?}" "${OUT:?}"
CLIENT=${CLIENT:-pq-vllm-cli}
NAME=${NAME:-pq-vllm-gen}
touch "$DATA/STOP-$OUT"
for _ in $(seq 1 $(( ${WAIT:-300} / 5 ))); do
    docker ps -q -f "name=^$CLIENT\$" | grep -q . || break
    sleep 5
done
docker rm -f "$CLIENT" >/dev/null 2>&1
docker stop -t 60 "$NAME" >/dev/null 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "$(date -u +%FT%TZ) stopped $CLIENT and $NAME"
