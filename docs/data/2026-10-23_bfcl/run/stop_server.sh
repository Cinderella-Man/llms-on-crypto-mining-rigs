#!/bin/bash
# Q30 BFCL v3 check (2026-10-23): keep the server log in OUT/logs (the harness folder), then stop and remove the
# container.
# stop_server.sh NAME
docker logs "$1" > "${OUT:?}/logs/server-$1-$(date +%H%M).log" 2>&1 || true
docker stop "$1" >/dev/null && docker rm "$1" >/dev/null && echo "stopped $1"
