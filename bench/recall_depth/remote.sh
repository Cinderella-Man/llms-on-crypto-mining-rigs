#!/usr/bin/env bash
# remote.sh put|start|wait|stop|collect: serve.sh on a rig over ssh, from the machine that runs recall_depth.py.
#   put      copies serve.sh, the preset (PRESET) and scripts/ram_watchdog.sh to WORK on the rig
#   start    serve.sh start there (the idle check, then the server and its watchdog, detached)
#   wait     serve.sh wait: returns when the server answers
#   stop     serve.sh stop: the server and watchdog stopped, the idle check after
#   collect  WORK's files to OUT here, then WORK removed on the rig (CLEAN=0 keeps it)
# The client reaches the server through an ssh tunnel, as the server listens on the rig's loopback only:
#   ssh -N -L 18080:127.0.0.1:8080 "$RIG" &   then URL=http://127.0.0.1:18080 python3 recall_depth.py run ...
# Env: RIG (the rig's ssh alias, non-interactive), WORK (on the rig, default /tmp/recall-depth), PRESET (here, default
# release/flash-next-v4/settings.env), BIN, MODEL, DRAFT, PORT (on the rig; see serve.sh), OUT (here, for collect).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${1:?usage: remote.sh put|start|wait|stop|collect}
: "${RIG:?RIG=<ssh alias of the rig>}"
WORK=${WORK:-/tmp/recall-depth}
PORT=${PORT:-8080}
rig() { ssh "$RIG" "$@"; }

case $STEP in
put)
  PRESET=${PRESET:-$HERE/../../release/flash-next-v4/settings.env}
  rig "mkdir -p $WORK" || exit 1
  rig "cat > $WORK/settings.env" < "$PRESET"
  rig "cat > $WORK/ram_watchdog.sh" < "$HERE/../../scripts/ram_watchdog.sh"
  rig "cat > $WORK/serve.sh" < "$HERE/serve.sh"
  rig "ls -l $WORK && md5sum $WORK/settings.env"
  ;;
start)
  : "${BIN:?BIN=<llama-server on the rig>}" "${MODEL:?MODEL=<first shard on the rig>}"
  # an ssh call that starts a detached job may wait for it: the timeout ends the call, not the job
  timeout 30 ssh "$RIG" "WORK=$WORK PORT=$PORT BIN=$BIN MODEL=$MODEL DRAFT=${DRAFT:-} bash $WORK/serve.sh start"
  rc=$?
  [ "$rc" = 124 ] && rc=0
  exit "$rc"
  ;;
wait|stop)
  rig "WORK=$WORK PORT=$PORT bash $WORK/serve.sh $STEP"
  ;;
collect)
  : "${OUT:?OUT=<folder here>}"
  mkdir -p "$OUT"
  rig "tar -C $WORK -cf - --exclude=serve.sh --exclude=ram_watchdog.sh ." | tar -C "$OUT" -xf - || { echo "collect failed"; exit 1; }
  echo "files in $OUT"
  if [ "${CLEAN:-1}" = 1 ]; then
    rig "[ -f $WORK/serve.sh ] && rm -rf $WORK && echo 'removed $WORK on the rig'"
  fi
  ;;
*)
  echo "usage: remote.sh put|start|wait|stop|collect"; exit 1
  ;;
esac
