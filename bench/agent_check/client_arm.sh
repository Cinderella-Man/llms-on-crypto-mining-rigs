#!/usr/bin/env bash
# client_arm.sh: one arm of release_check.sh on the client machine (the one with Docker). The logging proxy runs in a
# container; run.py starts one container per attempt (pi, then the tests), all labelled and at a low CPU share, and
# itself runs under nice. release_check.sh starts this detached over ssh and polls OUT/ARM.rc, which holds run.py's
# exit code at the end (OUT/ARM.pid while it runs). Environment:
#   OUT            results folder (res/, proxy.jsonl, work/ during the run)
#   ARM            the arm's name                   UPSTREAM_PORT  the server as seen on this machine's loopback
#   PROXY_PORT     the proxy on this machine's loopback (pi's base URL)
#   TASK_SET (bugfix), TASKS (all), TIME_CAP_MIN (15), TURN_CAP (40), BUDGET_MIN (none: no attempt starts after it),
#   CTX (65536, pi's context window = the server's -c), CPU_SHARES (256), IMAGE (agent-check:0.87.1),
#   LABEL (pascal-quants=1)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${OUT:?}" "${ARM:?}" "${UPSTREAM_PORT:?}" "${PROXY_PORT:?}"
TASK_SET=${TASK_SET:-bugfix}; TASKS=${TASKS:-all}; TIME_CAP_MIN=${TIME_CAP_MIN:-15}; TURN_CAP=${TURN_CAP:-40}
CTX=${CTX:-65536}; CPU_SHARES=${CPU_SHARES:-256}; IMAGE=${IMAGE:-agent-check:0.87.1}
LABEL=${LABEL:-pascal-quants=1}
PFX=acrel-$ARM
mkdir -p "$OUT"
rm -f "$OUT/$ARM.rc"
echo $$ > "$OUT/$ARM.pid"
finish() { echo "${1:-1}" > "$OUT/$ARM.rc"; }
cleanup() {
  docker logs "$PFX-proxy" > "$OUT/proxy-$ARM.out" 2>&1
  docker ps -aq --filter "name=^$PFX-" | xargs -r docker rm -f > /dev/null 2>&1
  rm -f "$OUT/$ARM.pid"
  [ -f "$OUT/$ARM.rc" ] || finish 1
}
trap cleanup EXIT
trap 'echo "stopped"; exit 143' TERM INT HUP

docker rm -f "$PFX-proxy" > /dev/null 2>&1
docker run -d --name "$PFX-proxy" --label "$LABEL" --cpu-shares "$CPU_SHARES" --network host \
  -v "$HERE:$HERE:ro" -v "$OUT:$OUT" "$IMAGE" python3 -I "$HERE/proxy.py" --listen "127.0.0.1:$PROXY_PORT" \
  --upstream "127.0.0.1:$UPSTREAM_PORT" --log "$OUT/proxy.jsonl" > /dev/null \
  || { echo "proxy container did not start"; exit 1; }
sleep 2
[ "$(docker inspect -f '{{.State.Running}}' "$PFX-proxy" 2> /dev/null)" = true ] \
  || { echo "proxy exited:"; docker logs "$PFX-proxy" 2>&1 | tail -5; exit 1; }
echo "$(date +%H:%M:%S) $ARM: proxy 127.0.0.1:$PROXY_PORT -> 127.0.0.1:$UPSTREAM_PORT; tasks $TASK_SET/$TASKS"
DEADLINE=()
[ -n "${BUDGET_MIN:-}" ] && DEADLINE=(--start-deadline-min "$BUDGET_MIN")
nice -n 19 python3 "$HERE/run.py" --arm "$ARM" --base-url "http://127.0.0.1:$PROXY_PORT/v1" --out "$OUT/res" \
  --task-set "$TASK_SET" --tasks "$TASKS" --time-cap-min "$TIME_CAP_MIN" --turn-cap "$TURN_CAP" --context "$CTX" \
  --health-wait-min 5 --work-root "$OUT/work" "${DEADLINE[@]}" --sandbox docker --image "$IMAGE" \
  --docker-label "$LABEL" --docker-prefix "$PFX" --docker-cpu-shares "$CPU_SHARES"
rc=$?
rm -rf "$OUT/work/$ARM"
finish $rc
exit $rc
