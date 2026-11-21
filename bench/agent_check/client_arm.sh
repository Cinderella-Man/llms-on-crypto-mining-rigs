#!/usr/bin/env bash
# client_arm.sh: one arm of release_check.sh on the client machine (the one with Docker). The logging proxy runs in a
# container; run.py starts one container per attempt (pi, then the tests), all labelled and at a low CPU share, and
# itself runs under nice. release_check.sh starts this detached over ssh and polls OUT/ARM.rc, which holds run.py's
# exit code at the end (OUT/ARM.pid while it runs). Environment:
#   OUT            results folder (res/, proxy.jsonl, work/ during the run)
#   ARM            the arm's name                   UPSTREAM_PORT  the server as seen on this machine's loopback
#   PROXY_PORT     the proxy on this machine's loopback (pi's base URL)
#   UPSTREAM_PORTS (instead of UPSTREAM_PORT) several servers, space-separated: the proxy is race_proxy.py over them
#                  in that order (the first-finish race, Q1195), with RACE_ARGS as its options (e.g. --win-at
#                  reasoning); its per-sample rows go to OUT/proxy.samples.jsonl
#   TASK_SET (bugfix), TASKS (all), TIME_CAP_MIN (15), TURN_CAP (40), BUDGET_MIN (none: no attempt starts after it),
#   CTX (65536, pi's context window = the server's -c), CPU_SHARES (256), IMAGE (agent-check:0.87.1),
#   LABEL (pascal-quants=1), SAMPLING (JSON set on every request; default run.py's, Flint's makers' settings),
#   PREFIX (default: one of the run's own, acrel-<OUT's folder name>-<OUT's full path as a CRC-32 in hex>, so two runs
#          on one client never share container names; the containers are PREFIX-ARM-*, both ends remove only those,
#          and OUT/ARM.prefix holds PREFIX-ARM; a PREFIX given wins (the drivers pass one worked out the same way from
#          CLIENT_OUT), and the arm refuses to start while another run's proxy holds the same name),
#   RUN_ARGS (more run.py options, e.g. the token mark: "--gen-budget 65536 --mark-tries 2"),
#   AC_DOCKER_USER ("uid:gid": every container runs as that user; CLIENT=local sets it in release_check_s12.sh and
#                  race_rigs.sh on a
#                  client that is not root, see common.py)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${OUT:?}" "${ARM:?}" "${PROXY_PORT:?}"
read -r -a UPS <<< "${UPSTREAM_PORTS:-${UPSTREAM_PORT:?UPSTREAM_PORT or UPSTREAM_PORTS}}"
if [ -n "${UPSTREAM_PORTS:-}" ]; then
  # shellcheck disable=SC2206
  PROXY=(race_proxy.py serve $(printf -- '--upstream 127.0.0.1:%s ' "${UPS[@]}") ${RACE_ARGS:-})
else
  PROXY=(proxy.py --upstream "127.0.0.1:$UPSTREAM_PORT")
fi
TASK_SET=${TASK_SET:-bugfix}; TASKS=${TASKS:-all}; TIME_CAP_MIN=${TIME_CAP_MIN:-15}; TURN_CAP=${TURN_CAP:-40}
CTX=${CTX:-65536}; CPU_SHARES=${CPU_SHARES:-256}; IMAGE=${IMAGE:-agent-check:0.87.1}
LABEL=${LABEL:-pascal-quants=1}
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd -P)
run_prefix() {  # acrel-<OUT's folder name: at most 24 of A-Z a-z 0-9 _ ->-<CRC-32 of OUT's full path, 8 hex digits>
  local name crc
  name=$(printf %s "${OUT_ABS##*/}" | tr -c 'A-Za-z0-9_-' '-' | cut -c1-24)
  crc=$(printf %s "$OUT_ABS" | cksum | cut -d' ' -f1)
  printf 'acrel-%s-%08x\n' "$name" "$crc"
}
PFX=${PREFIX:-$(run_prefix)}-$ARM
port_busy() { timeout 3 bash -c 'exec 3<> "/dev/tcp/127.0.0.1/$0"' "$1" 2> /dev/null; }  # something listens there

# Checks before this run starts anything. Another run of this arm in OUT stops it with nothing touched. Another run's
# proxy under the same container name (one PREFIX given to two runs) or a port in use stops it with no ARM.rc, so the
# driver following it stops too ("ended without an exit code") and no other run's container is removed.
say() { echo "$(date +%H:%M:%S) $ARM: $*"; }
old=$(cat "$OUT/$ARM.pid" 2> /dev/null)
if [[ $old =~ ^[0-9]+$ ]] && [ "$old" != $$ ] && tr '\0' ' ' < "/proc/$old/cmdline" 2> /dev/null | grep -q client_arm.sh
then
  say "another client_arm.sh (pid $old) runs this arm in $OUT; give this run its own OUT"
  exit 1
fi
rm -f "$OUT/$ARM.rc"
holder=$(docker ps --filter "name=^$PFX-proxy\$" --format 'x{{.Label "ac-out"}}' 2> /dev/null)
if [ -n "$holder" ] && [ "${holder#x}" != "$OUT_ABS" ]; then
  holder=${holder#x}
  say "container $PFX-proxy runs for another run (${holder:-an older client_arm.sh}): leave PREFIX unset, or give" \
    "this run a PREFIX of its own; if that run is over, remove its containers ($PFX-*)"
  exit 1
fi
for u in "${UPS[@]}"; do
  [ "$u" != "$PROXY_PORT" ] || { say "PROXY_PORT $PROXY_PORT is also an upstream port"; exit 1; }
done
if port_busy "$PROXY_PORT"; then
  say "PROXY_PORT $PROXY_PORT is in use on this machine (another agent check's proxy or relay?); give this run its" \
    "own PROXY_PORT"
  exit 1
fi

echo $$ > "$OUT/$ARM.pid"
echo "$PFX" > "$OUT/$ARM.prefix"
finish() { echo "${1:-1}" > "$OUT/$ARM.rc"; }
cleanup() {
  docker logs "$PFX-proxy" > "$OUT/proxy-$ARM.out" 2>&1
  docker ps -aq --filter "name=^$PFX-" | xargs -r docker rm -f > /dev/null 2>&1
  rm -f "$OUT/$ARM.pid"
  [ -f "$OUT/$ARM.rc" ] || finish 1
}
trap cleanup EXIT
trap 'echo "stopped"; exit 143' TERM INT HUP

docker rm -f "$PFX-proxy" > /dev/null 2>&1  # this run's own, left by an earlier start of it
docker run -d --name "$PFX-proxy" --label "$LABEL" --label "ac-out=$OUT_ABS" --cpu-shares "$CPU_SHARES" --network host \
  ${AC_DOCKER_USER:+--user "$AC_DOCKER_USER"} -v "$HERE:$HERE:ro" -v "$OUT:$OUT" "$IMAGE" \
  python3 -I "$HERE/${PROXY[0]}" "${PROXY[@]:1}" --listen "127.0.0.1:$PROXY_PORT" --log "$OUT/proxy.jsonl" > /dev/null \
  || { echo "proxy container did not start"; exit 1; }
sleep 2
[ "$(docker inspect -f '{{.State.Running}}' "$PFX-proxy" 2> /dev/null)" = true ] \
  || { echo "proxy exited:"; docker logs "$PFX-proxy" 2>&1 | tail -5; exit 1; }
echo "$(date +%H:%M:%S) $ARM: ${PROXY[0]} 127.0.0.1:$PROXY_PORT -> $(printf '127.0.0.1:%s ' "${UPS[@]}")${RACE_ARGS:+($RACE_ARGS) }; tasks $TASK_SET/$TASKS; containers $PFX-*"
DEADLINE=()
[ -n "${BUDGET_MIN:-}" ] && DEADLINE=(--start-deadline-min "$BUDGET_MIN")
SAMP=()
[ -n "${SAMPLING:-}" ] && SAMP=(--sampling "$SAMPLING")
read -r -a MORE <<< "${RUN_ARGS:-}"
nice -n 19 python3 "$HERE/run.py" --arm "$ARM" --base-url "http://127.0.0.1:$PROXY_PORT/v1" --out "$OUT/res" \
  --task-set "$TASK_SET" --tasks "$TASKS" --time-cap-min "$TIME_CAP_MIN" --turn-cap "$TURN_CAP" --context "$CTX" \
  --health-wait-min 5 --work-root "$OUT/work" "${DEADLINE[@]}" "${SAMP[@]}" --sandbox docker --image "$IMAGE" \
  --docker-label "$LABEL" --docker-prefix "$PFX" --docker-cpu-shares "$CPU_SHARES" "${MORE[@]}"
rc=$?
rm -rf "$OUT/work/$ARM"
finish $rc
exit $rc
