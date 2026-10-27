#!/usr/bin/env bash
# Dry run of bench/agent_check/release_check.sh's replay mode without the rig: the client machine also plays the rig,
# with bench/agent_check/mock_server.py as llama-server, and the machine this runs on only relays (ssh). Two labels, a
# (20 ms per generated token) and b (25 ms), replay one short recorded trace; the report must give b/a = 1.25 per
# generated token, and the mock and its log folder must be gone afterwards. Run from the repository root.
# Env: CLIENT (ssh alias of the client), WORK (a scratch folder on the client), REPLAY_LOG (a recorded proxy.jsonl on
# the client), REPLAY_TRACES (default one trace of the answer store's control arm), OUT.
set -uo pipefail
: "${CLIENT:?}" "${WORK:?}" "${REPLAY_LOG:?}"
O=${OUT:-replay_dryrun}
export RIG=$CLIENT CLIENT RIG_PORT=18591 RELAY_PORT=8198 CLIENT_PORT=18593 PROXY_PORT=18594
export CLIENT_DIR=$WORK/bench CLIENT_OUT=$WORK/out RIG_LOGDIR=/tmp/agent-check-replaytest
export REL_BIN=$WORK/bench/agent_check/mock_server.py MODEL=/dev/null CTX=262144 ARMS=rel REPLAY_LOG
export REPLAY_TRACES=${REPLAY_TRACES:-rel/bugfix-javascript-triangle/a1/t1}
ssh "$CLIENT" "rm -rf $WORK/out"
REPLAY_LABEL=a REL_ARGS="--decode-ms 20" STEPS='prep run report collect' bench/agent_check/release_check.sh "$O" \
  || echo "=== a failed ($?)"
REPLAY_LABEL=b REL_ARGS="--decode-ms 25" STEPS='run report collect' bench/agent_check/release_check.sh "$O" \
  || echo "=== b failed ($?)"
ssh "$CLIENT" "pgrep -af mock_server.py | grep -v pgrep; ls /tmp | grep agent-check-replaytest"
echo "=== done"
