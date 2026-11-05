#!/usr/bin/env bash
# Release tests T2 and T3 (2026-11-05), step 1: greedy identity of engine builds on one model at one server line.
# Each build's llama-server is started on the rig with the agent check's flags (bench/agent_check/release_check.sh's
# COMMON at CTX, then LINE) through bench/agent_check/rig_server.sh (its RAM watchdog and monitors), and
# greedy_requests.py sends the four prompts of prompts.jsonl at temperature 0, N tokens each, once on the server's
# own path (pipelined speculation when LINE has --pipe-depth) and once with top-2 probabilities. compare_greedy.py
# then compares the builds' outputs. Run from a machine that only runs ssh, with nothing else on the rig.
# Env: RIG (ssh alias), MODEL (on the rig), BUILDS ('<folder under /opt/llama-builds> ...': the first is the
#   reference), LINE (the server's arguments after COMMON), DIR (a work folder on the rig), OUT (a local folder);
#   optional CTX (262144), N (256), PORT (8091), SERVER_ENV (variables for every build's server), TAG_SUFFIX (appended
#   to each output's name, e.g. -v3add with SERVER_ENV), MIN_AVAIL_MB (400), PROMPTS (run/prompts.jsonl)
# Output: OUT/greedy-<build tag><suffix>.jsonl (build tag = the folder name after "pascal-p106-"),
#   OUT/rig-logs<suffix>.tar.gz (server logs, GPU and memory monitors)
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BUILDS:?}" "${LINE:?}" "${DIR:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
PROMPTS=${PROMPTS:-$HERE/prompts.jsonl}
CTX=${CTX:-262144}; N=${N:-256}; PORT=${PORT:-8091}; SFX=${TAG_SUFFIX:-}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
SRV=$REPO/bench/agent_check/rig_server.sh
mkdir -p "$OUT"
ssh "$RIG" "mkdir -p $DIR/logs && cat > $DIR/prompts.jsonl" < "$PROMPTS"
ssh "$RIG" "cat > $DIR/greedy_requests.py" < "$HERE/greedy_requests.py"
for f in $(grep -o '"attach": "[^"]*"' "$PROMPTS" | cut -d'"' -f4); do
    ssh "$RIG" "cat > $DIR/$(basename "$f")" < "$REPO/$f"
done
for b in $BUILDS; do
    t=${b##*-pascal-p106-}$SFX
    echo "=== $t $(date -Is): ${SERVER_ENV:-} $LINE"
    if ! ssh "$RIG" "ACTION=start ARM=$t BIN=/opt/llama-builds/$b/bin/llama-server MODEL=$MODEL PORT=$PORT \
        LOGDIR=$DIR/logs MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} SERVER_ENV='${SERVER_ENV:-}' SERVER_ARGS='$COMMON $LINE' \
        bash -s" < "$SRV"; then
        echo "=== $t: the server did not start"
        ssh "$RIG" "ACTION=stop PORT=$PORT LOGDIR=$DIR/logs bash -s" < "$SRV"
        continue
    fi
    ssh "$RIG" "cd $DIR && timeout 3600 python3 -I greedy_requests.py $PORT prompts.jsonl $N greedy-$t.jsonl; \
        echo \"requests rc=\$?\""
    ssh "$RIG" "ACTION=stop PORT=$PORT LOGDIR=$DIR/logs bash -s" < "$SRV"
    ssh "$RIG" "cat $DIR/greedy-$t.jsonl" > "$OUT/greedy-$t.jsonl"
done
ssh "$RIG" "ACTION=end PORT=$PORT LOGDIR=$DIR/logs bash -s" < "$SRV"
ssh "$RIG" "tar czf - -C $DIR logs" > "$OUT/rig-logs$SFX.tar.gz"
ssh "$RIG" "rm -rf $DIR/logs $DIR/greedy-*.jsonl"
echo "=== done $(date -Is)"
