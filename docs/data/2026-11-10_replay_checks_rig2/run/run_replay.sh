#!/usr/bin/env bash
# The `ladder3-rest` suite replayed on one P106 with the release build's own test-backend-ops: the sparse decode
# attention's six runs first, on a card nothing else has touched since boot, then the other 57 runs in the suite's
# order. Two calls of scripts/replay_checks.sh, so two output folders (qsa/ and rest/), the kernel log's Xid lines
# before and after, and one done marker for both.
#
# GGML_TEST_REPLAY_ALLOW_MISSING=1: the release build's test file holds cases that were added after the suite was
# recorded. Its filters select them too, and without the switch each such case fails its run as "not recorded". With
# it every recorded case is replayed and judged, and each run's summary counts the cases left out.
#
# Env: SET (the suite's folder), BIN (bin/ of the build), D (the output folder), CHECKS (scripts/replay_checks.sh),
#   DEV (the card, default 0), CPU_CHECK (default 200: a few small cases of each call's first run against the host's CPU)
# Usage, detached: SET=... BIN=... D=... CHECKS=... setsid nohup bash run_replay.sh > log 2>&1 < /dev/null &
set -uo pipefail
SET=${SET:?SET=<the recorded suite>}; BIN=${BIN:?BIN=<bin/ of the build>}; D=${D:?D=<output folder>}
CHECKS=${CHECKS:?CHECKS=<scripts/replay_checks.sh>}
export SET BIN DEV=${DEV:-0} CPU_CHECK=${CPU_CHECK:-200} ENV_ALL=GGML_TEST_REPLAY_ALLOW_MISSING=1
mkdir -p "$D"
rm -f "$D/done"
xid() { dmesg 2>/dev/null | grep -c 'NVRM: Xid'; }
{
    date -Is
    md5sum "$BIN/test-backend-ops"
    nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader
    echo "Xid lines in the kernel log: $(xid)"
} > "$D/before.txt" 2>&1
QSA=$(cut -f1 "$SET/runs.tsv" | grep '^qsa-' | tr '\n' ' ')
REST=$(cut -f1 "$SET/runs.tsv" | grep -v '^qsa-' | tr '\n' ' ')
OUT=$D/qsa RUNS=$QSA bash "$CHECKS"
echo "Xid lines after the sparse runs: $(xid)" >> "$D/before.txt"
OUT=$D/rest RUNS=$REST bash "$CHECKS"
{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader
    echo "Xid lines in the kernel log: $(xid)"
    dmesg 2>/dev/null | grep 'NVRM: Xid'
} > "$D/after.txt" 2>&1
date -Is > "$D/done"
