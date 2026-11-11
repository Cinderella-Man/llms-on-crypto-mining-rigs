#!/usr/bin/env bash
# The v4night set replayed on one P106 (Q1321): scripts/replay_checks.sh copied to the rig and started detached, every
# run of the set's runs.tsv on card 0, no live CPU check.
# Env: RIG (the command that reaches the rig, default 'ssh rig'), W (a folder on the rig holding the unpacked set as
#   v4night/; the output goes to W/v4night-replay), BIN (bin/ of the merged rig build on the rig)
# Run from the repository root; the run's end is the file W/v4night-replay/done.
set -euo pipefail
RIG=${RIG:-ssh rig}
W=${W:?W=<folder on the rig holding v4night/>}
BIN=${BIN:?BIN=<bin/ of the rig build>}
$RIG "mkdir -p $W/v4night-replay-run && cat > $W/v4night-replay-run/replay_checks.sh" < scripts/replay_checks.sh
$RIG "md5sum $W/v4night-replay-run/replay_checks.sh $W/v4night/runs.tsv"
# an ssh call that starts a background job waits for it unless cut short
timeout 30 $RIG "cd $W/v4night-replay-run && SET=$W/v4night BIN=$BIN OUT=$W/v4night-replay DEV=0 CPU_CHECK=0 \
    setsid nohup bash replay_checks.sh > nohup.log 2>&1 < /dev/null &" || true
