#!/usr/bin/env bash
# A minimal job queue for unattended runs on the rig: runs $RQ/jobs/*.sh one at a time in name order (jobs can be added
# while it runs) and stops when the file $RQ/STOP exists. Before each job it kills any llama.cpp binary still running
# from /opt/llama-builds, so a job that left a server behind cannot hold the cards for the next one. Each job's output
# goes to $RQ/logs/<job>.log; start and end times with the exit code to $RQ/runner.log; finished jobs move to $RQ/done/.
#
#   RIG=${RIG:-rig}                       # the rig's SSH alias
#   scp scripts/rig_queue.sh "$RIG":/tmp/ && ssh "$RIG" 'mkdir -p /tmp/rq/jobs'
#   scp 10-first.sh 20-second.sh "$RIG":/tmp/rq/jobs/
#   ssh "$RIG" 'nohup bash /tmp/rig_queue.sh > /dev/null 2>&1 &'
#   ssh "$RIG" 'touch /tmp/rq/STOP'       # stop after the running job; tail /tmp/rq/runner.log to follow it
#
# Files in jobs/ that don't end in .sh are never run, so shared settings can sit there (e.g. jobs/common.inc, sourced
# by each job). Used for the overnight queue of 2026-10-23 (docs/data/2026-10-23_overnight/rig/).
# Env: RQ (queue directory, default /tmp/rq).
RQ=${RQ:-/tmp/rq}
cd "$RQ" || exit 1
mkdir -p jobs running done logs out
while [ ! -f STOP ]; do
    j=$(ls jobs/*.sh 2> /dev/null | sort | head -1)
    if [ -z "$j" ]; then sleep 30; continue; fi
    n=$(basename "$j" .sh)
    mv "$j" "running/$n.sh"
    pkill -f "^/opt/llama-builds/.*/bin/llama-" 2> /dev/null; sleep 5
    echo "$(date -Is) start $n" >> runner.log
    bash "running/$n.sh" > "logs/$n.log" 2>&1 < /dev/null
    echo "$(date -Is) end $n rc=$?" >> runner.log
    mv "running/$n.sh" done/
done
echo "$(date -Is) runner stopped" >> runner.log
