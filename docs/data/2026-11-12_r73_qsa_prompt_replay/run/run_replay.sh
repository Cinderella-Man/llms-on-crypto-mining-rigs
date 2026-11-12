#!/usr/bin/env bash
# R73 (1) (Q1272): the qsa-prompt set replayed on one P106: scripts/replay_checks.sh copied to the rig and started
# detached, every run of the set's runs.tsv in its order on card 0. From R112's run_replay.sh (itself from
# docs/data/2026-11-11_v4night_replay/run/run_replay.sh), with CPU_CHECK passed through: a pid file for one waiting
# command and a record of the cards and the kernel log's Xid lines before and after.
# Env: RIG (the command that reaches the rig, default 'ssh rig'), SET (the unpacked set on the rig: runs.tsv, cases/),
#   BIN (bin/ of the rig build on the rig), W (a folder on the rig for this run: the copied script, the output in W/out,
#   the run's pid in W/pid), TAG (default empty: the output goes to W/out$TAG, the pid to W/pid$TAG, the cards to
#   W/cards$TAG-start.txt and -end.txt), RUNS (default every run of the set's runs.tsv: the runs to replay),
#   CPU_CHECK (default 200: about one case in 200 with inputs under 32 MB also checked on the rig's CPU, first run only)
# Usage, from the repository root:
#   SET=... BIN=... W=... bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh start  # copy, record, start
#   SET=... BIN=... W=... bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh wait   # ends with the run
#   SET=... BIN=... W=... bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh end    # the rig afterwards
set -euo pipefail
RIG=${RIG:-ssh rig}
SET=${SET:?SET=<the unpacked qsa-prompt folder on the rig>}
BIN=${BIN:?BIN=<bin/ of the rig build>}
W=${W:?W=<a folder on the rig for this run>}
TAG=${TAG:-}
RUNS=${RUNS:-}
CPU_CHECK=${CPU_CHECK:-200}
state() { # <tag>: the cards, the kernel log's Xid lines since boot, the llama and test processes, the load
    $RIG "cd $W && { date -Is; nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader; \
        echo \"xid lines: \$(journalctl -k -b --no-pager | grep -c 'Xid')\"; \
        echo \"processes: \$(ps -eo comm | grep -c -x -E 'llama-.*|test-backend-op|multitoken_bitid|rpc-server')\"; \
        echo \"load: \$(cut -d' ' -f1-3 /proc/loadavg)\"; } > cards$TAG-$1.txt; cat cards$TAG-$1.txt"
}
case ${1:?start|wait|end} in
start)
    $RIG "mkdir -p $W && cat > $W/replay_checks.sh" < scripts/replay_checks.sh
    $RIG "md5sum $W/replay_checks.sh $SET/runs.tsv $BIN/test-backend-ops; grep -c . $SET/runs.tsv; ls $SET/cases | wc -l"
    state start
    # an ssh call that starts a background job waits for it unless cut short
    timeout 30 $RIG "cd $W && SET=$SET BIN=$BIN OUT=$W/out$TAG DEV=0 CPU_CHECK=$CPU_CHECK RUNS='$RUNS' \
        setsid nohup bash replay_checks.sh > nohup$TAG.log 2>&1 < /dev/null & echo \$! > $W/pid$TAG" || true
    $RIG "cat $W/pid$TAG; sleep 5; head -n 3 $W/out$TAG/console.txt"
    ;;
wait)
    $RIG "tail --pid=\$(cat $W/pid$TAG) -s 15 -f /dev/null; cat $W/out$TAG/done; tail -n 1 $W/out$TAG/console.txt"
    ;;
end)
    state end
    ;;
esac
