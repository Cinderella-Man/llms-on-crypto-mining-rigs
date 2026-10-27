#!/usr/bin/env bash
# Q149 end to end (2026-10-24): every e2e.sh step in order, most important first; one log per step in
# $WORK/e2e/logs/<step>.log and start/exit lines in $WORK/e2e/logs/progress.txt. Then e2e_ctl.sh and e2e_summary.sh.
# Usage: the environment of e2e.sh (BUILD, MODEL, TEXT, WORK, GPU), then run_all.sh
S=$(cd "$(dirname "$0")" && pwd)
L=${WORK:?WORK=<work dir>}/e2e/logs
mkdir -p $L
for s in ppl-off1 ppl-on1 gen-off gen-on ppl-off4 ppl-on4; do
    echo "== $s $(date +%T)" >> $L/progress.txt
    $S/e2e.sh $s > $L/$s.log 2>&1
    echo "   exit $? $(date +%T)" >> $L/progress.txt
done
