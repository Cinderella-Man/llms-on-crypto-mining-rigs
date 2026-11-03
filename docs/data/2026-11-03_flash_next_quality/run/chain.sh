#!/usr/bin/env bash
# The Flash-Next quality chain as one detached job. Each step is <step>.sh next to this file, read when the step
# starts; it logs to $WORK/logs/<step>.log and leaves $WORK/done/<step> (its end time) or $WORK/failed/<step> (why).
# Steps already done are skipped, so after a failure the chain is simply started again; a step whose prerequisite
# failed is skipped and marked failed too. The background steps run beside the main line once their prerequisites are
# done. A file $WORK/STOP stops the chain before its next step.
#   ENV=<environment file> setsid nohup bash chain.sh >> $WORK/logs/chain.log 2>&1 < /dev/null &
#   MAIN="..." / BACKGROUND="..." override the step lists (e.g. to rerun one step).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "${ENV:?}"
. "$HERE/common.sh"
: "${WORK:?}"
mkdir -p "$WORK/logs" "$WORK/done" "$WORK/failed"
rm -f "$WORK"/failed/*

declare -A NEEDS=(
    [fn1a_download]="" [fn1a_check]="fn1a_download" [fn1a_archive]="fn1a_check" [tools]=""
    [fn1b]="fn1a_check" [fn2]="fn1b" [fn4]="fn1a_check tools" [fn18]="fn1b fn4 tools" [fn22]="fn1b" [fn3]="fn1b tools"
)
BACKGROUND=${BACKGROUND:-"tools fn1a_archive"}
MAIN=${MAIN:-"fn1a_download fn1a_check fn1b fn2 fn4 fn18 fn22"}

finished() { [ -e "$WORK/done/$1" ] || [ -e "$WORK/failed/$1" ]; }

run_step() {
    local s=$1 p rc
    if [ -e "$WORK/done/$s" ]; then log "$s: already done"; return 0; fi
    for p in ${NEEDS[$s]}; do
        until finished "$p"; do sleep 60; done
        if [ ! -e "$WORK/done/$p" ]; then
            echo "skipped: $p failed" > "$WORK/failed/$s"; log "$s: skipped, $p failed"; return 1
        fi
    done
    if [ -e "$WORK/STOP" ]; then echo "stopped: $WORK/STOP" > "$WORK/failed/$s"; log "$s: not started (STOP)"; return 1; fi
    log "$s: start"
    bash "$HERE/$s.sh" >> "$WORK/logs/$s.log" 2>&1 < /dev/null
    rc=$?
    if [ $rc -eq 0 ]; then
        date -Is > "$WORK/done/$s"; log "$s: done"
    else
        echo "exit $rc at $(date -Is)" > "$WORK/failed/$s"; log "$s: FAILED (exit $rc, see logs/$s.log)"
    fi
    return $rc
}

log "chain start (pid $$): main $MAIN; background $BACKGROUND"
for s in $BACKGROUND; do run_step "$s" & done
for s in $MAIN; do run_step "$s"; done
wait
log "chain end: done $(ls "$WORK/done" | grep -v '^chain$' | tr '\n' ' '); failed $(ls "$WORK/failed" | tr '\n' ' ')"
date -Is > "$WORK/done/chain"
