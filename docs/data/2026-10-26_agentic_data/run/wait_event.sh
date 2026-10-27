#!/usr/bin/env bash
# Waits for the next event in run1-events.log (watch_run1.sh, agentic data run 1, 2026-10-27) that needs a look (a
# failed gate, a traceback, a stall, a run's end, the start of stage 2 or "ALL DONE"), prints every line after line N
# up to it, then "CURSOR=<line>" for the next call.
#   RUN1=<the watcher's folder> bash wait_event.sh N
EV=${RUN1:?folder of the watcher}/run1-events.log
n=${1:-0}
while :; do
    hit=$(tail -n +$((n + 1)) "$EV" 2>/dev/null \
        | grep -n -m 1 -E 'FAIL|Traceback|STALL|ENDED|ALL DONE|stage 2|exit' | cut -d: -f1)
    if [ -n "$hit" ]; then
        tail -n +$((n + 1)) "$EV" | head -n "$hit"
        echo "CURSOR=$((n + hit))"
        exit 0
    fi
    sleep 60
done
