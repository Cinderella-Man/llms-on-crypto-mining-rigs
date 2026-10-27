#!/usr/bin/env bash
# Watches agentic data run 1 (2026-10-26/27; writers run1-{q8,va,vb}, each a bench/agentic_data/run.sh folder under
# RUN1) and appends one line per event to RUN1/run1-events.log: each run's step lines and gate verdicts, session
# errors, a stall (no session finished in any run for 75 min), a progress line every 2 hours, each run's end and
# resumption, and "ALL DONE" once all three have ended after their checks step. wait_event.sh reads it.
#   RUN1=<folder, as the runs' WORK paths start> nohup setsid bash watch_run1.sh > /dev/null 2>&1 &
cd "${RUN1:?folder holding run1-q8, run1-va, run1-vb}" || exit 1
RUNS=${RUNS:-q8 va vb}
EV=run1-events.log
for r in $RUNS; do
    tail -n 0 -F "run1-$r/run.out" 2>/dev/null \
        | grep --line-buffered -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} |^PASS|FAIL|Traceback|Error| tasks -> ' \
        | sed -u "s/^/[$r] /" >> "$EV" &
    for s in s1 s2; do
        tail -n 0 -F "run1-$r/$s/run.log" 2>/dev/null | grep --line-buffered ERROR | sed -u "s/^/[$r $s] /" >> "$EV" &
    done
done
alive() {   # a run.sh process whose WORK is this run's folder
    local p
    for p in $(pgrep -f 'agentic_data/run.sh'); do
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "WORK=$RUN1/run1-$1" && return 0
    done
    return 1
}
declare -A state   # alive or ended; a resumed run (stage2_resume.sh) goes from ended back to alive
for r in $RUNS; do if alive "$r"; then state[$r]=alive; else state[$r]=ended; fi; done
stalled=0 tick=0
echo "WATCH START $(date +%T)" >> "$EV"
while :; do
    sleep 300
    tick=$((tick + 1))
    n_end=0
    for r in $RUNS; do
        if alive "$r"; then
            [ "${state[$r]}" = ended ] && echo "[$r] RESUMED $(date +%T)" >> "$EV"
            state[$r]=alive
        else
            [ "${state[$r]}" = alive ] && echo "[$r] ENDED $(date +%T): $(tail -1 "run1-$r/run.out" | cut -c1-160)" >> "$EV"
            state[$r]=ended
            [ -s "run1-$r/checks/map.jsonl" ] && n_end=$((n_end + 1))   # ended after its checks step
        fi
    done
    if [ "$n_end" -eq "$(echo $RUNS | wc -w)" ]; then
        echo "ALL DONE $(date +%T)" >> "$EV"
        kill $(jobs -p) 2>/dev/null
        exit 0
    fi
    if [ -z "$(find run1-*/s1/sessions run1-*/s2/sessions -name '*.json' -mmin -75 2>/dev/null | head -1)" ]; then
        [ "$stalled" -eq 0 ] && echo "STALL $(date +%T): no session finished in any run for 75 min" >> "$EV"
        stalled=1
    else
        stalled=0
    fi
    if [ $((tick % 24)) -eq 0 ]; then
        line="PROGRESS $(date +%T):"
        for r in $RUNS; do
            line+=" $r s1 $(ls run1-$r/s1/sessions 2>/dev/null | grep -v error | grep -c '\.json$') s2 $(ls run1-$r/s2/sessions 2>/dev/null | grep -v error | grep -c '\.json$');"
        done
        echo "$line claims s2 $(ls run1-claims/s2 2>/dev/null | wc -l)" >> "$EV"
    fi
done
