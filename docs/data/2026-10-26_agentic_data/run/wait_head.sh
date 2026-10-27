#!/usr/bin/env bash
# Waits for the next event of the MTP head run (agentic_head.sh, 2026-10-27): a step line in head.out other than the
# dump's progress lines, a failed gate, an error, or the end of the script. Prints the lines after line N up to it,
# then "CURSOR=<line>" for the next call.
#   OUT=<WORK>/head.out bash wait_head.sh N
OUT=${OUT:?the output file of agentic_head.sh, WORK/head.out}
n=${1:-0}
while :; do
    hit=$(tail -n +$((n + 1)) "$OUT" 2>/dev/null | awk '
        /^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] dump: [0-9]+ of / { next }
        /^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] |FAIL|Traceback|[Ee]rror|stopped|Killed/ { print NR; exit }')
    if [ -n "$hit" ]; then
        tail -n +$((n + 1)) "$OUT" | head -n "$hit"
        echo "CURSOR=$((n + hit))"
        exit 0
    fi
    if ! pgrep -f 'bash .*agentic_head.sh' > /dev/null; then
        tail -n +$((n + 1)) "$OUT" | tail -20
        echo "HEAD SCRIPT GONE; CURSOR=$(wc -l < "$OUT")"
        exit 0
    fi
    sleep 60
done
