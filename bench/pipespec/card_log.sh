# card_log.sh: the card log for the rig_*.sh scripts in this folder, which source it. card_log.py beside it is
# scripts/card_log.py byte for byte (scripts/card_log_test.sh checks it), so a copy of this folder carries both.
#   card_log_start CSV   samples every card every CARD_LOG_S seconds (0.5) through NVML, read-only, one process:
#                        P-state, SM and memory clock, power, temperature, memory used, utilization and throttle
#                        reasons, appended to CSV (scripts/card_table.py summarises it). It runs until card_log_stop,
#                        until the script that started it is gone (--parent) or until CSV.stop exists. It waits up to
#                        10 s for the first sample and says on stdout whether the log runs; a log that does not come up
#                        is reported (card_log.out beside CSV says why) and the run goes on without it.
#   card_log_stop        ends it with SIGTERM (it writes its stop line: samples, CPU seconds), waits up to 3 s, and
#                        never waits on one that hangs.
CL_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CL_PID="" CL_OUT=""
card_log_start() {
    local csv=$1 n=0
    CL_OUT=$(dirname "$csv")/card_log.out
    rm -f "$csv.stop"
    python3 -I "$CL_HERE/card_log.py" --log "$csv" --interval "${CARD_LOG_S:-0.5}" --parent $$ --stop "$csv.stop" \
        > "$CL_OUT" 2>&1 < /dev/null &
    CL_PID=$!
    until grep -qs '^[0-9]' "$csv"; do
        n=$((n + 1))
        if ! kill -0 "$CL_PID" 2> /dev/null; then
            echo "$(date -Is) card log: not running: $(tr '\n' ' ' < "$CL_OUT")"; CL_PID=""; return 0
        fi
        if [ "$n" -gt 20 ]; then echo "$(date -Is) card log: no sample after 10 s (it keeps trying; card_log.out)"; return 0; fi
        sleep 0.5
    done
    echo "$(date -Is) card log: every card every ${CARD_LOG_S:-0.5} s to $(basename "$csv")"
}
card_log_stop() {
    local n=0
    [ -n "$CL_PID" ] || return 0
    kill "$CL_PID" 2> /dev/null
    while kill -0 "$CL_PID" 2> /dev/null && [ "$n" -lt 6 ]; do sleep 0.5; n=$((n + 1)); done
    kill -9 "$CL_PID" 2> /dev/null
    CL_PID=""
    [ -s "$CL_OUT" ] || rm -f "$CL_OUT"
}
