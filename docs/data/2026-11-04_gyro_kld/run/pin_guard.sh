#!/usr/bin/env bash
# Releases the n-gram table pin (pin_range.py) when free memory runs low, or when the scoring chain has ended.
#   (env: OUT [MIN_AVAIL_GB]) bash pin_guard.sh
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$OUT/score.log"; }
while pgrep -f '[p]in_range.py' > /dev/null; do
    avail=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ "$avail" -lt "${MIN_AVAIL_GB:-15}" ]; then pkill -f '[p]in_range.py'; log "pin released: MemAvailable $avail GB"; exit 0; fi
    if [ -f "$OUT/score.done" ]; then pkill -f '[p]in_range.py'; log "pin released: the chain has ended"; exit 0; fi
    sleep 10
done
log "pin ended on its own (timeout)"
