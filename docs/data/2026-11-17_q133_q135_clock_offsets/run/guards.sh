#!/usr/bin/env bash
# guards.sh: the safety guards of a clock-offset sitting (Q133/Q135). Run as root on the rig.
#
#   guards.sh state FILE        record every card's processes, memory, power limits, clocks, temperature, P-state and
#                               both clock offsets (read-only; run at the start and at the end and compare)
#   guards.sh watchdog W        start scripts/nvml_watchdog.py detached (70 C, card $GPU above W watts for 5 s, any Xid);
#                               an already running watchdog is stopped first, so this also changes the power rule
#   guards.sh stop              stop the watchdog (it exits at its next sample)
#   guards.sh cap W             set the power limit of every card to W (never above the card's default)
#   guards.sh trip-test         test the watchdog's three trips with lowered thresholds and stand-in processes, before
#                               any offset: temperature (15 C), card $GPU's power (1 W) and a kernel-log line with a test
#                               pattern; each must set card $GPU's offsets to 0 and kill the stand-ins
#
# Env [defaults]: WORK [/tmp/q133] (the watchdog's log wd.csv and marker TRIPPED go there), WATCHDOG
#   [$WORK/nvml_watchdog.py], KIT [$WORK/mem_offset-kit-20261024], GPU [0] (the card under test)
set -uo pipefail
WORK=${WORK:-/tmp/q133}
WATCHDOG=${WATCHDOG:-$WORK/nvml_watchdog.py}
KIT=${KIT:-$WORK/mem_offset-kit-20261024}
GPU=${GPU:-0}
cd "$WORK" || exit 2

running() { ps -eo args | grep -v grep | grep -q "nvml_watchdog.py"; }
stop() {
    touch "$WORK/WD_STOP"
    for _ in $(seq 1 20); do running || break; sleep 0.5; done
    rm -f "$WORK/WD_STOP"
    running && { echo "watchdog still running"; return 1; }
    echo "watchdog stopped"
}

case "${1:-}" in
state)
    {
        date '+%F %T'
        echo "## processes"
        ps -eo pid,comm,args | grep -E "llama|python|mmvq|memtest|nvidia-smi" | grep -v grep || echo none
        echo "## cards"
        nvidia-smi --query-gpu=index,pci.bus_id,memory.used,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,pstate,persistence_mode,power.draw,vbios_version --format=csv
        echo "## offsets (NVML, read-only)"
        python3 "$KIT/nvml_memoc.py" --no-system probe | grep -E "^gpu|vf offset"
        echo "## Xid this boot"
        journalctl -k -b -q | grep -c "NVRM: Xid"
    } > "${2:?file}" 2>&1
    cat "$2" ;;
watchdog)
    running && stop
    rm -f "$WORK/WD_STOP"
    setsid nohup python3 "$WATCHDOG" --gpu "$GPU" --log "$WORK/wd.csv" --marker "$WORK/TRIPPED" --temp-max 70 \
        --power-max "${2:?watts}" --power-secs 5 --stop "$WORK/WD_STOP" > "$WORK/wd.out" 2>&1 < /dev/null &
    sleep 2
    running && grep "^#" "$WORK/wd.csv" | tail -n 1 ;;
stop) stop ;;
cap)
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        def=$(nvidia-smi -i "$g" --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
        [ "${2:?watts}" -le "$def" ] || { echo "card $g: $2 W is above its default $def W: refused"; continue; }
        nvidia-smi -i "$g" -pl "$2" | grep -o "set to .*"
    done
    nvidia-smi --query-gpu=index,power.limit,enforced.power.limit --format=csv,noheader ;;
trip-test)
    mkdir -p "$WORK/test"
    ln -sf /bin/sleep "$WORK/test/mmvq_pascal"
    ln -sf /bin/sleep "$WORK/test/llama-server"
    (setsid nohup "$WORK/test/mmvq_pascal" 300 > /dev/null 2>&1 < /dev/null &)
    (setsid nohup "$WORK/test/llama-server" 300 > /dev/null 2>&1 < /dev/null &)
    sleep 1
    echo "== A: temperature limit lowered to 15 C"
    python3 "$WATCHDOG" --gpu "$GPU" --log "$WORK/test/wdA.csv" --marker "$WORK/test/TRIPPED_A" --temp-max 15 --duration 4
    cat "$WORK/test/TRIPPED_A"
    ps -eo comm | grep -qE "^(mmvq_pascal|llama-server)$" && echo "A: stand-ins still running: FAIL" || echo "A: stand-ins killed"
    (setsid nohup "$WORK/test/mmvq_pascal" 300 > /dev/null 2>&1 < /dev/null &)
    sleep 0.5
    echo "== B: a kernel-log line with a test pattern, and card $GPU's power limit lowered to 1 W"
    (sleep 2; echo "clock-offset watchdog test line Q133WDTEST" > /dev/kmsg) &
    python3 "$WATCHDOG" --gpu "$GPU" --log "$WORK/test/wdB.csv" --marker "$WORK/test/TRIPPED_B" --power-max 1 \
        --power-secs 5 --xid-pattern Q133WDTEST --duration 9
    cat "$WORK/test/TRIPPED_B"
    ps -eo comm | grep -qx mmvq_pascal && echo "B: stand-in still running: FAIL" || echo "B: stand-in killed" ;;
*) sed -n '2,18p' "$0"; exit 2 ;;
esac
