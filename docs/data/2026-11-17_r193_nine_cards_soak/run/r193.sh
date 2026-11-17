#!/usr/bin/env bash
# r193.sh: every card of a P106-090 rig at one memory and core clock offset, checked card by card, soaked all at once,
# then checked again (R193, Q1569). The checks are the one-card sitting's (Q133/Q135,
# docs/data/2026-11-17_q133_q135_clock_offsets/). Run as root on the rig, with the memory-offset kit (bench/mem_offset)
# unpacked in $KIT and this folder's files (plus scripts/nvml_watchdog.py and the sitting's run/greedy_check.py) in the
# same folder as this script.
#
#   r193.sh state FILE        read-only: processes, every card's memory, power limit, clocks, temperature, P-state,
#                             both offsets, Xid lines this boot, host memory
#   r193.sh trip-test         the watchdog's three trips on all cards before any offset, each with stand-in processes
#                             that must be killed: temperature (15 C), a kernel-log test line, power (1 W)
#   r193.sh watchdog          start the watchdog detached on all cards: any card above 70 C, any card above 80 W for
#                             5 s, any new Xid -> every offset of every card to 0, the load processes killed
#   r193.sh stop-watchdog     stop it (it exits at its next sample)
#   r193.sh pre               at offset 0, detached: the exact check's base file and stock rerun, the server, the
#                             greedy check at stock against the sitting's reference hash, each card's streaming read
#   r193.sh cards             step 1, detached: card by card, memory +$MEM and core +$CORE (going up only), read back
#                             from a new process, the full mmvq sweep on that card, the greedy check, an Xid scan
#   r193.sh soak MIN          step 2, detached: one mmvq loop per card and one greedy loop, all at once, MIN minutes
#   r193.sh post              step 3, detached: the greedy check on the still loaded server, the server stopped,
#                             cuda_memtest on every card at once (tests 0-5, 7, 8 and 10, then test 6), the exact check
#   r193.sh server start|stop llama-server with $MODEL on all cards, greedy only (port $PORT)
#   r193.sh reset-all         every offset of every card to 0, read back
#   r193.sh hottest [MIN]     each card's highest temperature and power in the watchdog log over the last MIN minutes
#
# Any failure in pre, cards, soak or post sets every offset of every card to 0, stops the server and writes one line
# to $OUT/FAILED. Each detached step writes $OUT/<STEP>_DONE with its exit status.
#
# Env [defaults]: WORK [/tmp/r193], KIT [$WORK/mem_offset-kit-20261024], OUT [$WORK/out], BIN (llama.cpp bin
#   directory), MODEL (GGUF), WIKI (wikitext-2's wiki.test.raw), MEM [1200] (MT/s), CORE [300] (MHz),
#   CARDS ["0 1 2 3 4 5 6 7 8"], DEVS [0,1,2,3,4,5,6,7,8], PORT [8193], REF_SHA [200fe90f478c1933: the sitting's
#   greedy hash at stock], UNITS [memoc_units.json next to this script: NVML's memory unit, measured on card 0],
#   MMVQ_ARGS [--types q8_0,q5_0,q4_1,q5_K,iq4_xs]. An env.sh next to this script is read first, if there is one.
set -uo pipefail
[ -f "$(dirname "$0")/env.sh" ] && . "$(dirname "$0")/env.sh"
WORK=${WORK:-/tmp/r193}
KIT=${KIT:-$WORK/mem_offset-kit-20261024}
OUT=${OUT:-$WORK/out}
here=$(cd "$(dirname "$0")" && pwd)
self="$here/$(basename "$0")"
WATCHDOG=${WATCHDOG:-$here/nvml_watchdog.py}
GREEDY=${GREEDY:-$here/greedy_check.py}
UNITS=${UNITS:-$here/memoc_units.json}
MEM=${MEM:-1200}
CORE=${CORE:-300}
CARDS=${CARDS:-0 1 2 3 4 5 6 7 8}
DEVS=${DEVS:-0,1,2,3,4,5,6,7,8}
PORT=${PORT:-8193}
REF_SHA=${REF_SHA:-200fe90f478c1933}
MMVQ_ARGS=${MMVQ_ARGS:---types q8_0,q5_0,q4_1,q5_K,iq4_xs}
P="python3 $KIT/nvml_memoc.py"
mkdir -p "$OUT"
cd "$WORK" || exit 2

log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/r193.log"; }
tnow() { date '+%F %T.%3N'; }
tripped() { [ -e "$WORK/TRIPPED" ]; }
xid_since() { journalctl -k -q --since "$1" 2>/dev/null | grep -ci 'NVRM: Xid'; }
readback() { echo "$($P get "$1" 2>&1 | head -n 1 | cut -d' ' -f6) $($P --core get "$1" 2>&1 | head -n 1 | cut -d' ' -f6)"; }

reset_all() {
    local g bad=0
    for g in $CARDS; do
        $P reset "$g" >> "$OUT/set.log" 2>&1 || bad=1
        $P --core reset "$g" >> "$OUT/set.log" 2>&1 || bad=1
    done
    for g in $CARDS; do log "reset: card $g offsets now (mem core) $(readback "$g")"; done
    return $bad
}

fail_all() {  # fail_all "one plain line"
    log "FAIL: $1"
    echo "$(date '+%F %T') $1" >> "$OUT/FAILED"
    touch "$WORK/STOP_SOAK"
    server stop
    reset_all
}

server() {
    case "$1" in
    start)
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVS" setsid nohup "$BIN/llama-server" -m "$MODEL" -ngl 99 \
            -fa on -c 4096 -np 1 -t 2 --host 127.0.0.1 --port "$PORT" --no-warmup > "$OUT/server.log" 2>&1 < /dev/null &
        log "server: pid $!, the model on cards $DEVS"
        for _ in $(seq 1 240); do
            sleep 3
            tripped && { log "server: the watchdog tripped during the load"; return 9; }
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { log "server: ready"; return 0; }
            ps -eo comm | grep -qx llama-server || { log "server: exited during the load (see server.log)"; return 1; }
        done
        log "server: not ready after 12 min"; return 1 ;;
    stop)
        ps -eo comm | grep -qx llama-server || return 0
        pkill -TERM -x llama-server; sleep 3; pkill -KILL -x llama-server 2>/dev/null; log "server: stopped" ;;
    esac
}

greedy() {  # greedy TAG: 0 same hash as $REF_SHA, 1 differs, 3 a request failed
    [ -s "$OUT/greedy-ref.sha" ] || echo "$REF_SHA" > "$OUT/greedy-ref.sha"
    python3 "$GREEDY" --url "http://127.0.0.1:$PORT" --out "$OUT" --tag "$1" --ref ref >> "$OUT/greedy.log" 2>&1
    local rc=$?
    tail -n 1 "$OUT/greedy.log" >> "$OUT/r193.log"
    return $rc
}

mmvq() {  # mmvq CARD TAG [ARGS...]: prints "exit checks_pass checks_fail peak_rss_kb stream_gbs"
    local g=$1 tag=$2 f rc
    shift 2
    f="$OUT/mmvq_$tag.csv"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" /usr/bin/time -f "peak_rss_kb %M" -o "$OUT/mmvq_$tag.time" \
        "$KIT/bin/mmvq_pascal" ${@:-$MMVQ_ARGS} --csv "$f" > "$OUT/mmvq_$tag.log" 2>&1
    rc=$?
    echo "$rc $(awk -F, '$23 == "PASS"' "$f" 2>/dev/null | wc -l) $(awk -F, '$23 == "FAIL"' "$f" 2>/dev/null | wc -l)" \
        "$(awk '{ print $2 }' "$OUT/mmvq_$tag.time" 2>/dev/null | tail -n 1)" \
        "$(awk -F, '$3 == "STREAM_READ_PLAIN" { print $15 }' "$f" 2>/dev/null)"
}

state() {
    {
        date '+%F %T'
        echo "## processes"
        ps -eo pid,comm,args | grep -E "llama|python|mmvq|memtest|nvidia-smi" | grep -v grep || echo none
        echo "## cards"
        nvidia-smi --query-gpu=index,memory.used,power.limit,power.default_limit,clocks.sm,clocks.mem,temperature.gpu,pstate,persistence_mode,power.draw,vbios_version --format=csv
        echo "## offsets (NVML, read-only)"
        $P --no-system probe | grep -E "^gpu|vf offset"
        echo "## Xid this boot"
        journalctl -k -b -q | grep -c "NVRM: Xid"
        echo "## host memory"
        free -m
    } > "${1:?file}" 2>&1
    cat "$1"
}

trip_test() {
    local t="$WORK/test"
    mkdir -p "$t"
    ln -sf /bin/sleep "$t/mmvq_pascal"
    ln -sf /bin/sleep "$t/llama-server"
    standins() {
        (setsid nohup "$t/mmvq_pascal" 300 > /dev/null 2>&1 < /dev/null &)
        (setsid nohup "$t/llama-server" 300 > /dev/null 2>&1 < /dev/null &)
        sleep 1
    }
    killed() { ps -eo comm | grep -qE "^(mmvq_pascal|llama-server)$" && echo "$1: stand-ins still running: FAIL" || echo "$1: stand-ins killed"; }
    echo "== A: temperature limit lowered to 15 C, all cards under test"
    standins
    python3 "$WATCHDOG" --gpus all --log "$t/wdA.csv" --marker "$t/TRIPPED_A" --temp-max 15 --power-max 80 --duration 4
    cat "$t/TRIPPED_A"; killed A
    echo "== B: a kernel-log line with a test pattern"
    standins
    (sleep 2; echo "clock-offset watchdog test line R193WDTEST" > /dev/kmsg) &
    python3 "$WATCHDOG" --gpus all --log "$t/wdB.csv" --marker "$t/TRIPPED_B" --power-max 80 --xid-pattern R193WDTEST --duration 6
    cat "$t/TRIPPED_B"; killed B
    echo "== C: power limit lowered to 1 W for 5 s, all cards under test"
    standins
    python3 "$WATCHDOG" --gpus all --log "$t/wdC.csv" --marker "$t/TRIPPED_C" --power-max 1 --power-secs 5 --duration 9
    cat "$t/TRIPPED_C"; killed C
}

watchdog() {
    ps -eo args | grep -v grep | grep -q "nvml_watchdog.py" && { echo "a watchdog already runs"; return 1; }
    rm -f "$WORK/WD_STOP"
    setsid nohup nice -n -10 python3 "$WATCHDOG" --gpus all --log "$WORK/wd.csv" --marker "$WORK/TRIPPED" \
        --temp-max 70 --power-max 80 --power-secs 5 --stop "$WORK/WD_STOP" > "$WORK/wd.out" 2>&1 < /dev/null &
    sleep 3
    grep "^#" "$WORK/wd.csv" | tail -n 1
}

stop_watchdog() {
    touch "$WORK/WD_STOP"
    for _ in $(seq 1 20); do ps -eo args | grep -v grep | grep -q "nvml_watchdog.py" || break; sleep 0.5; done
    rm -f "$WORK/WD_STOP"
    ps -eo args | grep -v grep | grep -q "nvml_watchdog.py" && { echo "watchdog still running"; return 1; }
    echo "watchdog stopped"
}

exact() {  # exact base | exact kld TAG
    B=$BIN M=$MODEL WIKI=$WIKI DEVS=$DEVS E=CUDA_DEVICE_ORDER=PCI_BUS_ID OUT=$OUT bash "$KIT/exact_check.sh" "$@"
}

stream() {  # stream TAG: a short streaming read of each card in turn (Q8_0 attn_q only)
    local g rc pass fails rss gbs
    [ -s "$OUT/stream.csv" ] || echo "tag,card,exit,checks_pass,checks_fail,stream_gbs" > "$OUT/stream.csv"
    for g in $CARDS; do
        tripped && return 9
        read -r rc pass fails rss gbs < <(mmvq "$g" "$1-card$g" --types q8_0 --shapes attn_q)
        echo "$1,$g,$rc,$pass,$fails,$gbs" >> "$OUT/stream.csv"
        log "stream $1 card $g: exit $rc, checks $pass/$fails fail, $gbs GB/s"
    done
}

pre() {
    local t0
    t0=$(date '+%F %T')
    for g in $CARDS; do [ "$(readback "$g")" = "0 0" ] || { fail_all "pre: card $g does not start at offset 0"; return 1; }; done
    log "pre: the exact check's base file and stock rerun at offset 0"
    exact base > "$OUT/exact_base_console.log" 2>&1 || { fail_all "pre: the exact check's stock reference failed"; return 1; }
    tripped && { fail_all "pre: the watchdog tripped"; return 1; }
    server start || { fail_all "pre: the server did not start"; return 1; }
    greedy stock; local rc=$?
    [ $rc = 0 ] || { fail_all "pre: the greedy check at stock does not give the sitting's hash $REF_SHA (exit $rc)"; return 1; }
    stream stock || { fail_all "pre: the watchdog tripped"; return 1; }
    [ "$(xid_since "$t0")" = 0 ] || { fail_all "pre: an Xid at stock"; return 1; }
    log "pre: done, the server holds the model on every card at stock"
}

cards() {
    local g t0 rb rc pass fails rss gbs grc gtxt xid v
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { log "cards: the server is not running"; return 1; }
    [ -s "$OUT/cards.csv" ] || echo "time,card,readback_mem,readback_core,mmvq_exit,checks_pass,checks_fail,stream_gbs,greedy,xid,mmvq_peak_rss_kb,verdict" > "$OUT/cards.csv"
    for g in $CARDS; do
        tripped && { fail_all "cards: the watchdog tripped before card $g"; return 9; }
        t0=$(tnow)
        $P --state "$UNITS" --cap "$MEM" set "$g" "$MEM" >> "$OUT/set.log" 2>&1 \
            && $P --core set "$g" "$CORE" >> "$OUT/set.log" 2>&1 \
            || { fail_all "card $g refused the offset"; return 1; }
        rb=$(readback "$g")
        log "card $g: set, read back from a new process (mem core): $rb"
        read -r rc pass fails rss gbs < <(mmvq "$g" "card$g")
        greedy "card$g"; grc=$?
        case $grc in 0) gtxt=same ;; 1) gtxt=DIFFERS ;; *) gtxt=ERROR ;; esac
        xid=$(xid_since "${t0%.*}")
        v=PASS
        [ "$rb" = "$MEM $CORE" ] && [ "$rc" = 0 ] && [ "$fails" = 0 ] && [ "$pass" -gt 0 ] && [ -n "$gbs" ] \
            && [ "$grc" = 0 ] && [ "$xid" = 0 ] && ! tripped || v=FAIL
        echo "${t0%.*},$g,${rb% *},${rb#* },$rc,$pass,$fails,$gbs,$gtxt,$xid,$rss,$v" >> "$OUT/cards.csv"
        log "card $g: $v | mmvq exit $rc, checks $pass pass / $fails fail, $gbs GB/s | greedy $gtxt | Xid $xid | mmvq peak RSS $rss kB"
        [ $v = PASS ] || { fail_all "card $g failed at memory +$MEM MT/s and core +$CORE MHz"; return 1; }
    done
    log "cards: every card PASS"
}

soak_stop() {  # soak_stop "what failed": record it, every offset to 0 at once, the other loads stopped
    echo "$1" >> "$OUT/SOAK_FAIL"
    touch "$WORK/STOP_SOAK"
    reset_all
    pkill -TERM -x mmvq_pascal
}

soak_card() {  # soak_card CARD END: the mmvq sweep on one card, round after round
    local g=$1 end=$2 i=0 t0 rc pass fails rss gbs v
    while [ "$(date +%s)" -lt "$end" ]; do
        tripped || [ -e "$WORK/STOP_SOAK" ] && break
        i=$((i + 1))
        t0=$(tnow)
        read -r rc pass fails rss gbs < <(mmvq "$g" "soak-card$g-$i")
        v=PASS
        [ "$rc" = 0 ] && [ "$fails" = 0 ] && [ "$pass" -gt 0 ] && [ -n "$gbs" ] || v=FAIL
        echo "${t0%.*},$(date '+%F %T'),$g,$i,$rc,$pass,$fails,$gbs,$rss,$v" >> "$OUT/soak_cards.csv"
        if [ $v = FAIL ]; then
            sleep 3
            tripped || [ -e "$WORK/STOP_SOAK" ] || soak_stop "card $g round $i: mmvq exit $rc, checks $pass pass / $fails fail"
            break
        fi
    done
}

soak_greedy() {  # soak_greedy END: the greedy check, again and again
    local end=$1 i=0 t0 rc
    while [ "$(date +%s)" -lt "$end" ]; do
        tripped || [ -e "$WORK/STOP_SOAK" ] && break
        i=$((i + 1))
        t0=$(tnow)
        greedy "soak-$i"; rc=$?
        echo "${t0%.*},$(date '+%F %T'),$i,$rc,$(cat "$OUT/greedy-soak-$i.sha" 2>/dev/null)" >> "$OUT/soak_greedy.csv"
        if [ $rc != 0 ]; then
            sleep 3
            tripped || [ -e "$WORK/STOP_SOAK" ] || soak_stop "greedy round $i: exit $rc ($([ $rc = 1 ] && echo "another hash" || echo "a request failed"))"
            break
        fi
    done
}

soak() {  # soak MIN
    local end t0 g
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { log "soak: the server is not running"; return 1; }
    rm -f "$WORK/STOP_SOAK" "$OUT/SOAK_FAIL"
    [ -s "$OUT/soak_cards.csv" ] || echo "start,end,card,round,mmvq_exit,checks_pass,checks_fail,stream_gbs,peak_rss_kb,verdict" > "$OUT/soak_cards.csv"
    [ -s "$OUT/soak_greedy.csv" ] || echo "start,end,round,exit,hash" > "$OUT/soak_greedy.csv"
    t0=$(date '+%F %T')
    end=$(($(date +%s) + ${1:?minutes} * 60))
    log "soak: $1 min, one mmvq loop on each of the cards $CARDS and one greedy loop"
    for g in $CARDS; do soak_card "$g" "$end" & sleep 2; done
    soak_greedy "$end" &
    wait
    local xid
    xid=$(xid_since "$t0")
    log "soak: ended | mmvq rounds $(($(wc -l < "$OUT/soak_cards.csv") - 1)), of which FAIL $(grep -c ',FAIL$' "$OUT/soak_cards.csv") | greedy rounds $(($(wc -l < "$OUT/soak_greedy.csv") - 1)) | Xid $xid"
    if tripped; then fail_all "soak: the watchdog tripped ($(tail -n 1 "$WORK/TRIPPED"))"; return 9; fi
    if [ -s "$OUT/SOAK_FAIL" ]; then fail_all "soak: $(head -n 1 "$OUT/SOAK_FAIL")"; return 1; fi
    if [ "$xid" != 0 ]; then fail_all "soak: $xid Xid lines"; return 1; fi
    log "soak: PASS"
}

memtest_card() {  # memtest_card CARD
    local g=$1 rc rc6 b b6
    timeout 900 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" "$KIT/bin/cuda_memtest" --device 0 \
        --num_passes 1 --exit_on_error --disable_test 6 > "$OUT/memtest_card$g.log" 2>&1
    rc=$?
    b=$(grep -c 'errors found in block' "$OUT/memtest_card$g.log")
    tripped && { echo "$g,$rc,$b,,,tripped" >> "$OUT/memtest.csv"; return 9; }
    timeout 900 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" "$KIT/bin/cuda_memtest" --device 0 \
        --num_passes 1 --exit_on_error --disable_all --enable_test 6 > "$OUT/memtest6_card$g.log" 2>&1
    rc6=$?
    b6=$(grep -c 'errors found in block' "$OUT/memtest6_card$g.log")
    echo "$g,$rc,$b,$rc6,$b6,$(date '+%F %T')" >> "$OUT/memtest.csv"
}

post() {
    local t0 g bad rc
    for g in $CARDS; do [ "$(readback "$g")" = "$MEM $CORE" ] || { fail_all "post: card $g is not at its offsets"; return 1; }; done
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || server start || { fail_all "post: the server did not start"; return 1; }
    greedy final; rc=$?
    log "post: greedy check after the soak: $([ $rc = 0 ] && echo "same hash as stock" || echo "exit $rc")"
    [ $rc = 0 ] || { fail_all "post: the greedy check after the soak gave another hash (exit $rc)"; return 1; }
    server stop
    t0=$(date '+%F %T')
    echo "card,exit_0_5_7_8_10,blocks_with_errors,exit_6,blocks_with_errors_6,end" > "$OUT/memtest.csv"
    log "post: cuda_memtest on every card at once"
    for g in $CARDS; do memtest_card "$g" & sleep 1; done
    wait
    bad=$(awk -F, 'NR > 1 && ($2 != 0 || $3 != 0 || $4 != 0 || $5 != 0)' "$OUT/memtest.csv" | wc -l)
    log "post: cuda_memtest: $(($(wc -l < "$OUT/memtest.csv") - 1)) cards, $bad with an error or a non-zero exit, Xid $(xid_since "$t0")"
    tripped && { fail_all "post: the watchdog tripped during cuda_memtest"; return 9; }
    [ "$bad" = 0 ] && [ "$(xid_since "$t0")" = 0 ] || { fail_all "post: cuda_memtest found errors (memtest.csv)"; return 1; }
    t0=$(date '+%F %T')
    exact kld "m$MEM-c$CORE" > "$OUT/exact_kld_console.log" 2>&1; rc=$?
    log "post: exact check at the offsets: exit $rc ($([ $rc = 0 ] && echo PASS || echo FAIL)), Xid $(xid_since "$t0")"
    tripped && { fail_all "post: the watchdog tripped during the exact check"; return 9; }
    [ $rc = 0 ] && [ "$(xid_since "$t0")" = 0 ] || { fail_all "post: the exact check failed"; return 1; }
    for g in $CARDS; do log "post: card $g offsets (mem core) $(readback "$g")"; done
    log "post: PASS"
}

hottest() {  # hottest [MIN]
    local n=$(( ${1:-10} * 120 + 5 ))
    tail -n "$n" "$WORK/wd.csv" | awk -F, '
        /^#/ || $1 == "time" { next }
        { n++; for (i = 0; i < 9; i++) { c = $(5 + i * 8 + 5); w = $(5 + i * 8 + 3)
              if (c > tc[i]) tc[i] = c; if (w > tw[i]) tw[i] = w } t1 = $1 }
        END { h = 0; for (i = 0; i < 9; i++) { if (tc[i] > tc[h]) h = i; printf "card %d: %d C, %.1f W | ", i, tc[i], tw[i] }
              printf "\nhottest card %d at %d C (%d samples to %s)\n", h, tc[h], n, t1 }'
}

detached() {  # detached STEP ARGS...: run "$self STEP ARGS" with a DONE marker, surviving the ssh session
    local step=$1
    rm -f "$OUT/${step^^}_DONE"
    setsid nohup bash -c "bash '$self' run-$step ${*:2} > '$OUT/$step.console' 2>&1; echo \"exit \$?\" > '$OUT/${step^^}_DONE'" \
        > /dev/null 2>&1 < /dev/null &
    echo "$step started detached (pid $!), marker $OUT/${step^^}_DONE"
}

case "${1:-}" in
state) state "${2:-}" ;;
trip-test) trip_test ;;
watchdog) watchdog ;;
stop-watchdog) stop_watchdog ;;
server) server "${2:?start|stop}" ;;
reset-all) reset_all ;;
hottest) hottest "${2:-10}" ;;
pre | cards | soak | post) detached "$@" ;;
run-pre) pre ;;
run-cards) cards ;;
run-soak) soak "${2:?minutes}" ;;
run-post) post ;;
*) sed -n '2,36p' "$0"; exit 2 ;;
esac
