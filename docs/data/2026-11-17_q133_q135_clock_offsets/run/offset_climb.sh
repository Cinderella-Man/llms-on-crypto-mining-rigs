#!/usr/bin/env bash
# offset_climb.sh: one step of a memory or core clock offset climb on one P106 card, with the checks of Q133/Q135
# (README.md next to this folder). Run as root on the rig, with scripts/nvml_watchdog.py already running and writing
# $WORK/wd.csv (it trips to $WORK/TRIPPED), and the memory-offset kit (bench/mem_offset) unpacked in $KIT.
#
#   offset_climb.sh server start|stop   llama-server with $MODEL on the cards $DEVS, greedy only (port $PORT)
#   offset_climb.sh step mem|core X     set the offset (memory in MT/s, core in MHz), read it back from a new process,
#                                       run the mmvq sweep on the card (correctness check + streaming GB/s), the greedy
#                                       check against the server (when it runs), an Xid scan; one row in steps.csv;
#                                       on any failure the offset goes back to 0 at once
#   offset_climb.sh soak mem|core X MIN  repeat {mmvq sweep, greedy check} at X for MIN minutes, one row a round
#   offset_climb.sh window T0 T1        the card's peaks in the watchdog log between two times
#
# Env [defaults]: GPU [0], WORK [/tmp/q133], KIT [$WORK/mem_offset-kit-20261024], OUT [$WORK/out], BIN (llama.cpp bin
#   directory), MODEL (GGUF), DEVS [0,1,2,3,4,5,6,7,8], PORT [8133], REF [ref: the greedy reference tag],
#   MEM_CAP [960] (MT/s, +12 % of 8,008), CORE_CAP [150] (MHz), MMVQ_ARGS [the kit's five types].
set -uo pipefail
GPU=${GPU:-0}
WORK=${WORK:-/tmp/q133}
KIT=${KIT:-$WORK/mem_offset-kit-20261024}
OUT=${OUT:-$WORK/out}
DEVS=${DEVS:-0,1,2,3,4,5,6,7,8}
PORT=${PORT:-8133}
REF=${REF:-ref}
MEM_CAP=${MEM_CAP:-960}
CORE_CAP=${CORE_CAP:-150}
MMVQ_ARGS=${MMVQ_ARGS:---types q8_0,q5_0,q4_1,q5_K,iq4_xs}
here=$(cd "$(dirname "$0")" && pwd)
PY="python3 $KIT/nvml_memoc.py"
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/climb.log"; }
tnow() { date '+%F %T.%3N'; }

window() {  # window T0 T1: peaks of the card under test (and the hottest card) in the watchdog log
    awk -F, -v t0="$1" -v t1="$2" -v g="$GPU" '
        /^#/ || $1 == "time" { next }
        $1 >= t0 && $1 <= t1 {
            b = 5 + g * 8; n++
            if ($(b+1) > sm) sm = $(b+1); if ($(b+2) > mem) mem = $(b+2)
            if ($(b+3) > w) w = $(b+3); if ($(b+5) > c) c = $(b+5)
            for (i = 0; i * 8 + 5 <= NF; i++) if ($(5 + i * 8 + 5) > hot) hot = $(5 + i * 8 + 5)
            if ($(b+3) > 50) w50++
        }
        END { printf "%d,%d,%d,%.2f,%d,%d,%d\n", n, sm, mem, w, c, hot, w50 }' "$WORK/wd.csv"
}

set_offset() {  # set_offset mem|core X
    local dom=$1 x=$2 flag=""
    [ "$dom" = core ] && flag="--core"
    if [ "$x" = 0 ]; then $PY $flag reset "$GPU" >> "$OUT/climb.log" 2>&1
    elif [ "$dom" = mem ]; then $PY --cap "$MEM_CAP" set "$GPU" "$x" >> "$OUT/climb.log" 2>&1
    else $PY --core set "$GPU" "$x" >> "$OUT/climb.log" 2>&1
    fi
}

readback() { local f=""; [ "$1" = core ] && f=--core; $PY $f get "$GPU" 2>&1 | head -n 1 | sed "s/.*offset: //"; }

server() {
    case "$1" in
    start)
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVS" setsid nohup "$BIN/llama-server" -m "$MODEL" -ngl 99 \
            -fa on -c 4096 -np 1 -t 2 --host 127.0.0.1 --port "$PORT" --no-warmup > "$OUT/server.log" 2>&1 < /dev/null &
        log "server: pid $!, $MODEL on cards $DEVS"
        for _ in $(seq 1 240); do
            sleep 3
            [ -e "$WORK/TRIPPED" ] && { log "server: watchdog tripped during load"; return 9; }
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { log "server: ready"; return 0; }
            pgrep -x llama-server > /dev/null || { log "server: exited during load (see server.log)"; return 1; }
        done
        log "server: not ready after 12 min"; return 1 ;;
    stop)
        pkill -TERM -x llama-server; sleep 3; pkill -KILL -x llama-server 2>/dev/null; log "server: stopped" ;;
    esac
}

one_round() {  # one_round TAG: mmvq + greedy at the current offset; sets R_* and returns 0 when both pass
    local tag=$1
    R_MMVQ_RC=0 R_FAILS=0 R_KCHK=0 R_GBS="" R_LDG="" R_GREEDY="not run"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$KIT/bin/mmvq_pascal" $MMVQ_ARGS \
        --csv "$OUT/mmvq_$tag.csv" > "$OUT/mmvq_$tag.log" 2>&1
    R_MMVQ_RC=$?
    R_GBS=$(awk -F, '$3 == "STREAM_READ_PLAIN" { print $15 }' "$OUT/mmvq_$tag.csv" 2>/dev/null)
    R_LDG=$(awk -F, '$3 == "STREAM_READ_LDG" { print $15 }' "$OUT/mmvq_$tag.csv" 2>/dev/null)
    R_FAILS=$(awk -F, '$23 == "FAIL"' "$OUT/mmvq_$tag.csv" 2>/dev/null | wc -l)
    R_KCHK=$(awk -F, '$23 == "PASS"' "$OUT/mmvq_$tag.csv" 2>/dev/null | wc -l)
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        if [ "$tag" = "$REF" ]; then
            python3 "$here/greedy_check.py" --url "http://127.0.0.1:$PORT" --out "$OUT" --tag "$tag" >> "$OUT/climb.log" 2>&1
            R_GREEDY=$([ $? = 0 ] && echo reference || echo ERROR)
        else
            python3 "$here/greedy_check.py" --url "http://127.0.0.1:$PORT" --out "$OUT" --tag "$tag" --ref "$REF" \
                >> "$OUT/climb.log" 2>&1
            case $? in 0) R_GREEDY=same ;; 1) R_GREEDY=DIFFERS ;; *) R_GREEDY=ERROR ;; esac
        fi
        tail -n 1 "$OUT/climb.log"
    fi
    [ "$R_MMVQ_RC" = 0 ] && [ "$R_FAILS" = 0 ] && [ -n "$R_GBS" ] && [ "$R_KCHK" -gt 0 ] \
        && { [ "$R_GREEDY" = same ] || [ "$R_GREEDY" = reference ] || [ "$R_GREEDY" = "not run" ]; }
}

step() {  # step mem|core X [TAG]
    local dom=$1 x=$2 tag=${3:-$1$2} t0 t1 rb xid ok verdict
    [ -e "$WORK/TRIPPED" ] && { log "step $tag: the watchdog has tripped ($(tail -n 1 "$WORK/TRIPPED")): no step"; return 9; }
    if [ "$dom" = mem ] && [ "$x" -gt "$MEM_CAP" ]; then log "step $tag: +$x MT/s is above the cap $MEM_CAP"; return 2; fi
    if [ "$dom" = core ] && [ "$x" -gt "$CORE_CAP" ]; then log "step $tag: +$x MHz is above the cap $CORE_CAP"; return 2; fi
    t0=$(tnow)
    set_offset "$dom" "$x" || { log "step $tag: set REFUSED"; set_offset "$dom" 0; return 1; }
    rb=$(readback "$dom")
    log "step $tag: $dom offset set to $x, read back from a new process: $rb"
    one_round "$tag"; ok=$?
    xid=$(journalctl -k -q --since "${t0%.*}" 2>/dev/null | grep -ci 'NVRM: Xid')
    t1=$(tnow)
    [ "$xid" = 0 ] || ok=1
    [ -e "$WORK/TRIPPED" ] && ok=1
    verdict=$([ $ok = 0 ] && echo PASS || echo FAIL)
    [ -s "$OUT/steps.csv" ] || echo "time,dom,offset,readback,mmvq_exit,checks_pass,checks_fail,stream_gbs,stream_ldg_gbs,greedy,xid,samples,g_sm_max_mhz,g_mem_max_mhz,g_w_max,g_c_max,all_c_max,g_samples_over_50w,verdict" > "$OUT/steps.csv"
    echo "${t0%.*},$dom,$x,\"$rb\",$R_MMVQ_RC,$R_KCHK,$R_FAILS,$R_GBS,$R_LDG,$R_GREEDY,$xid,$(window "$t0" "$t1"),$verdict" >> "$OUT/steps.csv"
    log "step $tag: $verdict | mmvq exit $R_MMVQ_RC, checks $R_KCHK pass / $R_FAILS fail, streaming $R_GBS GB/s (ldg $R_LDG) | greedy $R_GREEDY | Xid $xid | peaks (samples,sm,mem,W,C,hottest C,>50W): $(window "$t0" "$t1")"
    if [ $ok != 0 ]; then set_offset "$dom" 0; log "step $tag: $dom offset back to 0 ($(readback "$dom"))"; fi
    return $ok
}

soak() {  # soak mem|core X MIN
    local dom=$1 x=$2 end i=0 ok=0
    step "$dom" "$x" "soak-$dom$x-0" || return 1
    end=$(($(date +%s) + $3 * 60))
    while [ "$(date +%s)" -lt "$end" ]; do
        [ -e "$WORK/STOP_SOAK" ] && { log "soak: STOP_SOAK found"; break; }
        i=$((i + 1))
        local t0 t1 xid
        t0=$(tnow)
        one_round "soak-$dom$x-$i" || ok=1
        xid=$(journalctl -k -q --since "${t0%.*}" 2>/dev/null | grep -ci 'NVRM: Xid')
        t1=$(tnow)
        [ "$xid" = 0 ] || ok=1
        [ -e "$WORK/TRIPPED" ] && ok=1
        echo "${t0%.*},$dom,$x,soak round $i,$R_MMVQ_RC,$R_KCHK,$R_FAILS,$R_GBS,$R_LDG,$R_GREEDY,$xid,$(window "$t0" "$t1"),$([ $ok = 0 ] && echo PASS || echo FAIL)" >> "$OUT/steps.csv"
        log "soak $dom +$x round $i: $([ $ok = 0 ] && echo PASS || echo FAIL) | checks $R_KCHK/$R_FAILS fail, $R_GBS GB/s, greedy $R_GREEDY, Xid $xid | peaks $(window "$t0" "$t1")"
        [ $ok = 0 ] || { set_offset "$dom" 0; log "soak: FAIL: $dom offset back to 0 ($(readback "$dom"))"; return 1; }
    done
    log "soak $dom +$x: $i rounds clean"
}

case "${1:-}" in
server) server "$2" ;;
step) step "$2" "$3" "${4:-}" ;;
soak) soak "$2" "$3" "$4" ;;
window) window "$2" "$3" ;;
*) sed -n '2,19p' "$0"; exit 2 ;;
esac
