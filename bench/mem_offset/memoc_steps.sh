#!/usr/bin/env bash
# memoc_steps.sh: memory clock offset test on one card of the P106 rig (bench/mem_offset/README.md). Run as root on the
# rig, from the unpacked kit, with nothing else on the card under test.
#
#   memoc_steps.sh probe    read-only, ~1 min: NVML offsets and their ranges, clocks, PCIe link, CPU settings
#   memoc_steps.sh all      units, base, sweep, memtest, exact, soak (~1 h 45 min with the 60-minute soak)
#   memoc_steps.sh STAGE    one stage: units | base | sweep | memtest | exact | soak (they share state in $OUT)
#
#   units    NVML's unit for the offset, measured on the card: a raw +100 under the mmvq load, then back to 0
#   base     at offset 0: the KLD base file, the stock rerun (reference statistics) and the stock greedy hash
#   sweep    +STEP MT/s steps up to MAX, each with the mmvq sweep (correctness check, streaming GB/s) and an Xid scan;
#            the first failure ends it and the last clean step becomes BEST ($OUT/best)
#   memtest  cuda_memtest at BEST, one pass of tests 0-5, 7, 8 and 10, at most MEMTEST_MIN minutes (test 6, the 32-bit
#            moving inversions, takes ~70 % of a pass, est. ~18 min on a P106: MEMTEST_ARGS="--disable_all --enable_test 6"
#            runs it alone)
#   exact    KLD statistics and greedy hash at BEST against stock
#   soak     llama-bench (512-token prompt + 128 generated, repeated) at BEST for SOAK_MIN minutes on the cards DEVS,
#            Xid scan, temperature and power every minute; then the KLD check again
#   In `all`, a memtest or exact failure at BEST steps back one STEP and tests again.
#
# The offset goes back to 0 after every stage and whenever the script ends, fails or is interrupted (Ctrl-C, TERM,
# HUP: a trap), and `touch $OUT/STOP` ends the run cleanly at the next check. If the card hangs, the reboot clears it.
#
# Env [defaults]: GPU [0], STEP [200], MAX [1200] (MT/s), MIN_RISE [0.5] (% the streaming read must gain per step),
#   MMVQ_ARGS, MEMTEST_MIN [10], MEMTEST_ARGS [--disable_test 6], SOAK_MIN [60], TEMP_MAX [70] (°C, GPU), OUT [out/ next to this script], FORCE [0: refuse
#   if a compute process is on the card], and exact_check.sh's B, M, WIKI, DEVS, E.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}
STEP=${STEP:-200}
MAX=${MAX:-1200}
MIN_RISE=${MIN_RISE:-0.5}
MMVQ_ARGS=${MMVQ_ARGS:---types q8_0,q5_0,q4_1,q5_K,iq4_xs}
MEMTEST_MIN=${MEMTEST_MIN:-10}
MEMTEST_ARGS=${MEMTEST_ARGS:---disable_test 6}
SOAK_MIN=${SOAK_MIN:-60}
TEMP_MAX=${TEMP_MAX:-70}
OUT=${OUT:-$here/out}
FORCE=${FORCE:-0}
export B=${B:-/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin}
export M=${M:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
export WIKI=${WIKI:-/opt/llama-builds/data/wiki.test.raw}
export DEVS=${DEVS:-0,1,2,3,4,5}
export E=${E:-LLAMA_NO_MOE_DOWN_RESCALE=tg}
export OUT
PY="python3 $here/nvml_memoc.py"
MMVQ=$here/bin/mmvq_pascal
MEMTEST=$here/bin/cuda_memtest
mkdir -p "$OUT"
LOG=$OUT/steps.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
on_gpu() { CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU "$@"; }

CHILD=""
SMI_PIDS=""
BENCH_PID=""
killtree() { local p; for p in $(pgrep -P "$1"); do killtree "$p"; done; kill "$1" 2>/dev/null; }
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    for p in $CHILD $BENCH_PID; do killtree "$p"; done
    sleep 2  # let the CUDA processes exit before the offset changes under them
    if $PY reset "$GPU" >> "$LOG" 2>&1; then
        log "offset of gpu $GPU back to 0"
    else
        log "WARNING: resetting gpu $GPU failed; run: python3 $here/nvml_memoc.py reset $GPU"
    fi
    for p in $SMI_PIDS; do kill "$p" 2>/dev/null; done
    exit "$rc"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

# child CMD...: runs CMD in the background and waits for it, so that a signal reaches the trap at once
child() {
    "$@" &
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return "$rc"
}
# logged CMD...: the same, with the output on the console and in the log; the status is CMD's
logged() {
    { "$@" 2>&1 | tee -a "$LOG"; exit "${PIPESTATUS[0]}"; } &
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return "$rc"
}
pause() { sleep "$1" & CHILD=$!; wait "$CHILD"; CHILD=""; }
stopped() { [ -e "$OUT/STOP" ] && log "STOP file found" ; [ -e "$OUT/STOP" ]; }

smi_start() {  # GPU every 500 ms (clocks, power, temperature, event reasons), the cards DEVS every 2 s
    [ -n "$SMI_PIDS" ] && return
    nvidia-smi -i "$GPU" --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv,noheader,nounits -lms 500 >> "$OUT/smi.csv" 2>&1 &
    SMI_PIDS=$!
    nvidia-smi -i "$DEVS" --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu \
        --format=csv,noheader,nounits -l 2 >> "$OUT/smi-cards.csv" 2>&1 &
    SMI_PIDS="$SMI_PIDS $!"
    sleep 1
}
smi_lines() { wc -l < "$OUT/smi.csv"; }
smi_max() {  # smi_max FIELD FROM_LINE: the largest value of a smi.csv field since a line (4 = memory clock, 6 = temperature)
    tail -n +"$(($2 + 1))" "$OUT/smi.csv" | awk -F', ' -v f="$1" '$f + 0 > m { m = $f + 0 } END { print m + 0 }'
}

XID_N0=0
XID_T0=0
xid_mark() { XID_N0=$(dmesg 2>/dev/null | wc -l); XID_T0=$(date +%s); }
xid_scan() {  # status 1 and a log line if the kernel logged an NVRM Xid since the mark
    local new
    new=$({ dmesg 2>/dev/null | tail -n +"$((XID_N0 + 1))"; journalctl -k -q --since "@$XID_T0" 2>/dev/null; } |
        grep -i 'NVRM: Xid' | sort -u)
    [ -z "$new" ] && return 0
    log "Xid since the stage began:"; echo "$new" | tee -a "$LOG"
    return 1
}

set_offset() {  # set_offset MTS
    if [ "$1" = 0 ]; then $PY reset "$GPU" >> "$LOG" 2>&1; else $PY set "$GPU" "$1" >> "$LOG" 2>&1; fi
    local rc=$?
    log "offset +$1 MT/s on gpu $GPU: $([ $rc = 0 ] && echo set || echo REFUSED) | $($PY clocks "$GPU" | cut -d' ' -f3-)"
    return $rc
}

preflight() {
    [ "$(id -u)" = 0 ] || { echo "run as root: NVML offsets need it"; exit 2; }
    [ -x "$MMVQ" ] && [ -x "$MEMTEST" ] || { echo "bin/mmvq_pascal or bin/cuda_memtest missing"; exit 2; }
    "$MMVQ" --list > /dev/null 2>&1 && "$MEMTEST" --list_tests > /dev/null 2>&1 \
        || { echo "bin/mmvq_pascal or bin/cuda_memtest does not start on this machine"; exit 2; }
    local apps
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
        echo "compute processes on gpu $GPU (set FORCE=1 to go on anyway):"; echo "$apps"; exit 2
    fi
    log "memoc_steps.sh $* on gpu $GPU: STEP $STEP, MAX $MAX, MIN_RISE $MIN_RISE %, MEMTEST_MIN $MEMTEST_MIN, SOAK_MIN $SOAK_MIN, OUT $OUT"
}

stage_units() {
    log "units: raw +100 under the mmvq load, then 0"
    logged $PY units "$GPU" -- env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$MMVQ" --types q8_0 \
        --shapes attn_q,moe_gate_up --reps 60
}

stage_base() {
    set_offset 0 || return 1
    log "base: KLD base file, stock rerun, stock greedy hash (offset 0)"
    child env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$MEMTEST" --device 0 --max_num_blocks 64 --num_passes 1 \
        --exit_on_error > "$OUT/memtest_smoke.log" 2>&1
    local rc=$?
    log "base: cuda_memtest on 64 MB at offset 0: exit $rc, $(grep -c 'ERROR' "$OUT/memtest_smoke.log") ERROR lines"
    [ "$rc" = 0 ] && ! grep -q 'ERROR' "$OUT/memtest_smoke.log" || { log "base: cuda_memtest fails at stock: stop"; return 1; }
    logged bash "$here/exact_check.sh" base || return 1
    logged bash "$here/exact_check.sh" greedy stock
}

mmvq_step() {  # mmvq_step MTS: the bench at the current offset; sets GBS; status 1 on any failure
    local off=$1 n0 rc fails ldg clk
    n0=$(smi_lines)
    child env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$MMVQ" $MMVQ_ARGS --csv "$OUT/mmvq_$off.csv" \
        > "$OUT/mmvq_$off.log" 2>&1
    rc=$?
    GBS=$(awk -F, '$3 == "STREAM_READ_PLAIN" { print $15 }' "$OUT/mmvq_$off.csv" 2>/dev/null)
    ldg=$(awk -F, '$3 == "STREAM_READ_LDG" { print $15 }' "$OUT/mmvq_$off.csv" 2>/dev/null)
    fails=$(awk -F, '$23 == "FAIL"' "$OUT/mmvq_$off.csv" 2>/dev/null | wc -l)
    clk=$(smi_max 4 "$n0")
    log "+$off MT/s: mmvq exit $rc, failed checks $fails, streaming read ${GBS:-?} GB/s (ldg ${ldg:-?}), memory clock under load $clk MHz"
    echo "$off,$rc,$fails,${GBS:-},${ldg:-},$clk" >> "$OUT/sweep.csv"
    xid_scan || return 1
    [ "$rc" = 0 ] && [ "$fails" = 0 ] && [ -n "$GBS" ]
}

stage_sweep() {
    local best=0 prev off rise
    xid_mark; smi_start
    set_offset 0 || return 1
    echo "offset_mts,mmvq_exit,failed_checks,stream_gbs,stream_ldg_gbs,mem_clock_mhz" > "$OUT/sweep.csv"
    mmvq_step 0 || { log "sweep: the bench fails at offset 0: stop"; return 1; }
    prev=$GBS
    for ((off = STEP; off <= MAX; off += STEP)); do
        stopped && break
        set_offset "$off" || { log "sweep: +$off refused: the end of the allowed range"; break; }
        if ! mmvq_step "$off"; then
            log "sweep: +$off FAILED: falling back to +$best"
            break
        fi
        rise=$(awk -v a="$GBS" -v b="$prev" 'BEGIN { printf "%.2f", (a / b - 1) * 100 }')
        if awk -v r="$rise" -v m="$MIN_RISE" 'BEGIN { exit !(r < m) }'; then
            log "sweep: +$off raised the streaming read by only $rise % (< $MIN_RISE %): treated as a failure, falling back to +$best"
            break
        fi
        log "sweep: +$off clean (streaming read +$rise %)"
        best=$off; prev=$GBS
    done
    set_offset 0
    echo "$best" > "$OUT/best"
    log "sweep: BEST +$best MT/s"
    [ "$best" -gt 0 ]
}

best() { cat "$OUT/best" 2>/dev/null || echo 0; }

stage_memtest() {
    local b rc errs n0
    b=$(best); [ "$b" -gt 0 ] || { log "memtest: no BEST step"; return 1; }
    xid_mark; smi_start
    set_offset "$b" || return 1
    n0=$(smi_lines)
    log "memtest: cuda_memtest at +$b for up to $MEMTEST_MIN min, one pass, $MEMTEST_ARGS"
    child timeout "$((MEMTEST_MIN * 60))" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$MEMTEST" --device 0 \
        --num_passes 1 --exit_on_error $MEMTEST_ARGS > "$OUT/memtest_$b.log" 2>&1
    rc=$?
    errs=$(grep -c 'errors found in block' "$OUT/memtest_$b.log")
    set_offset 0
    log "memtest +$b: exit $rc ($([ $rc = 124 ] && echo 'time limit' || echo 'pass done')), blocks with errors $errs, peak temperature $(smi_max 6 "$n0") C"
    xid_scan || return 1
    if [ "$errs" = 0 ] && [ "$rc" != 0 ] && [ "$rc" != 124 ]; then
        log "memtest +$b: cuda_memtest failed without reporting a memory error (see $OUT/memtest_$b.log): stop"
        return 2
    fi
    [ "$errs" = 0 ] && { [ "$rc" = 0 ] || [ "$rc" = 124 ]; }
}

stage_exact() {  # stage_exact [TAG]
    local b tag=${1:-best} ok=0
    b=$(best); [ "$b" -gt 0 ] || { log "exact: no BEST step"; return 1; }
    xid_mark
    set_offset "$b" || return 1
    logged bash "$here/exact_check.sh" kld "$tag-$b" || ok=1
    if [ "$tag" = best ]; then
        logged bash "$here/exact_check.sh" greedy "$tag-$b" || ok=1
    fi
    set_offset 0
    xid_scan || ok=1
    log "exact +$b ($tag): $([ $ok = 0 ] && echo PASS || echo FAIL)"
    return $ok
}

stage_soak() {
    local b end n0 i=0 temp
    b=$(best); [ "$b" -gt 0 ] || { log "soak: no BEST step"; return 1; }
    xid_mark; smi_start
    set_offset "$b" || return 1
    n0=$(smi_lines)
    log "soak: llama-bench pp512+tg128 on cards $DEVS at +$b for $SOAK_MIN min"
    env $E CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVS" "$B/llama-bench" -m "$M" -ngl 99 -fa on -p 0 -n 0 \
        -pg 512,128 -r 100000 > "$OUT/soak-bench.log" 2>&1 &
    BENCH_PID=$!
    end=$(($(date +%s) + SOAK_MIN * 60))
    while [ "$(date +%s)" -lt "$end" ]; do
        pause 60; i=$((i + 1))
        if ! kill -0 "$BENCH_PID" 2>/dev/null; then
            log "soak: llama-bench ended early (see $OUT/soak-bench.log): FAIL"; BENCH_PID=""; set_offset 0; return 1
        fi
        xid_scan || { set_offset 0; return 1; }
        temp=$(tail -n 1 "$OUT/smi.csv" | awk -F', ' '{ print $6 + 0 }')
        log "soak: minute $i, gpu $GPU at $(tail -n 1 "$OUT/smi.csv" | awk -F', ' '{ print $4 " MHz memory, " $5 " W, " $6 " C" }')"
        if [ "$temp" -gt "$TEMP_MAX" ]; then log "soak: $temp C is above $TEMP_MAX C: stop"; break; fi
        stopped && break
    done
    killtree "$BENCH_PID"; wait "$BENCH_PID" 2>/dev/null; BENCH_PID=""
    log "soak: $i min done, peak temperature $(smi_max 6 "$n0") C"
    set_offset 0
    [ "$i" -ge "$SOAK_MIN" ] || { log "soak: ended before $SOAK_MIN min"; return 1; }
    stage_exact soak
}

summary() {
    {
        echo "gpu $GPU, kit $here, $(date '+%F %T')"
        echo "sweep (offset MT/s, mmvq exit, failed checks, streaming GB/s, ldg GB/s, memory clock MHz):"
        sed 's/^/  /' "$OUT/sweep.csv" 2>/dev/null
        echo "BEST +$(best) MT/s"
        grep -hE 'memtest \+|exact \+|exact_check: (kld|greedy) .*(PASS|FAIL)|soak: [0-9]+ min done|units|NVML raw unit' "$LOG" | sed 's/^/  /'
    } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
}

stage=${1:-}
case "$stage" in
probe)
    $PY probe "$GPU" 2>&1 | tee -a "$LOG"; trap - EXIT; exit 0 ;;
units|base|sweep|memtest|exact|soak)
    preflight "$stage"; "stage_$stage"; rc=$?; summary; exit $rc ;;
all)
    preflight all
    stage_units || exit 1
    stage_base || exit 1
    stage_sweep || { summary; exit 1; }
    while :; do
        stage_memtest; rc=$?
        [ "$rc" = 2 ] && { summary; exit 1; }
        if [ "$rc" = 0 ] && stage_exact; then break; fi
        b=$(best)
        if [ "$b" -le "$STEP" ]; then log "all: no clean step left"; echo 0 > "$OUT/best"; summary; exit 1; fi
        echo "$((b - STEP))" > "$OUT/best"
        log "all: +$b failed memtest or exact: stepping back to +$((b - STEP))"
    done
    stage_soak; rc=$?
    summary
    log "all: done, BEST +$(best) MT/s, soak $([ $rc = 0 ] && echo PASS || echo FAIL); recommended setting: one step below the last clean step"
    exit $rc ;;
*)
    sed -n '2,27p' "$0"; trap - EXIT; exit 2 ;;
esac
