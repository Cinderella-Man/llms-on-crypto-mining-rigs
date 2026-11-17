#!/usr/bin/env bash
# final_checks.sh: the long checks of a clock-offset sitting (Q133/Q135), on card $GPU with the offsets already set and
# no other process on the card. Run as root on the rig, with the watchdog running, detached:
#   setsid nohup bash final_checks.sh reference > /dev/null 2>&1 < /dev/null &
#
#   final_checks.sh reference      at offset 0: the exact check's base file (4,096 one-token steps over wikitext-2 on
#                                  the cards $DEVS) and the stock rerun whose statistics block is the reference
#   final_checks.sh memtest TAG    cuda_memtest on card $GPU at the offsets in place: one pass of tests 0-5, 7, 8 and
#                                  10, then test 6 alone (each at most 15 min)
#   final_checks.sh memtest-at MEM TAG   one pass of tests 0-5, 7, 8 and 10 with the memory offset at MEM and the core at
#                                  0 (does cuda_memtest see a step the other checks failed?), then the offsets back to 0
#   final_checks.sh exact TAG      the exact check at the offsets in place: PASS when its statistics block equals the
#                                  stock rerun's, digit for digit
# Each writes one line to $OUT/final.log (exit, memtest blocks with errors, Xid lines since its start) and a DONE marker.
#
# Env [defaults]: WORK [/tmp/q133], KIT [$WORK/mem_offset-kit-20261024], OUT [$WORK/out], GPU [0], BIN (llama.cpp bin
#   directory), MODEL (GGUF), WIKI (wikitext-2's wiki.test.raw), DEVS [0,1,2,3,4,5,6,7,8]
set -uo pipefail
WORK=${WORK:-/tmp/q133}
KIT=${KIT:-$WORK/mem_offset-kit-20261024}
OUT=${OUT:-$WORK/out}
GPU=${GPU:-0}
DEVS=${DEVS:-0,1,2,3,4,5,6,7,8}
P="python3 $KIT/nvml_memoc.py"
mkdir -p "$OUT"
note() { echo "$(date '+%F %T') $*" | tee -a "$OUT/final.log"; }
offsets() { echo "mem $($P get "$GPU" | head -n 1 | cut -d' ' -f6) core $($P --core get "$GPU" | head -n 1 | cut -d' ' -f6)"; }
xid_since() { journalctl -k -q --since "$1" | grep -ci 'NVRM: Xid'; }
memtest() {  # memtest LOG ARGS...
    local log=$1; shift
    timeout 900 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$KIT/bin/cuda_memtest" --device 0 \
        --num_passes 1 --exit_on_error "$@" > "$log" 2>&1
}
exact() { B=$BIN M=$MODEL WIKI=$WIKI DEVS=$DEVS E=CUDA_DEVICE_ORDER=PCI_BUS_ID OUT=$OUT bash "$KIT/exact_check.sh" "$@"; }
[ -e "$WORK/TRIPPED" ] && { note "the watchdog has tripped: nothing run"; exit 1; }

case "${1:-}" in
reference)
    t0=$(date '+%F %T')
    exact base > "$OUT/base_console.log" 2>&1; rc=$?
    note "reference at $(offsets): exit $rc, xid $(xid_since "$t0")"
    echo "exit $rc" > "$OUT/REFERENCE_DONE" ;;
memtest)
    tag=${2:?tag}
    t0=$(date '+%F %T')
    memtest "$OUT/memtest_$tag.log" --disable_test 6; rc=$?
    note "memtest tests 0-5,7,8,10 at $(offsets): exit $rc, blocks with errors $(grep -c 'errors found in block' "$OUT/memtest_$tag.log"), xid $(xid_since "$t0")"
    [ -e "$WORK/TRIPPED" ] && { note "the watchdog tripped"; exit 1; }
    t0=$(date '+%F %T')
    memtest "$OUT/memtest6_$tag.log" --disable_all --enable_test 6; rc=$?
    note "memtest test 6 at $(offsets): exit $rc, blocks with errors $(grep -c 'errors found in block' "$OUT/memtest6_$tag.log"), xid $(xid_since "$t0")"
    echo done > "$OUT/MEMTEST_DONE" ;;
memtest-at)
    mem=${2:?MT/s} tag=${3:?tag}
    $P --core reset "$GPU" > /dev/null
    $P --cap "$mem" set "$GPU" "$mem" > /dev/null
    t0=$(date '+%F %T')
    memtest "$OUT/memtest_$tag.log" --disable_test 6; rc=$?
    note "memtest tests 0-5,7,8,10 at $(offsets): exit $rc, blocks with errors $(grep -c 'errors found in block' "$OUT/memtest_$tag.log"), xid $(xid_since "$t0")"
    $P reset "$GPU" > /dev/null
    note "offsets back to $(offsets)"
    echo done > "$OUT/MEMTEST_AT_DONE" ;;
exact)
    tag=${2:?tag}
    t0=$(date '+%F %T')
    exact kld "$tag" > "$OUT/kld_${tag}_console.log" 2>&1; rc=$?
    note "exact check at $(offsets): exit $rc ($([ $rc = 0 ] && echo PASS || echo FAIL)), xid $(xid_since "$t0")"
    echo "exit $rc" > "$OUT/EXACT_DONE" ;;
*) sed -n '2,20p' "$0"; exit 2 ;;
esac
