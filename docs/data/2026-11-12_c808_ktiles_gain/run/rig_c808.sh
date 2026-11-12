#!/usr/bin/env bash
# rig_c808.sh: C808 on a P106 rig: C728's K tiles only for Q4_K's 4-token products (from at most 3 warps a block of C722's
# kernel), two row blocks a round by default (GGML_CUDA_MMVQ_PASCAL_DMT_KT=1, _DMT_KT_M=2 by default), timed as R120b
# timed the tiles (2026-11-12_r120b_kern_speed). Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "perf alone served"):
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid dmtkt, every product against its columns' one-token products, bit
#           for bit, at _DMT_KT=1 (the new rule), _DMT_KT=2 (every tile, M = 2), _DMT_KT=2 _DMT_KT_M=1 and the fault
#           _DMT_KT=2 _DMT_KT_FAULT=1 (must differ); optional (R120a found the same tile kernels exact at M = 1 and 2)
#   perf    test-backend-ops perf at Qwen3.8-Flash-Next's IQ4_XS, Q4_K, Q5_K and Q6_K shapes, 1-8 tokens (R120b's cases),
#           the arms of PERF_ARMS alternated over PASSES passes: off (the preset's k-column switches), mt (+ the read-once
#           kernels _DMT=1 _Q5K_MT=1), kern (mt + _Q6K_MT=1 _DMT_KT=1 _RO8=1: the tiles now for Q4_K at 4 tokens only, at
#           M = 2), kernm1 (kern + _DMT_KT_M=1) and kernm2 (kern + _DMT_KT_M=2: kern's choices exactly, with one more
#           variable in the environment)
#   alone   the two products R120b saw move under _DMT_KT_M=2 with no tile (Q6_K 2,560 x 6,144 at 4 tokens, Q4_K at 5)
#           and the tile product (Q4_K at 4), each in a test-backend-ops process of its own (no tile launched before it
#           in the process), the arms of ALONE_ARMS alternated over PASSES passes. With kern, kernm1 and kernm2: a move
#           that follows the earlier tile launch in a process shows in perf and not here; one that follows the
#           environment shows kernm2 against kern in both
#   served  rig_kern.sh's served step (R49's profile of a served verify pass, 2026-11-10_served_verify_parts) on the
#           Miner Llama v3 preset with the MTP draft: both arms add _DMT=1 _Q5K_MT=1, "on" adds SERVED_ENV (default
#           GGML_CUDA_MMVQ_PASCAL_DMT_KT=1), loads in ORDER (default off on on off), each LOADS (default warm prof)
# Env: BIN (bin/ of the C808 rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID), PASSES (2),
#   PERF_ARMS (default "off mt kern kernm1 kernm2"), ALONE_ARMS (default "kern kernm1 kernm2"); bitid: BITID (the
#   multitoken_bitid binary built against the same tree); served: RIG_KERN (2026-11-11_v4_kernel_merge/run/rig_kern.sh),
#   MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh), REQ
#   (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF, ORDER, LOADS, SERVED_ENV
# Output: $OUT/perf/us_run.tsv and $OUT/alone/us_run.tsv (arm, pass, type, rows, K, tokens, us per product; the format of
#   rig_kern.sh's, for 2026-11-12_r120b_kern_speed/run/perf_table.py), $OUT/bitid/*.txt, $OUT/served/ (rig_kern.sh's)
# Usage: BIN=... OUT=... RIG_KERN=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... VERIFY_PROF=... \
#   setsid -f nohup bash rig_c808.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-perf alone served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off mt kern kernm1 kernm2}
ALONE_ARMS=${ALONE_ARMS:-kern kernm1 kernm2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" # the Flash-Next preset's k-column switches
MT="${P}_DMT=1 ${P}_Q5K_MT=1"                # the read-once kernels (C722, C690)
KERN="${P}_Q6K_MT=1 ${P}_DMT_KT=1 ${P}_RO8=1" # R120b's three merged switches; _DMT_KT=1 is C808's rule here
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)    echo "$KP" ;;
        mt)     echo "$KP $MT" ;;
        kern)   echo "$KP $MT $KERN" ;;
        kernm1) echo "$KP $MT $KERN ${P}_DMT_KT_M=1" ;;
        kernm2) echo "$KP $MT $KERN ${P}_DMT_KT_M=2" ;;
        *)      return 1 ;;
    esac
}
# one line per arm, pass, shape and token count: the case's us/run (rig_kern.sh's format)
us_table() { # <dir>: its <arm>-<pass>[-<case>].txt files into <dir>/us_run.tsv
    local f b
    for f in "$1"/*.txt; do
        b=$(basename "$f" .txt)
        b=${b%-c[0-9]*}
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(type_a=([^,]+),type_b=f32,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\4\t\3\t\5/"
    done | sort > "$1/us_run.tsv"
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    D="${P}_DMT"
    for a in "kt1:$KP $D=1 ${D}_KT=1" "kt2:$KP $D=1 ${D}_KT=2" "kt2m1:$KP $D=1 ${D}_KT=2 ${D}_KT_M=1" \
             "fault:$KP $D=1 ${D}_KT=2 ${D}_KT_FAULT=1"; do
        n=${a%%:*} sw=${a#*:}
        log "bitid dmtkt $n start ($sw)"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" $sw "$BITID" dmtkt > "$OUT/bitid/dmtkt-$n.txt" 2>&1
        log "bitid dmtkt $n exit $?: $(tail -n 1 "$OUT/bitid/dmtkt-$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=(iq4_xs|q4_K|q5_K),type_b=f32,m=(10240|12288|6144|2560|248320|512),n=[1-8],k=(2560|6144),|type_a=q6_K,type_b=f32,m=2560,n=[1-8],k=6144,'
    for pass in $(seq 1 "$PASSES"); do
        for arm in $PERF_ARMS; do
            e=$(arm_env "$arm") || { log "unknown arm $arm"; continue; }
            quiet
            log "perf $arm pass $pass start ($e)"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" GGML_TEST_RAW_QUANT=1 $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "$PP" \
                > "$OUT/perf/$arm-$pass.txt" 2>&1
            log "perf $arm pass $pass exit $?: $(grep -c 'us/run' "$OUT/perf/$arm-$pass.txt") lines"
        done
    done
    us_table "$OUT/perf"
fi

if step alone; then
    mkdir -p "$OUT/alone"
    CASES=('type_a=q6_K,type_b=f32,m=2560,n=4,k=6144,' 'type_a=q4_K,type_b=f32,m=2560,n=5,k=6144,'
           'type_a=q4_K,type_b=f32,m=2560,n=4,k=6144,')
    quiet
    for pass in $(seq 1 "$PASSES"); do
        for i in "${!CASES[@]}"; do
            for arm in $ALONE_ARMS; do
                e=$(arm_env "$arm") || { log "unknown arm $arm"; continue; }
                # shellcheck disable=SC2086
                env CUDA_VISIBLE_DEVICES="$DEV" GGML_TEST_RAW_QUANT=1 $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT \
                    -p "${CASES[$i]}" > "$OUT/alone/$arm-$pass-c$i.txt" 2>&1
                log "alone $arm pass $pass case $i exit $?: $(grep 'us/run' "$OUT/alone/$arm-$pass-c$i.txt" | sed -E 's/.*runs - +//' | head -n 1)"
            done
        done
    done
    us_table "$OUT/alone"
fi

if step served; then
    : "${RIG_KERN:?}" "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    log "served start (rig_kern.sh, on: ${SERVED_ENV:-${P}_DMT_KT=1})"
    STEPS=served BIN="$BIN" OUT="$OUT/kern" DEV="$DEV" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$PRESET" RIG_SERVER="$RIG_SERVER" \
        REQ="$REQ" VERIFY_PROF="$VERIFY_PROF" ORDER="${ORDER:-off on on off}" LOADS="${LOADS:-warm prof}" \
        SERVED_ENV="${SERVED_ENV:-${P}_DMT_KT=1}" bash "$RIG_KERN" > "$OUT/served-console.txt" 2>&1
    log "served exit $?: $(tail -n 2 "$OUT/kern/status.txt" 2> /dev/null | tr '\n' ' ')"
    rm -rf "$OUT/served" && mv "$OUT/kern/served" "$OUT/served"
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
