#!/usr/bin/env bash
# rig_kern.sh: the v4 kernel merge (C796) on a P106 rig: C727's Q6_K read-once products (GGML_CUDA_MMVQ_PASCAL_Q6K_MT),
# C728's K tiles (_DMT_KT) and C729's 5-8 tokens (_RO8), merged into pascal-p106-v4, checked natively and timed, each
# alone and the three on together. From the three branches' rig scripts (docs/data/2026-11-11_c727_q6k_readonce,
# 2026-11-11_c728_readonce_k6144, 2026-11-11_c729_readonce_5to8: run/rig_*.sh). Runs ON the rig, detached, the rig idle;
# ends with $OUT/DONE. Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set v4kern (scripts/record_checks.sh SUITE=v4kern: the suites q6kmt, dmtkt and ro8 with their
#           runs, the three switches on together at six settings, and each fault with all three on) through
#           scripts/replay_checks.sh on card DEV, about one small case in CPU_CHECK (default 20) also against the rig's
#           own CPU; passing: every run ends "ok" and each fault run fails cases
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid, every product against its columns' one-token products, bit for
#           bit: each branch's own arms (q6k: C727's; dmtkt: C728's; dmt8 and q5k8: C729's), then every mode (q6k, dmtkt,
#           dmt, dmt8, q5k, q5k8) with the three switches on together (and _KT=2); 0 comparisons may differ, and each
#           fault and control must differ
#   perf    test-backend-ops perf at Qwen3.8-Flash-Next's IQ4_XS, Q4_K, Q5_K and Q6_K shapes, 1-8 tokens, the arms of
#           PERF_ARMS alternated over PASSES passes: off (the preset's k-column switches), mt (+ the day's read-once
#           kernels _DMT=1 _Q5K_MT=1), kern (mt + the three switches), kernm2 (kern with two row blocks a tile round)
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft and the day's read-once kernels (_DMT=1 _Q5K_MT=1) in both arms, the three switches off and on
#           in ORDER, each with LOADS; the products' time per verify pass comes out of verify_parts.py on the traces, the
#           step time out of the warm loads' server timings
# Env: BIN (bin/ of the merged rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET (the
#   v4kern folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the same
#   tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh),
#   REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES (default 2), PERF_ARMS (default
#   "off mt kern kernm2"), ORDER (default "off on on off"), LOADS (default "warm prof"), SERVED_ENV (the switches of "on"
#   beyond both arms'; default the three at their defaults)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_kern.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off mt kern kernm2}
ORDER=${ORDER:-off on on off}
LOADS=${LOADS:-warm prof}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" # the Flash-Next preset's k-column switches
MT="${P}_DMT=1 ${P}_Q5K_MT=1"                # the day's read-once kernels (C722, C690; R = 1 since C758 and C766)
KERN="${P}_Q6K_MT=1 ${P}_DMT_KT=1 ${P}_RO8=1" # the three merged switches at their defaults
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)    echo "$KP" ;;
        mt)     echo "$KP $MT" ;;
        kern)   echo "$KP $MT $KERN" ;;
        kernm2) echo "$KP $MT $KERN ${P}_DMT_KT_M=2" ;;
        *)      return 1 ;;
    esac
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step replay; then
    : "${SET:?}" "${REPLAY:?}"
    log "replay start"
    SET="$SET" BIN="$BIN" OUT="$OUT/replay" DEV="$DEV" CPU_CHECK="${CPU_CHECK:-20}" bash "$REPLAY" > "$OUT/replay-console.txt" 2>&1
    log "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt" 2>/dev/null)"
fi

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    Q6="${P}_Q6K_MT" D="${P}_DMT" R8="${P}_RO8" ALL="$KP $MT $KERN"
    # mode:name:switches
    for a in "q6k:on:$KP $Q6=1" "q6k:nopf:$KP $Q6=2" "q6k:r2:$KP $Q6=1 ${D}_R=2" "q6k:nw1:$KP $Q6=1 ${D}_NW=1" \
             "q6k:kcol:$KP" "q6k:fault:$KP $Q6=1 ${Q6}_FAULT=1" \
             "dmtkt:dmt:$KP $D=1" "dmtkt:kt1:$KP $D=1 ${D}_KT=1" "dmtkt:kt2:$KP $D=1 ${D}_KT=2" \
             "dmtkt:kt2m2:$KP $D=1 ${D}_KT=2 ${D}_KT_M=2" "dmtkt:kt2nopf:$KP $D=2 ${D}_KT=2" \
             "dmtkt:kt2nw4:$KP $D=1 ${D}_KT=2 ${D}_KT_NW=4" "dmtkt:fault:$KP $D=1 ${D}_KT=1 ${D}_KT_FAULT=1" \
             "dmt8:on:$KP $R8=1" "dmt8:nopf:$KP $R8=2" "dmt8:mt:$KP $MT $R8=1" "dmt8:control:$KP BITID_ALL=1" \
             "dmt8:fault:$KP $R8=1 ${R8}_FAULT=1" \
             "q5k8:on:$KP $R8=1" "q5k8:nopf:$KP $R8=2" "q5k8:pf:$KP $R8=1 ${P}_Q5K_MT_PF=1" "q5k8:mt:$KP $MT $R8=1" \
             "q5k8:control:$KP BITID_ALL=1" "q5k8:fault:$KP $R8=1 ${R8}_FAULT=1" \
             "q6k:all:$ALL" "dmtkt:all:$ALL" "dmt:all:$ALL" "dmt8:all:$ALL" "q5k:all:$ALL" "q5k8:all:$ALL" \
             "dmtkt:all-kt2:$KP $MT $Q6=1 $R8=1 ${D}_KT=2" "q6k:all-fault:$ALL ${Q6}_FAULT=1" \
             "dmtkt:all-fault:$ALL ${D}_KT_FAULT=1" "dmt8:all-fault:$ALL ${R8}_FAULT=1"; do
        mode=${a%%:*} rest=${a#*:}
        n=${rest%%:*} sw=${rest#*:}
        log "bitid $mode $n start ($sw)"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" $sw "$BITID" "$mode" > "$OUT/bitid/$mode-$n.txt" 2>&1
        log "bitid $mode $n exit $?: $(tail -n 1 "$OUT/bitid/$mode-$n.txt")"
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
    # one line per arm, pass, shape and token count: the case's us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(type_a=([^,]+),type_b=f32,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\4\t\3\t\5/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step served; then
    : "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    mkdir -p "$OUT/served"
    SERVED_ENV=${SERVED_ENV:-$KERN}
    log "served: both arms + $MT; on + $SERVED_ENV"
    k=0
    for arm in $ORDER; do
        k=$((k + 1))
        pf=$OUT/served/preset-$arm.env
        cp "$PRESET" "$pf"
        # the preset exactly, plus the day's read-once kernels in both arms and the three switches for "on"
        if [ "$arm" = on ]; then
            echo "PRESET_ENV=\"\$PRESET_ENV $MT $SERVED_ENV\"" >> "$pf"
        else
            echo "PRESET_ENV=\"\$PRESET_ENV $MT\"" >> "$pf"
        fi
        log "served $k $arm start"
        BIN="$BIN" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$VERIFY_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
