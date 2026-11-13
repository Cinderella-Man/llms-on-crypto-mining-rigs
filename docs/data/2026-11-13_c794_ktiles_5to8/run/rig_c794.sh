#!/usr/bin/env bash
# rig_c794.sh: R149, C794 on a P106 rig: C728's K tiles at 5 to 8 tokens (GGML_CUDA_MMVQ_PASCAL_RO8_KT) for the
# products C729's whole staging holds at few warps or leaves to the k-column kernel (Qwen3.8-Flash-Next's ssm_out and
# attn_output, 2,560 x 6,144). Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "replay bitid perf"):
#   replay  scripts/replay_checks.sh on the set ro8kt (SET: 197 cases, 16 runs): every run ok, ro8kt-fault failing
#           cases at K = 6144
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid kt8, every product of 5 to 8 tokens against its columns' one-token
#           products, bit for bit, at _RO8_KT=1, =2, =2 with _DMT_KT_M=1, =2 without the prefetch (_RO8=2), and two
#           that must differ: the fault (_RO8_KT=2 _DMT_KT_FAULT=1) and the control (_RO8=1 without tiles, BITID_ALL=1:
#           the k-column kernel's sums of the products whole staging declines)
#   perf    test-backend-ops perf at 2,560 x 6,144, IQ4_XS and Q4_K, 5 to 8 tokens (and 4, for the step from it), the arms
#           of PERF_ARMS alternated over PASSES passes: off (the Flash-Next preset's k-column switches), kern (+ the
#           read-once kernels _DMT=1 _Q5K_MT=1 and C729's _RO8=1: whole staging where it fits, else the k-column kernel)
#           and kt (kern + _RO8_KT=1, the tiles)
# Env: BIN (bin/ of the C794 rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID), PASSES (2),
#   PERF_ARMS (default "off kern kt"); bitid: BITID (the multitoken_bitid binary built against the same tree); replay:
#   SET (the unpacked set ro8kt), REPLAY (scripts/replay_checks.sh)
# Output: $OUT/perf/us_run.tsv (arm, pass, type, rows, K, tokens, us per product), $OUT/bitid/*.txt, $OUT/replay/
#   (replay_checks.sh's), $OUT/status.txt; run/tables.sh turns us_run.tsv into the per-token-count table and the est. ms
#   a verify pass
# Usage: BIN=... OUT=... BITID=... SET=... REPLAY=... setsid -f nohup bash rig_c794.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off kern kt}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" # the Flash-Next preset's k-column switches
KERN="${P}_DMT=1 ${P}_Q5K_MT=1 ${P}_RO8=1"  # the read-once kernels at 2-8 tokens (C722, C690, C729)
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)  echo "$KP" ;;
        kern) echo "$KP $KERN" ;;
        kt)   echo "$KP $KERN ${P}_RO8_KT=1" ;;
        kt2)  echo "$KP $KERN ${P}_RO8_KT=2" ;;
        ktm1) echo "$KP $KERN ${P}_RO8_KT=1 ${P}_DMT_KT_M=1" ;;
        *)    return 1 ;;
    esac
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step replay; then
    : "${SET:?}" "${REPLAY:?}"
    log "replay ro8kt start"
    SET="$SET" BIN="$BIN" OUT="$OUT/replay" DEV="$DEV" bash "$REPLAY" > "$OUT/replay-console.txt" 2>&1
    log "replay exit $?: $(cut -f1,8 "$OUT/replay/results.tsv" 2> /dev/null | tr '\t\n' ' ;')"
fi

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    R="${P}_RO8" T="${P}_RO8_KT"
    for a in "kt1:$KP $R=1 $T=1" "kt2:$KP $R=1 $T=2" "kt2m1:$KP $R=1 $T=2 ${P}_DMT_KT_M=1" "kt2nopf:$KP $R=2 $T=2" \
             "fault:$KP $R=1 $T=2 ${P}_DMT_KT_FAULT=1" "control:$KP $R=1 BITID_ALL=1"; do
        n=${a%%:*} sw=${a#*:}
        log "bitid kt8 $n start ($sw)"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" $sw "$BITID" kt8 > "$OUT/bitid/kt8-$n.txt" 2>&1
        log "bitid kt8 $n exit $?: $(tail -n 2 "$OUT/bitid/kt8-$n.txt" | tr '\n' ' ')"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=(iq4_xs|q4_K),type_b=f32,m=2560,n=[4-8],k=6144,'
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
    for f in "$OUT"/perf/*-[0-9]*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(type_a=([^,]+),type_b=f32,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\4\t\3\t\5/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
