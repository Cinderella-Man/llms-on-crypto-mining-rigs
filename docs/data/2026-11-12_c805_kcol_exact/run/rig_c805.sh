#!/usr/bin/env bash
# rig_c805.sh: C805 on a P106 rig: GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT (the k-column kernel keeps the products of 2-8 tokens
# it used to leave to other kernels for shared memory) checked natively and timed, under the Flash-Next preset's k-column
# switches (KCOL_SMEM=32768, KCOL_TABLE=0). Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "bitid perf"):
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid, every product against its columns' one-token products, bit for bit,
#           modes q3k and glu (R120a's two failing modes): the controls without the switch (R120a's "kcol" and "off": they
#           must differ, the k-column kernel's declined products), the switch at =1 and =2, with C726's and C757's kernels,
#           with all six verify-pass switches, without the budget, at a 24 KB budget with =2 (three launches of 2 tokens
#           at Flash-Next's 6144-wide Q3_K rows: the split on Flash-Next's own width), and the fault (=1 with
#           KCOL_EXACT_FAULT=1: the split products must differ); 0 comparisons may differ in the others
#   perf    test-backend-ops perf at Qwen3.8-Flash-Next's Q3_K 6144 -> 2560 (ssm_out), 1-8 tokens, the arms of PERF_ARMS
#           alternated over PASSES passes: off (the preset: 6-8 tokens declined to llama.cpp's kernel), x1 (=1: the 48 KB
#           shape, one launch), x2 (=2: two launches within the preset's 32 KB), and the same three with C726's Q3_K kernel
#           on (q3off, q3x1, q3x2; it leaves this width to the k-column kernel from 5 tokens on)
# Env: BIN (bin/ of the C805 rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); bitid: BITID (the
#   multitoken_bitid binary built against the same tree); PASSES (default 2), PERF_ARMS (default "off x1 x2 q3off q3x1
#   q3x2")
# Usage: BIN=... OUT=... BITID=... setsid -f nohup bash rig_c805.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-bitid perf}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off x1 x2 q3off q3x1 q3x2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"      # the Flash-Next preset's k-column switches
X=${P}_KCOL_EXACT
Q3=${P}_Q3K_MT
GL=${P}_GLU_MT
RO="${P}_DMT=1 ${P}_Q5K_MT=1 ${P}_DMT_R=1"       # the read-once kernels C757's launch repeats
ALL6="$KP ${P}_DMT=1 ${P}_Q5K_MT=1 ${P}_Q6K_MT=1 ${P}_DMT_KT=1 ${P}_RO8=1 $Q3=1 $GL=1 GGML_CUDA_MMVF_F32_MT=1"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)   echo "$KP" ;;
        x1)    echo "$KP $X=1" ;;
        x2)    echo "$KP $X=2" ;;
        q3off) echo "$KP $Q3=1" ;;
        q3x1)  echo "$KP $Q3=1 $X=1" ;;
        q3x2)  echo "$KP $Q3=1 $X=2" ;;
        *)     return 1 ;;
    esac
}
quiet

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    # mode:name:switches; must: 0 (no comparison differs) or differ (control, fault)
    for a in "q3k:kcol:$KP:differ" "q3k:x1:$KP $X=1:0" "q3k:x2:$KP $X=2:0" "q3k:x1-on:$KP $Q3=1 $X=1:0" \
             "q3k:x1-r2:$KP $Q3=1 ${Q3}_R=2 $X=1:0" "q3k:x1-nobudget:${P}_KCOL_TABLE=0 $X=1:0" "q3k:x1-default:$X=1:0" \
             "q3k:x2-24k:${P}_KCOL_SMEM=24576 ${P}_KCOL_TABLE=0 $X=2:0" "q3k:x1-all6:$ALL6 $X=1:0" \
             "q3k:x1-fault:$KP $X=1 ${X}_FAULT=1:differ" \
             "glu:off:$KP $RO:differ" "glu:x1:$KP $RO $X=1:0" "glu:x2:$KP $RO $X=2:0" "glu:x1-on:$KP $RO $GL=1 $X=1:0" \
             "glu:x1-all6:$ALL6 $X=1:0" "glu:x1-fault:$KP $RO $X=1 ${X}_FAULT=1:differ"; do
        mode=${a%%:*} rest=${a#*:}
        n=${rest%%:*} rest=${rest#*:}
        sw=${rest%:*} must=${rest##*:}
        log "bitid $mode $n start ($sw; must: $must)"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" $sw "$BITID" "$mode" > "$OUT/bitid/$mode-$n.txt" 2>&1
        log "bitid $mode $n exit $?: $(tail -n 1 "$OUT/bitid/$mode-$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=q3_K,type_b=f32,m=2560,n=[1-8],k=6144,'
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

quiet
log "done"
date -Is > "$OUT/DONE"
