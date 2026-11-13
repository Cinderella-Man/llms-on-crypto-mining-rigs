#!/usr/bin/env bash
# rig_r144.sh: R144 (Q1448), C822 (Q1436) on a P106 rig. GGML_CUDA_MMVF_F32_MT=1 (fork branch
# pascal-p106-v4-b12246-c822, or pascal-p106-v4 from its merge) sends F32 weight products of 4-8 tokens to llama.cpp's
# mul_mat_vec_f on compute 6.1 for a matrix of at most GGML_CUDA_MMVF_F32_MT_ROWS rows (default 32); C822's
# _ROWS=auto sets 48 rows at 4 tokens and 64 at 5-8 (est. from R132), and a list one limit per token count. Where does
# mul_mat_vec_f stop beating cuBLAS between 33 and 64 rows, at each token count and at two row widths, and does auto
# route as designed on the card? Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "replay perf"):
#   replay  the recorded set f32x of this build (scripts/record_checks.sh SUITE=f32x, with C822's runs) through
#           scripts/replay_checks.sh on card DEV; passing: every run ends "ok": the fault runs that must fail fail every
#           one of their cases (f32x-fault-auto and f32x-fault2-auto: at most 48 rows at 4 tokens and 64 at 5-8;
#           f32x-fault-list: _ROWS=64,48), and the -above runs pass every case (the products the limit leaves to cuBLAS)
#   perf    test-backend-ops perf at F32 rows of 2560 and 2048: 32, 33, 40, 48, 56 and 64 rows, 1-8 tokens, the arms of
#           PERF_ARMS alternated over PASSES passes: off (cuBLAS at 4-8 tokens), on (the switch at its default, 32 rows),
#           rows0 (_ROWS=0: mul_mat_vec_f at every row count), auto (_ROWS=auto). 1-3 tokens run mul_mat_vec_f in every
#           arm, so they give the spread; rows0 against off at 4-8 tokens places the limit (perf_r144.py)
# Env: BIN (bin/ of the build's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET (the
#   f32x folder), REPLAY (scripts/replay_checks.sh); PASSES (default 2), PERF_ARMS (default "off on rows0 auto"), PP (the
#   perf filter)
# Usage: BIN=... OUT=... SET=... REPLAY=... setsid -f nohup bash rig_r144.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay perf}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on rows0 auto}
PP=${PP:-'type_a=f32,type_b=f32,m=(32|33|40|48|56|64),n=[1-8],k=(2560|2048),'}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(ps -eo comm | grep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
F=GGML_CUDA_MMVF_F32_MT
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)   echo "" ;;
        on)    echo "$F=1" ;;
        rows0) echo "$F=1 ${F}_ROWS=0" ;;
        auto)  echo "$F=1 ${F}_ROWS=auto" ;;
        *)     return 1 ;;
    esac
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step replay; then
    : "${SET:?}" "${REPLAY:?}"
    log "replay start"
    SET="$SET" BIN="$BIN" OUT="$OUT/replay" DEV="$DEV" CPU_CHECK="${CPU_CHECK:-20}" bash "$REPLAY" > "$OUT/replay-console.txt" 2>&1
    log "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt" 2>/dev/null)"
    # the fault runs: the verdicts of their cases
    for r in f32x-fault f32x-fault2 f32x-fault-above f32x-fault2-above f32x-fault-rows0 f32x-fault2-rows0 \
             f32x-fault-auto f32x-fault2-auto f32x-fault-auto-above f32x-fault2-auto-above f32x-fault-list f32x-fault-list-above; do
        [ -f "$OUT/replay/$r.tsv" ] && log "$r: $(cut -f 3 "$OUT/replay/$r.tsv" | sort | uniq -c | tr '\n' ' ')"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    for pass in $(seq 1 "$PASSES"); do
        for arm in $PERF_ARMS; do
            e=$(arm_env "$arm") || { log "unknown arm $arm"; continue; }
            quiet
            log "perf $arm pass $pass start ($e)"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "$PP" \
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
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
