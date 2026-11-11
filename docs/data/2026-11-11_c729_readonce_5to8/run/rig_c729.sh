#!/usr/bin/env bash
# rig_c729.sh: C729 (Q1288) on a P106 rig. The read-once kernels of C722 (IQ4_XS, Q4_K) and C690 (Q5_K) at 5 to 8
# tokens (fork branch pascal-p106-v4-b12246-c729, GGML_CUDA_MMVQ_PASCAL_RO8) checked natively and timed: do
# Qwen3.8-Flash-Next's projections in these types cost less at 5-8 tokens than on the k-column kernel, with each token's
# sums the one-token kernel's? From rig_c722.sh (2026-11-11_c722_short_batch_products/run/). Runs ON the rig, detached,
# the rig idle; ends with $OUT/DONE. Steps (STEPS, default "replay bitid perf"):
#   replay  the recorded set ro8 (scripts/record_checks.sh SUITE=ro8: the three types' MUL_MAT cases at 5-8 tokens at
#           twelve settings and the fault) through scripts/replay_checks.sh on card DEV, about one small case in
#           CPU_CHECK (default 20) also against the rig's own CPU; passing: every run ends "ok" (the fault's "ok" means it
#           failed cases, as it must)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid dmt8 and q5k8: every 5-8-token product the kernels take against
#           its columns' one-token products, bit for bit, at RO8=1, =2, with Q5_K's prefetch, and with the 2-4-token
#           switches (0 comparisons may differ); dmt and q5k at 2-4 tokens with all three switches (RO8 must not reach
#           them: 0 may differ); the control (RO8=0, BITID_ALL=1: the k-column kernel's sums, which must differ, exit 1)
#           and the fault (RO8_FAULT=1, exit 1)
#   perf    test-backend-ops perf at Flash-Next's shapes of the three types (IQ4_XS and Q4_K attn_qkv, attn_gate,
#           attn_q and ssm_out; Q5_K output, ssm_out and attn_k), 1-8 tokens, the arms of PERF_ARMS alternated over
#           PASSES passes. off = the k-column kernel at 2-8 tokens; mt = the 2-4-token switches (DMT=1, Q5K_MT=1); on,
#           nopf, pf = mt with RO8 (=1, =2, =1 with Q5_K's prefetch). The 1-token cases are the one-token kernel in every
#           arm and the 2-4-token cases the same kernels in mt, on, nopf, so they give the spread
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the ro8 folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the same
#   tree); PASSES (default 2), PERF_ARMS (default "off mt on nopf pf")
# The served or prompt timing (R42's driver, off = the 2-4-token switches, on = with RO8 too) runs from the build
# machine's side: README, "The rig run".
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... setsid -f nohup bash rig_c729.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off mt on nopf pf}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" # the Flash-Next preset's k-column switches
MT="${P}_DMT=1 ${P}_Q5K_MT=1"                # the 2-4-token read-once kernels (C722, C690; R = 1 by default)
M="${P}_RO8"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)  echo "$KP" ;;
        mt)   echo "$KP $MT" ;;
        on)   echo "$KP $MT $M=1" ;;
        nopf) echo "$KP $MT $M=2" ;;
        pf)   echo "$KP $MT $M=1 ${P}_Q5K_MT_PF=1" ;;
        nw4)  echo "$KP $MT $M=1 ${P}_DMT_NW=4 ${P}_Q5K_MT_NW=4" ;;
        *)    return 1 ;;
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
    # mode:name:switches
    for a in "dmt8:on:$KP $M=1" "dmt8:nopf:$KP $M=2" "dmt8:mt:$KP $MT $M=1" "dmt8:control:$KP BITID_ALL=1" \
             "dmt8:fault:$KP $M=1 ${M}_FAULT=1" \
             "q5k8:on:$KP $M=1" "q5k8:nopf:$KP $M=2" "q5k8:pf:$KP $M=1 ${P}_Q5K_MT_PF=1" "q5k8:mt:$KP $MT $M=1" \
             "q5k8:control:$KP BITID_ALL=1" "q5k8:fault:$KP $M=1 ${M}_FAULT=1" \
             "dmt:mt-ro8:$KP $MT $M=1" "q5k:mt-ro8:$KP $MT $M=1"; do
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
    PP='type_a=(iq4_xs|q4_K|q5_K),type_b=f32,m=(10240|12288|6144|2560|248320|512),n=[1-8],k=(2560|6144),'
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
    # one line per arm, pass, shape and token count: arm, pass, type, rows, K, tokens, us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(type_a=([^,]+),type_b=f32,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\4\t\3\t\5/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
