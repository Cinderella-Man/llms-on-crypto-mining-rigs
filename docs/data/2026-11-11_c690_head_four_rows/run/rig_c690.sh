#!/usr/bin/env bash
# rig_c690.sh: C690 (Q1222) on a P106 rig. The 2-4-token Q5_K kernel (fork branch pascal-p106-v4-b12246-c690,
# GGML_CUDA_MMVQ_PASCAL_Q5K_MT) checked natively and timed: does Qwen3.8-Flash-Next's output layer (Q5_K,
# 2560 x 248,320) cost about one row's time for four rows, with each row's sums the one-row kernel's? Runs ON the rig,
# detached, the rig idle; ends with $OUT/DONE. Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set q5kmt (scripts/record_checks.sh SUITE=q5kmt: Q5_K's MUL_MAT cases at 1-4 tokens, the
#           output layer's among them) through scripts/replay_checks.sh on card DEV, about one small case in CPU_CHECK
#           (default 20) also against the rig's own CPU; passing: every run ends "ok"
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid q5k: every 2-4-token product against its columns' one-token
#           products, bit for bit, at Q5K_MT=1, =2, =1 with _R=1 (0 comparisons may differ), and at 0 under the preset's
#           k-column switches, the control (the k-column kernel sums in another order: comparisons differ, exit 1)
#   perf    test-backend-ops perf at the output layer's shape, 1-4 tokens, the arms of PERF_ARMS alternated over PASSES
#           passes; the 1-token case is the one-token kernel in every arm
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft, the switch off and on in ORDER, each with LOADS; the head's product per verify pass comes out
#           of verify_parts.py on the traces, the step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the q5kmt folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the
#   same tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER
#   (bench/agent_check/rig_server.sh), REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES
#   (default 2), PERF_ARMS (default "off on nopf r1 r1nopf nw2"), ORDER (default "off on on off"), LOADS (default
#   "warm prof"), SERVED_ENV (the switches of "on"; default the perf step's fastest arm at 4 tokens)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c690.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on nopf r1 r1nopf nw2}
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
M="${P}_Q5K_MT"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)    echo "$KP" ;;
        on)     echo "$KP $M=1" ;;
        nopf)   echo "$KP $M=2" ;;
        r1)     echo "$KP $M=1 ${M}_R=1" ;;
        r1nopf) echo "$KP $M=2 ${M}_R=1" ;;
        nw2)    echo "$KP $M=1 ${M}_NW=2" ;;
        r1nw4)  echo "$KP $M=1 ${M}_R=1 ${M}_NW=4" ;;
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
    for a in "on:$M=1" "nopf:$M=2" "r1:$M=1 ${M}_R=1" "control:$KP"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" q5k > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=q5_K,type_b=f32,m=248320,n=[1-4],k=2560,'
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
    # one line per arm, pass and token count: the case's us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(.*,n=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step served; then
    : "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    mkdir -p "$OUT/served"
    # the kernel's settings for "on": SERVED_ENV, else the perf step's fastest arm at 4 tokens (the mean of its passes),
    # else the defaults
    if [ -z "${SERVED_ENV:-}" ]; then
        best=$(awk -F'\t' '$1 != "off" && $3 == 4 { s[$1] += $4; n[$1]++ }
            END { for (a in s) print s[a] / n[a] "\t" a }' "$OUT/perf/us_run.tsv" 2>/dev/null | sort -g | head -n 1 | cut -f2)
        SERVED_ENV=$([ -n "$best" ] && arm_env "$best" | sed "s|$KP ||" || echo "$M=1")
        log "served: on = ${best:-the defaults} ($SERVED_ENV)"
    fi
    k=0
    for arm in $ORDER; do
        k=$((k + 1))
        pf=$OUT/served/preset-$arm.env
        cp "$PRESET" "$pf"
        # the preset exactly, plus the kernel's switches for "on"
        [ "$arm" = on ] && echo "PRESET_ENV=\"\$PRESET_ENV $SERVED_ENV\"" >> "$pf"
        log "served $k $arm start"
        BIN="$BIN" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$VERIFY_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
