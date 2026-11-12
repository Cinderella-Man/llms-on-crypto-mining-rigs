#!/usr/bin/env bash
# rig_c807.sh: C807 (Q1413) on a P106 rig. GGML_CUDA_MMVF_F32_MT=1 (fork branch pascal-p106-v4-b12246-c807) sends F32
# weight products of 4-8 tokens to llama.cpp's mul_mat_vec_f on compute 6.1, now only for a matrix of at most
# GGML_CUDA_MMVF_F32_MT_ROWS rows (default 32): in a verify pass of Qwen3.8-Flash-Next the shared expert's gate
# (ffn_gate_inp_shexp, 2560 -> 1) stays on mul_mat_vec_f and the router (ffn_gate_inp, 2560 -> 512) goes back to cuBLAS.
# Does the limit hold on the card, where does mul_mat_vec_f stop gaining over cuBLAS as the rows grow, and what does
# the switch now take off a served verify pass? Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set f32x of this build (scripts/record_checks.sh SUITE=f32x) through scripts/replay_checks.sh
#           on card DEV; passing: every run ends "ok": f32x-fault and f32x-fault2 fail cases (every one of theirs, at most
#           32 rows), f32x-fault-above and f32x-fault2-above pass every case (above 32 rows the switch leaves the
#           product to cuBLAS), f32x-fault-rows0 and f32x-fault2-rows0 fail cases (_ROWS=0 takes them again)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid against each product's one-token columns, bit for bit: f32r (the
#           shapes of at most 32 rows) at =1 and =2 (0 may differ) and off (the control: cuBLAS at 4-8 tokens differs);
#           f32 (the router and the other shapes above 32 rows among them) at =1 with _ROWS=0 (0 may differ) and at =1
#           (the limit: the products above 32 rows differ at 4-8 tokens, as off)
#   perf    test-backend-ops perf at 2560-wide F32 rows: 1, 16, 32, 33, 64, 128, 256 and 512 rows, 1-8 tokens, the arms
#           of PERF_ARMS alternated over PASSES passes: off (cuBLAS at 4-8 tokens), on (the switch: mul_mat_vec_f up to 32
#           rows), rows0 (the switch with _ROWS=0: mul_mat_vec_f at every row count). 1-3 tokens run mul_mat_vec_f in
#           every arm, so they give the spread; rows0 against off at 4-8 tokens shows where the limit belongs
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft, the switch off and on in ORDER, each with LOADS; the router's and the shared expert's time per
#           verify pass come out of verify_parts.py on the traces, the step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the f32x folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the
#   same tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER
#   (bench/agent_check/rig_server.sh), REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES
#   (default 2), PERF_ARMS (default "off on rows0"), ORDER (default "off on on off"), LOADS (default "warm prof")
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c807.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on rows0}
ORDER=${ORDER:-off on on off}
LOADS=${LOADS:-warm prof}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
F=GGML_CUDA_MMVF_F32_MT
ON="$F=1"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)   echo "" ;;
        on)    echo "$ON" ;;
        rows0) echo "$ON ${F}_ROWS=0" ;;
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
    for r in f32x-fault f32x-fault2 f32x-fault-above f32x-fault2-above f32x-fault-rows0 f32x-fault2-rows0; do
        [ -f "$OUT/replay/$r.tsv" ] && log "$r: $(cut -f 3 "$OUT/replay/$r.tsv" | sort | uniq -c | tr '\n' ' ')"
    done
fi

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    for a in "f32r:on:$ON" "f32r:on2:$F=2" "f32r:control:" "f32:rows0:$ON ${F}_ROWS=0" "f32:limit:$ON"; do
        mode=${a%%:*}
        rest=${a#*:}
        n=${rest%%:*}
        log "bitid $mode $n start (${rest#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${rest#*:} "$BITID" "$mode" > "$OUT/bitid/$mode-$n.txt" 2>&1
        log "bitid $mode $n exit $?: $(tail -n 1 "$OUT/bitid/$mode-$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=f32,type_b=f32,m=(1|16|32|33|64|128|256|512),n=[1-8],k=2560,'
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

if step served; then
    : "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    mkdir -p "$OUT/served"
    k=0
    for arm in $ORDER; do
        k=$((k + 1))
        pf=$OUT/served/preset-$arm.env
        cp "$PRESET" "$pf"
        # the preset exactly, plus the switch for "on"
        [ "$arm" = on ] && echo "PRESET_ENV=\"\$PRESET_ENV $ON\"" >> "$pf"
        log "served $k $arm start"
        BIN="$BIN" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$VERIFY_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
