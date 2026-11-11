#!/usr/bin/env bash
# rig_c757.sh: C757 (Q1333) on a P106 rig. The one launch of a gate, an up and their SWIGLU at 2-8 tokens (fork branch
# pascal-p106-v4-b12246-c757, GGML_CUDA_MMVQ_PASCAL_GLU_MT) checked natively and timed: does Qwen3.8-Flash-Next's
# shared expert cost less as one launch per layer than as two products and a SWIGLU, with every sum the separate
# product's? The baseline is the separate products on the read-once kernels the launch repeats (C722's at R = 1,
# R79; C690's for Q5_K): RO below, in every arm. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE. C726's
# rig_c726.sh with this kernel's arms. Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set glumt (scripts/record_checks.sh SUITE=glumt: the MUL_MAT_GLU cases at 1-8 tokens, nine
#           switch settings, one of them the fault) through scripts/replay_checks.sh on card DEV, about one small case in
#           CPU_CHECK (default 20) also against the rig's own CPU; passing: every run ends "ok" (the fault run: its cases
#           fail, as expected)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid glu: every fused 2-8-token launch against the two products and the
#           SWIGLU run apart (at the same tokens and token by token), bit for bit, at GLU_MT=1, =2, =1 with _NP=1, and off
#           (the products apart against themselves and the one-token ones: 0 differences with RO), all under the
#           preset's k-column switches and RO; then the fault (_FAULT=1), the harness's own control: comparisons differ,
#           exit 1
#   perf    test-backend-ops perf -o MUL_MAT_GLU at the shared expert's shape (2560 -> 640) for the type pairs of PERF_TYPES
#           (gate and up both of them; default Q2_0, Q3_K, Q4_K, IQ4_XS: 12 of the file's 16 pairs, 37 of its 41
#           layers but Q5_K's two), PERF_N tokens (default 1-8), the arms of PERF_ARMS alternated over PASSES passes; the
#           1-token cases and the arms' cases the launch leaves apart are the separate products in every arm, so they
#           give the spread
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft and RO, the switch off and on in ORDER, each with LOADS; the shared expert's gate, up and
#           SWIGLU per verify pass come out of verify_parts.py on the traces (then run/shexp_table.py), the step time out
#           of the warm loads' server timings (2026-11-11_r79_c722_p106/run/served_table.py)
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the glumt folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the same
#   tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh),
#   REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES (default 2), PERF_ARMS (default
#   "off on nopf np1"), PERF_N (a regex digit class, default [1-8]), PERF_TYPES (default "q2_0|q3_K|q4_K|iq4_xs"), ORDER
#   (default "off on on off"), LOADS (default "warm prof"), SERVED_ENV (the switches of "on" besides RO; default the
#   perf step's fastest arm at 4 tokens)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c757.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on nopf np1}
PERF_N=${PERF_N:-[1-8]}
PERF_TYPES=${PERF_TYPES:-q2_0|q3_K|q4_K|iq4_xs}
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
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"            # the Flash-Next preset's k-column switches
RO="${P}_DMT=1 ${P}_DMT_R=1 ${P}_Q5K_MT=1"             # the read-once kernels the launch repeats (C722 at R = 1, C690)
M="${P}_GLU_MT"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)  echo "$KP $RO" ;;
        on)   echo "$KP $RO $M=1" ;;
        nopf) echo "$KP $RO $M=2" ;;
        np1)  echo "$KP $RO $M=1 ${M}_NP=1" ;;
        np2)  echo "$KP $RO $M=1 ${M}_NP=2" ;;
        np3)  echo "$KP $RO $M=1 ${M}_NP=3" ;;
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
    for a in "on:$KP $RO $M=1" "nopf:$KP $RO $M=2" "np1:$KP $RO $M=1 ${M}_NP=1" "off:$KP $RO" \
             "fault:$KP $RO $M=1 ${M}_FAULT=1"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" glu > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP="type_gate=($PERF_TYPES),type_up=($PERF_TYPES),m=640,n=$PERF_N,k=2560,"
    for pass in $(seq 1 "$PASSES"); do
        for arm in $PERF_ARMS; do
            e=$(arm_env "$arm") || { log "unknown arm $arm"; continue; }
            quiet
            log "perf $arm pass $pass start ($e)"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" GGML_TEST_RAW_QUANT=1 $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT_GLU -p "$PP" \
                > "$OUT/perf/$arm-$pass.txt" 2>&1
            log "perf $arm pass $pass exit $?: $(grep -c 'us/run' "$OUT/perf/$arm-$pass.txt") lines"
        done
    done
    # one line per arm, pass, type pair and token count: the case's us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT_GLU\(type_gate=([^,]+),type_up=([^,]+),m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\3\t\5\t\4\t\6/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step served; then
    : "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    mkdir -p "$OUT/served"
    # the launch's settings for "on": SERVED_ENV, else the perf step's fastest arm at 4 tokens (its times summed over the
    # pairs and passes, which every arm ran alike), else the defaults
    if [ -z "${SERVED_ENV:-}" ]; then
        best=$(awk -F'\t' '$1 != "off" && $7 == 4 { s[$1] += $8 }
            END { for (a in s) print s[a] "\t" a }' "$OUT/perf/us_run.tsv" 2>/dev/null | sort -g | head -n 1 | cut -f2)
        SERVED_ENV=$([ -n "$best" ] && arm_env "$best" | sed "s|$KP $RO ||" || echo "$M=1")
        log "served: on = ${best:-the defaults} ($SERVED_ENV)"
    fi
    k=0
    for arm in $ORDER; do
        k=$((k + 1))
        pf=$OUT/served/preset-$arm.env
        cp "$PRESET" "$pf"
        # the preset exactly, plus RO in every arm and the launch's switches for "on"
        if [ "$arm" = on ]; then
            echo "PRESET_ENV=\"\$PRESET_ENV $RO $SERVED_ENV\"" >> "$pf"
        else
            echo "PRESET_ENV=\"\$PRESET_ENV $RO\"" >> "$pf"
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
