#!/usr/bin/env bash
# rig_c767.sh: C767 (Q1348) on a P106 rig. Plain one-token Q5_K products on C690's kernel (fork branch
# pascal-p106-v4-b12246-c767, GGML_CUDA_MMVQ_PASCAL_Q5K_1R) checked natively and timed: does Qwen3.8-Flash-Next's
# output layer (Q5_K, 2560 x 248,320) take less than the one-token kernel's 3,023.54 us at one token, with every result
# that kernel's bit for bit? From rig_c690.sh (2026-11-11_c690_head_four_rows/run/). Runs ON the rig, detached, the rig
# idle; ends with $OUT/DONE. Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set q5k1r (scripts/record_checks.sh SUITE=q5k1r: Q5_K's one-token MUL_MAT cases, the output
#           layer's among them, at each setting, and the fault) through scripts/replay_checks.sh on card DEV, about one
#           small case in CPU_CHECK (default 20) also against the rig's own CPU; passing: every run ends "ok" (the
#           fault's "ok" means it failed cases, as it must)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid q5k1: the one-token kernel writes every product's values
#           (GGML_CUDA_MMVQ_PASCAL_Q5K_1R=0, the reference), then each setting of the switch compares with them bit for
#           bit (0 comparisons may differ), and the fault, which must differ (exit 1); then multitoken_bitid q5k at
#           Q5K_MT=1 with the switch on (each 2-4-token product against its columns' one-token products, now on the
#           same kernel: 0 may differ)
#   perf    test-backend-ops perf at one token at the output layer's shape and Flash-Next's 6144 -> 2560 and 2560 -> 512,
#           the arms of PERF_ARMS alternated over PASSES passes; "off" is the one-token kernel
#   served  a served plain decode step under the profiler (run/decode_prof.sh: the preset without the MTP draft, so each
#           generated token is a one-token pass), the switch off and on in ORDER, each with LOADS; the output layer's
#           product per pass comes out of 2026-11-10_served_verify_parts/run/verify_parts.py on each trace, the step
#           time out of the warm loads' server timings; the greedy replies must carry the same text off and on
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the q5k1r folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the
#   same tree); served: MODEL, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh),
#   REQ (2026-11-04_draft_one_card/run/greedy_request.json), DECODE_PROF (run/decode_prof.sh); PASSES (default 3),
#   PERF_ARMS (default "off on r2 pf nw4"), ORDER (default "off on on off"), LOADS (default "warm prof"), SERVED_ENV
#   (the switches of "on"; default the perf step's fastest arm at the output layer's shape)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... PRESET=... RIG_SERVER=... REQ=... DECODE_PROF=... \
#   setsid -f nohup bash rig_c767.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-3}
PERF_ARMS=${PERF_ARMS:-off on r2 pf nw4}
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
O="${P}_Q5K_1R"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)  echo "$KP" ;;
        on)   echo "$KP $O=1" ;;
        r2)   echo "$KP $O=1 ${M}_R=2" ;;
        pf)   echo "$KP $O=1 ${M}_PF=1" ;;
        r2pf) echo "$KP $O=1 ${M}_R=2 ${M}_PF=1" ;;
        nw4)  echo "$KP $O=1 ${M}_NW=4" ;;
        nw2)  echo "$KP $O=1 ${M}_NW=2" ;;
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
    B=$OUT/bitid
    mkdir -p "$B"
    rm -f "$B/ref.bin"
    for a in "ref:$KP $O=0" "on:$KP $O=1" "r2:$KP $O=1 ${M}_R=2" "pf:$KP $O=1 ${M}_PF=1" "r2pf:$KP $O=1 ${M}_R=2 ${M}_PF=1" \
             "nw1:$KP $O=1 ${M}_NW=1" "head:$KP $O=1 ${O}_ROWS=100000" "defaults:$O=1" "fault:$KP $O=1 ${O}_FAULT=1"; do
        n=${a%%:*}
        log "bitid q5k1 $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" BITID_REF="$B/ref.bin" ${a#*:} "$BITID" q5k1 > "$B/q5k1-$n.txt" 2>&1
        log "bitid q5k1 $n exit $?: $(tail -n 1 "$B/q5k1-$n.txt")"
    done
    md5sum "$B/ref.bin" > "$B/ref.md5"
    log "bitid q5k mt+1r start ($KP $M=1 $O=1)"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $KP $M=1 $O=1 "$BITID" q5k > "$B/q5k-mt-1r.txt" 2>&1
    log "bitid q5k mt+1r exit $?: $(tail -n 1 "$B/q5k-mt-1r.txt")"
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=q5_K,type_b=f32,m=(248320,n=1,k=2560|2560,n=1,k=6144|512,n=1,k=2560),'
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
    # one line per arm, pass and shape: the case's us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT\(.*,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1x\3\t\4/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step served; then
    : "${MODEL:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${DECODE_PROF:?}"
    mkdir -p "$OUT/served"
    # the kernel's settings for "on": SERVED_ENV, else the perf step's fastest arm at the output layer (the mean of its
    # passes), else the defaults
    if [ -z "${SERVED_ENV:-}" ]; then
        best=$(awk -F'\t' '$1 != "off" && $3 == "248320x2560" { s[$1] += $4; n[$1]++ }
            END { for (a in s) print s[a] / n[a] "\t" a }' "$OUT/perf/us_run.tsv" 2>/dev/null | sort -g | head -n 1 | cut -f2)
        SERVED_ENV=$([ -n "$best" ] && arm_env "$best" | sed "s|$KP ||" || echo "$O=1")
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
        BIN="$BIN" MODEL="$MODEL" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$DECODE_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
