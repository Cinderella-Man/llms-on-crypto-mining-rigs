#!/usr/bin/env bash
# rig_c727.sh: C727 (Q1286) on a P106 rig. Q6_K's 2-4-token products in C722's read-once kernel (fork branch
# pascal-p106-v4-b12246-c727, GGML_CUDA_MMVQ_PASCAL_Q6K_MT) checked natively and timed: do Qwen3.8-Flash-Next's Q6_K
# attn_output and ssm_out (6144 -> 2560) cost less at 2-4 tokens than on the k-column kernel, with each token's sums the
# one-token kernel's? Runs ON the rig, detached, the rig idle; ends with $OUT/DONE. C726's rig_c726.sh with Q6_K's arms.
# Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set q6kmt (scripts/record_checks.sh SUITE=q6kmt: Q6_K's MUL_MAT cases at 1-5 tokens, Flash-Next's
#           shape among them, thirteen switch settings, one of them the fault) through scripts/replay_checks.sh on card
#           DEV, about one small case in CPU_CHECK (default 20) also against the rig's own CPU; passing: every run ends
#           "ok" (the fault run: its cases at 2-4 tokens fail, as expected)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid q6k: every 2-4-token product against its columns' one-token
#           products, bit for bit, at Q6K_MT=1, =2, =1 with _DMT_R=2, =1 with _DMT_NW=1, all under the preset's k-column
#           switches (0 differences); then the controls, which must differ: the switch off (the k-column kernel at 2-4
#           tokens) and the fault (_FAULT=1)
#   perf    test-backend-ops perf at Flash-Next's Q6_K shape (6144 -> 2560), PERF_N tokens (default 1-5), the arms of
#           PERF_ARMS alternated over PASSES passes; the 1-token cases are the one-token kernel and the 5-token ones the
#           k-column kernel in every arm, so they give the spread
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft, the switch off and on in ORDER, each with LOADS; the Q3_K products' time per verify pass comes
#           out of verify_parts.py on the traces (then 2026-11-11_r79_c722_p106/run/served_table.py --types q6_K), the
#           step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the q6kmt folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the same
#   tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh),
#   REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES (default 2), PERF_ARMS (default
#   "off on nopf r2 nw1 nw1nopf"), PERF_N (a regex digit class, default [1-5]), ORDER (default "off on on off"),
#   LOADS (default "warm prof"), SERVED_ENV (the switches of "on"; default the perf step's fastest arm at 4 tokens)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c726.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on nopf r2 nw1 nw1nopf}
PERF_N=${PERF_N:-[1-5]}
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
M="${P}_Q6K_MT"
D="${P}_DMT"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)    echo "$KP" ;;
        on)     echo "$KP $M=1" ;;
        nopf)   echo "$KP $M=2" ;;
        r2)      echo "$KP $M=1 ${D}_R=2" ;;
        nw1)     echo "$KP $M=1 ${D}_NW=1" ;;
        nw1nopf) echo "$KP $M=2 ${D}_NW=1" ;;
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
    for a in "on:$KP $M=1" "nopf:$KP $M=2" "r2:$KP $M=1 ${D}_R=2" "nw1:$KP $M=1 ${D}_NW=1" "kcol:$KP" \
             "fault:$KP $M=1 ${M}_FAULT=1"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" q6k > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP="type_a=q6_K,type_b=f32,m=2560,n=$PERF_N,k=6144,"
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
        grep 'us/run' "$f" | grep 'bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,' | sed -E "s/^ *MUL_MAT\(type_a=([^,]+),type_b=f32,m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\4\t\3\t\5/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step served; then
    : "${MODEL:?}" "${DRAFT:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${REQ:?}" "${VERIFY_PROF:?}"
    mkdir -p "$OUT/served"
    # the kernel's settings for "on": SERVED_ENV, else the perf step's fastest arm at 4 tokens (its times summed over the
    # shapes and passes, which every arm ran alike), else the defaults
    if [ -z "${SERVED_ENV:-}" ]; then
        best=$(awk -F'\t' '$1 != "off" && $6 == 4 { s[$1] += $7 }
            END { for (a in s) print s[a] "\t" a }' "$OUT/perf/us_run.tsv" 2>/dev/null | sort -g | head -n 1 | cut -f2)
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
