#!/usr/bin/env bash
# rig_c728.sh: C728 (Q1287) on a P106 rig. The K tiles of the read-once IQ4_XS and Q4_K kernel (fork branch
# pascal-p106-v4-b12246-c728, GGML_CUDA_MMVQ_PASCAL_DMT_KT) checked natively and timed: do Qwen3.8-Flash-Next's ssm_out
# and attn_output (2560 x 6144) cost less at 3-4 tokens with the activations staged one K tile at a time, with each
# token's sums the one-token kernel's? Runs ON the rig, detached, the rig idle; ends with $OUT/DONE. Steps (STEPS, default
# "replay bitid perf served"):
#   replay  the recorded set dmtkt (scripts/record_checks.sh SUITE=dmtkt: IQ4_XS's and Q4_K's MUL_MAT cases at 1-5
#           tokens, Flash-Next's shapes among them, twelve switch settings and the fault) through scripts/replay_checks.sh
#           on card DEV, about one small case in CPU_CHECK (default 20) also against the rig's own CPU; passing: every
#           run ends "ok" and dmtkt-fault fails cases
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid dmtkt: every 2-4-token product against its columns' one-token
#           products, bit for bit, with C722's kernel alone (_DMT=1), its tiles where they help (_KT=1) and everywhere
#           (_KT=2), two row blocks a round, without the prefetch, 4 warps (0 comparisons may differ), and the fault
#           (_KT_FAULT=1: comparisons at K = 6144 must differ, exit 1)
#   perf    test-backend-ops perf at the K = 6144 shapes (2560 x 6144, both types), 1-5 tokens, the arms of PERF_ARMS
#           alternated over PASSES passes; the 1-token cases are the one-token kernel and the 5-token cases the k-column
#           kernel in every arm, so they give the spread
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft and C722's kernel (_DMT=1), the tiles off and on in ORDER, each with LOADS; the products' time
#           per verify pass comes out of verify_parts.py on the traces, the step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the dmtkt folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the
#   same tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh),
#   REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES (default 2), PERF_ARMS (default
#   "off dmt kt1 kt1m2 kt1nw5 kt1nopf kt2"), ORDER (default "off on on off"), LOADS (default "warm prof"),
#   SERVED_ENV (the tile switches of "on"; default the perf step's fastest kt arm at 4 tokens over the shapes)
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c728.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off dmt kt1 kt1m2 kt1nw5 kt1nopf kt2}
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
M="${P}_DMT"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)     echo "$KP" ;;
        dmt)     echo "$KP $M=1" ;;
        kt1)     echo "$KP $M=1 ${M}_KT=1" ;;
        kt1m2)   echo "$KP $M=1 ${M}_KT=1 ${M}_KT_M=2" ;;
        kt1nw5)  echo "$KP $M=1 ${M}_KT=1 ${M}_KT_NW=5" ;;
        kt1nopf) echo "$KP $M=2 ${M}_KT=1" ;;
        kt2)     echo "$KP $M=1 ${M}_KT=2" ;;
        *)       return 1 ;;
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
    for a in "dmt:$KP $M=1" "kt1:$KP $M=1 ${M}_KT=1" "kt2:$KP $M=1 ${M}_KT=2" "kt2m2:$KP $M=1 ${M}_KT=2 ${M}_KT_M=2" \
             "kt2nopf:$KP $M=2 ${M}_KT=2" "kt2nw4:$KP $M=1 ${M}_KT=2 ${M}_KT_NW=4" \
             "fault:$KP $M=1 ${M}_KT=1 ${M}_KT_FAULT=1"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" dmtkt > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=(iq4_xs|q4_K),type_b=f32,m=2560,n=[1-5],k=6144,'
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
    # the tiles' settings for "on": SERVED_ENV, else the perf step's fastest kt arm at 4 tokens (its times summed over
    # the shapes and passes, which every arm ran alike), else _KT=1
    if [ -z "${SERVED_ENV:-}" ]; then
        best=$(awk -F'\t' '$1 ~ /^kt/ && $6 == 4 { s[$1] += $7 }
            END { for (a in s) print s[a] "\t" a }' "$OUT/perf/us_run.tsv" 2>/dev/null | sort -g | head -n 1 | cut -f2)
        SERVED_ENV=$([ -n "$best" ] && arm_env "$best" | sed "s|$KP $M=[12] ||" || echo "${M}_KT=1")
        log "served: on = ${best:-kt1} ($SERVED_ENV)"
    fi
    k=0
    for arm in $ORDER; do
        k=$((k + 1))
        pf=$OUT/served/preset-$arm.env
        cp "$PRESET" "$pf"
        # the preset exactly, plus C722's kernel in both arms and the tiles' switches for "on"
        if [ "$arm" = on ]; then
            echo "PRESET_ENV=\"\$PRESET_ENV $M=1 $SERVED_ENV\"" >> "$pf"
        else
            echo "PRESET_ENV=\"\$PRESET_ENV $M=1\"" >> "$pf"
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
