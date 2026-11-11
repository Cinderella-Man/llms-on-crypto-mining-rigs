#!/usr/bin/env bash
# rig_c750.sh: C750 (Q1325) on a P106 rig. GGML_CUDA_MMVF_PASCAL_EXACT=1 (fork branch pascal-p106-v4-b12246-c750) adds
# the BF16 products of 2-4 tokens in the one-token order; on the P106-090's 5 SMs it runs Qwen3.8-Flash-Next's 10,240-wide
# hyper-connection products (hc_attn_down, hc_ffn_down, hc_head_down) at 3-4 tokens through a 32-lane kernel in the
# 128-lane order. Is every 2-4-token product the one-token products bit for bit, and at what cost against the rule's
# 32-lane shape (off) and the 128-lane shape (GGML_CUDA_MMVF_PASCAL_RULE=1)? Runs ON the rig, detached, the rig idle;
# ends with $OUT/DONE. Steps (STEPS, default "replay bitid perf served"):
#   replay  the recorded set bf16x (scripts/record_checks.sh SUITE=bf16x: BF16's 2-D MUL_MAT cases at 1-8 tokens, nine
#           runs) through scripts/replay_checks.sh on card DEV, about one small case in CPU_CHECK (default 20) also
#           against the rig's own CPU; passing: every run ends "ok", bf16x-fault and bf16x-fault2 with every one of
#           their cases failing
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid bf16: every 2-4-token product against its columns' one-token
#           products, bit for bit, with the switch at 1 and at 2 (0 comparisons may differ), at 1 as for 10 SMs (the
#           same), and off, the control (the 3-4-token products of the widths whose lanes change differ, exit 1)
#   perf    test-backend-ops perf at Flash-Next's BF16 shapes (its perf cases), 1-4 tokens, the arms of PERF_ARMS
#           alternated over PASSES passes: off (the rule: 32 lanes at 3-4 tokens), on (the switch: the four chains in
#           passes), on2 (the switch at 2: the chains interleaved), rule1 (GGML_CUDA_MMVF_PASCAL_RULE=1: 128 lanes at
#           3-4 tokens, exact). The 1-2-token cases and the shapes the switch leaves alone run the same kernel in every
#           arm, so they give the spread
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft, the switch off and on in ORDER, each with LOADS; the hyper-connections' time per verify pass
#           comes out of verify_parts.py on the traces, the step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the bf16x folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the
#   same tree); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER
#   (bench/agent_check/rig_server.sh), REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES
#   (default 2), PERF_ARMS (default "off on on2 rule1"), ORDER (default "off on on off"), LOADS (default "warm prof")
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... \
#   VERIFY_PROF=... setsid -f nohup bash rig_c750.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on on2 rule1}
ORDER=${ORDER:-off on on off}
LOADS=${LOADS:-warm prof}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
F=GGML_CUDA_MMVF_PASCAL
ON="${F}_EXACT=1"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)   echo "" ;;
        on)    echo "$ON" ;;
        on2)   echo "${F}_EXACT=2" ;;
        rule1) echo "${F}_RULE=1" ;;
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
    # the fault runs: every one of their cases must fail
    for r in bf16x-fault bf16x-fault2; do
        [ -f "$OUT/replay/$r.tsv" ] && log "$r: $(cut -f 3 "$OUT/replay/$r.tsv" | sort | uniq -c | tr '\n' ' ')"
    done
fi

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    for a in "on:$ON" "on2:${F}_EXACT=2" "on-nsm10:${F}_NSM=10 $ON" "control:"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" bf16 > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=bf16,type_b=f32,m=(320|10240|4|512|48),n=[1-4],k=(10240|320|2560),'
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
