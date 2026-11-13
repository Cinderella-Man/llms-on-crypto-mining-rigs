#!/usr/bin/env bash
# rig_c755.sh: C755 (Q1331) on a P106 rig, run R150 (Q1456). No engine change: GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0
# with the preset's GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 takes Qwen3.8-Flash-Next's shared-expert down projections in
# Q4_0 and Q5_0 (640 -> 2560) off the k-column kernel at 2-4 tokens and gives them to mul_mat_vec_q_pascal_mc, which runs
# the one-token kernel's block arithmetic for every token. Are they then bit for bit the one-token products on the card,
# is the multi-column kernel faster than the k-column kernel there, and what does a served four-token verify pass gain?
# Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Steps (STEPS, default "bitid perf served"):
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid shdown (Q4_0 and Q5_0 at 2560 x 640, 4097 x 640, 640 x 2560 and
#           333 x 4096, 2-4 tokens against their one-token products, three inputs each) in four arms:
#             on     the preset's k-column settings and the skip: 0 differences (exit 0)
#             table  no switch (the default dispatch table also leaves these products to mul_mat_vec_q_pascal_mc at
#                    2-4 tokens and K <= 4096): 0 differences (exit 0)
#             off    the preset's settings alone, the k-column kernel (a fault arm): the "big" products differ (exit 1)
#             mc0    the skip with GGML_CUDA_MMVQ_PASCAL_MC=0, llama.cpp's mul_mat_vec_q at 2-4 tokens (a fault arm):
#                    products differ (exit 1)
#   perf    test-backend-ops perf at Q4_0's and Q5_0's 640 -> 2560 (m = 2560, k = 640), 1-8 tokens, the arms kcol (the
#           preset's settings) and mc (the same with the skip) alternated over PASSES passes; 1 token runs the one-token
#           kernel in both arms, so it gives the spread; 2-4 tokens are mul_mat_vec_q_pascal_mc against the k-column
#           kernel, 5-8 llama.cpp's mul_mat_vec_q (where the skip sends them: the multi-column kernel stops at 4)
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset
#           with the MTP draft, the skip off and on in ORDER, each with LOADS; the two products' ms a four-token pass
#           come out of the traces (2026-11-10_release_profile's release_parts.py, then
#           2026-11-11_c731_shared_expert/run/shexp_times.sh), against C731's 0.880 (Q4_0 0.595, Q5_0 0.285)
# Env: BIN (bin/ of the pascal-p106-v4 rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); bitid:
#   BITID (multitoken_bitid built against that tree; it loads the rig's own CUDA 12.9 runtime libraries, as the
#   tarball's binaries do); served: MODEL, DRAFT, PRESET (release/flash-next-v3/settings.env), RIG_SERVER
#   (bench/agent_check/rig_server.sh), REQ (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES
#   (default 2), ORDER (default "off on on off"), LOADS (default "warm prof")
# Usage: BIN=... OUT=... BITID=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... VERIFY_PROF=... \
#   setsid -f nohup bash rig_c755.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-bitid perf served}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
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
PRESET_KCOL="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"
SKIP="${P}_KCOL_SKIP=q4_0,q5_0"
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    for a in "on:$PRESET_KCOL $SKIP" "table:" "off:$PRESET_KCOL" "mc0:$PRESET_KCOL $SKIP ${P}_MC=0"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" shdown > "$OUT/bitid/$n.txt" 2>&1
        rc=$?
        log "bitid $n exit $rc: $(grep -c ': DIFFERENT' "$OUT/bitid/$n.txt") differ, $(grep -c 'MUL_MAT/big .*: DIFFERENT' "$OUT/bitid/$n.txt") of them big; $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP='type_a=(q4_0|q5_0),type_b=f32,m=2560,n=[1-8],k=640,'
    for pass in $(seq 1 "$PASSES"); do
        for arm in kcol mc; do
            e="$PRESET_KCOL"
            [ "$arm" = mc ] && e="$e $SKIP"
            quiet
            log "perf $arm pass $pass start ($e)"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "$PP" \
                > "$OUT/perf/$arm-$pass.txt" 2>&1
            log "perf $arm pass $pass exit $?: $(grep -c 'us/run' "$OUT/perf/$arm-$pass.txt") lines"
        done
    done
    # one line per arm, pass, type and token count: the case's us/run
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
        # the preset exactly, plus the skip for "on"
        [ "$arm" = on ] && echo "PRESET_ENV=\"\$PRESET_ENV $SKIP\"" >> "$pf"
        log "served $k $arm start"
        BIN="$BIN" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$VERIFY_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
