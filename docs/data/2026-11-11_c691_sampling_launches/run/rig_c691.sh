#!/usr/bin/env bash
# rig_c691.sh: C691 (Q1223) on a P106 rig. The backend samplers' one top-k over the output rows (fork branch
# pascal-p106-v4-b12246-c691, LLAMA_SAMPLING_TOP_K_ROWS=1) checked natively and timed: does a served verify pass's
# sampling chain on the head's card (R49: 0.68 ms, a PAD 0.22 and four TOP_K 0.33) get shorter, with every row's
# candidates the same? Runs ON the rig, detached, the rig idle; ends with $OUT/DONE. Steps (STEPS, default
# "replay perf served"):
#   replay  the recorded set top-k-rows (scripts/record_checks.sh SUITE=top-k-rows: TOP_K at 248,320 columns over one
#           to eight rows against the build machine's CPU, and TOP_K_ROWS, one op's rows against each row's own top-k
#           on the card) through scripts/replay_checks.sh on card DEV, about one small case in CPU_CHECK (default 20)
#           also against the rig's own CPU; passing: every run ends "ok"
#   perf    test-backend-ops perf on card DEV, PASSES passes: TOP_K at 248,320 columns, k = 20, one, two and four rows
#           (four one-row launches against one four-row launch), and the PAD of four such rows that the switch drops
#   served  R49's profile of a served verify pass (2026-11-10_served_verify_parts/run/verify_prof.sh) on the preset with
#           the MTP draft, the switch off and on in ORDER, each with LOADS; the sampling chain per verify pass comes
#           out of verify_parts.py on the traces, the step time out of the warm loads' server timings
# Env: BIN (bin/ of the branch's rig tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET
#   (the top-k-rows folder), REPLAY (scripts/replay_checks.sh); served: MODEL, DRAFT, PRESET
#   (release/flash-next-v3/settings.env), RIG_SERVER (bench/agent_check/rig_server.sh), REQ
#   (2026-11-04_draft_one_card/run/greedy_request.json), VERIFY_PROF; PASSES (default 2), ORDER (default
#   "off on on off"), LOADS (default "warm prof")
# Usage: BIN=... OUT=... SET=... REPLAY=... MODEL=... DRAFT=... PRESET=... RIG_SERVER=... REQ=... VERIFY_PROF=... \
#   setsid -f nohup bash rig_c691.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay perf served}
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
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step replay; then
    : "${SET:?}" "${REPLAY:?}"
    log "replay start"
    SET="$SET" BIN="$BIN" OUT="$OUT/replay" DEV="$DEV" CPU_CHECK="${CPU_CHECK:-20}" bash "$REPLAY" > "$OUT/replay-console.txt" 2>&1
    log "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt" 2>/dev/null)"
fi

if step perf; then
    mkdir -p "$OUT/perf"
    for pass in $(seq 1 "$PASSES"); do
        for c in 'TOP_K|ne=\[248320,' 'PAD|ne_a=\[248320,4,'; do
            o=${c%%|*}
            quiet
            log "perf $o pass $pass start"
            env CUDA_VISIBLE_DEVICES="$DEV" "$BIN/test-backend-ops" perf -b CUDA0 -o "$o" -p "${c#*|}" \
                > "$OUT/perf/$o-$pass.txt" 2>&1
            log "perf $o pass $pass exit $?: $(grep -c 'us/run' "$OUT/perf/$o-$pass.txt") lines"
        done
    done
    # one line per op, pass and case: the case string and its us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *([A-Z_]+\(.*\)): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2/"
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
        [ "$arm" = on ] && echo "PRESET_ENV=\"\$PRESET_ENV LLAMA_SAMPLING_TOP_K_ROWS=1\"" >> "$pf"
        log "served $k $arm start"
        BIN="$BIN" MODEL="$MODEL" DRAFT="$DRAFT" PRESET="$pf" RIG_SERVER="$RIG_SERVER" REQ="$REQ" OUT="$OUT/served/$k-$arm" \
            LOADS="$LOADS" bash "$VERIFY_PROF" > "$OUT/served/$k-$arm.console.txt" 2>&1
        log "served $k $arm exit $?: the switch's line in the server logs $(cat "$OUT/served/$k-$arm"/*/server-*.log 2>/dev/null | grep -c 'LLAMA_SAMPLING_TOP_K_ROWS=1')"
    done
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
