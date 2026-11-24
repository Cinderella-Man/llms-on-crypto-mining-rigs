#!/usr/bin/env bash
# C1025's rig half (Q1758): rig_probe.py on one P106 rig, detached, with one build of pascal-p106-v5-nvml-probe and one
# release preset; arms GGML_CUDA_NVML_PROBE unset (the probe skipped: no P106 has fast BF16) and =1 (upstream's probe).
# Run from the repository root on a machine that only runs ssh:
#   RIG=<ssh alias> WORK=<folder on the rig> BIN=<llama-server on the rig> MODEL=<model file on the rig> \
#     PRESET=release/<flint|flash-next|ornith>-v5/settings.env [REPS=5] [LOADS="unset probe"] [PORT=18925] \
#     STEP=start bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
#   then STEP=wait (blocks until it ends, at most 1 h), STEP=collect with OUT=<folder here>, STEP=stop to end it early.
# STEP=start first prints the two load lines (DRY=1 stops there). Every ssh call runs under timeout.
set -uo pipefail
: "${RIG:?}" "${WORK:?}"
STEP=${STEP:-start}
HERE=docs/data/2026-11-24_c1025_nvml_probe/run
SSH="ssh -o BatchMode=yes"
case $STEP in
start)
    : "${BIN:?}" "${MODEL:?}" "${PRESET:?}"
    # shellcheck disable=SC1090
    . "$PRESET"
    q() { printf '%q' "$1"; }
    cmd="python3 -I $WORK/rig_probe.py --bin $(q "$BIN") --model $(q "$MODEL") --env $(q "$PRESET_ENV") --args $(q "$PRESET_ARGS") \
--out $WORK/out --reps ${REPS:-5} --loads $(q "${LOADS:-unset probe}") --port ${PORT:-18925}"
    echo "load unset: env $PRESET_ENV $BIN -lv 4 -m $MODEL $PRESET_ARGS --host 127.0.0.1 --port ${PORT:-18925}"
    echo "load probe: env GGML_CUDA_NVML_PROBE=1 $PRESET_ENV $BIN -lv 4 -m $MODEL $PRESET_ARGS --host 127.0.0.1 --port ${PORT:-18925}"
    [ -n "${DRY:-}" ] && { echo "DRY=1: nothing started"; exit 0; }
    timeout 60 $SSH "$RIG" "mkdir -p $WORK && test ! -e $WORK/out" || { echo "$WORK/out exists on the rig (or no ssh): stop"; exit 1; }
    timeout 60 $SSH "$RIG" "cat > $WORK/rig_probe.py" < "$HERE/rig_probe.py" || exit 1
    timeout 60 $SSH "$RIG" "test -x $(q "$BIN") && test -f $(q "$MODEL") && ! pgrep -x llama-server" \
        || { echo "the binary or the model is missing, or a llama-server runs: stop"; exit 1; }
    timeout 30 $SSH "$RIG" "cd $WORK && setsid -f nohup bash -c '$cmd > console.txt 2>&1; echo \$? > DONE' > /dev/null 2>&1 < /dev/null"
    sleep 5
    timeout 30 $SSH "$RIG" "cat $WORK/console.txt; ls $WORK/out/logs | head -n 3"
    ;;
wait)
    timeout 3600 $SSH "$RIG" "while [ ! -f $WORK/DONE ]; do sleep 15; done; echo rc=\$(cat $WORK/DONE); cat $WORK/console.txt"
    ;;
collect)
    : "${OUT:?}"
    mkdir -p "$OUT"
    timeout 300 $SSH "$RIG" "tar -C $WORK -czf - out console.txt DONE" | tar -C "$OUT" -xzf - && ls -l "$OUT/out"
    ;;
stop)
    timeout 60 $SSH "$RIG" "pkill -f $WORK/rig_probe.py; sleep 2; pkill -x llama-server; true"
    ;;
esac
