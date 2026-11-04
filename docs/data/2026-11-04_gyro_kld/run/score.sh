#!/usr/bin/env bash
# The whole scoring run on one set of GPUs, in order:
#   control       a file the reference chain already scored (ISTA's GSQ-RCO Coder), now on the fork's CUDA build: each
#                 text must land within noise of the old score, max(floor, 2 standard errors) per text, or the chain
#                 stops here ($OUT/control.verdict says which)
#   coder, m      Gyro-Coder and Gyro-M on the three texts (CUDA build)
#   coder-vulkan  Gyro-Coder's code text on the Vulkan build
# Every file is read from a copy on a fast disk (hot_copy.sh: the n-gram table is read at random). Gyro-Coder's copy
# comes first, the control's next, and Gyro-M's once the control's copy is removed, so one fast disk holds at most two
# of them; each copy is removed when its runs are done unless KEEP_HOT=1.
# START_WHEN (a shell condition, checked every minute) holds the start until the GPUs' previous job has ended; then
# each GPU must also hold less than FREE_MIB (default 2000) MiB. Logs $OUT/<name>.<text>.log, the chain's own
# $OUT/score.log, $OUT/score.done at the end ("ok" or what failed).
#   (env: BIN BASES OUT GPUS HOT CODER GYRO_M CONTROL_FILES CONTROL_REF FLOOR [MOUNTS] [START_WHEN] [FREE_MIB] [STEPS]
#    [KEEP_HOT]) bash score.sh
# CONTROL_FILES "<shard> <sha256> ..." (first shard first); CONTROL_REF "<text>:<kld>:<se> ..."; FLOOR "<text>:<kld> ..."
set -uo pipefail
: "${BIN:?}" "${BASES:?}" "${OUT:?}" "${GPUS:?}" "${HOT:?}" "${CODER:?}" "${GYRO_M:?}" "${CONTROL_FILES:?}"
: "${CONTROL_REF:?}" "${FLOOR:?}"
here=$(dirname "$(realpath "$0")")
mkdir -p "$OUT"
exec >> "$OUT/score.log" 2>&1
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
export BIN BASES OUT GPUS MOUNTS="${MOUNTS:-} $HOT"
hot() { echo "$HOT/$(basename "$1")"; }
unhot() { [ "${KEEP_HOT:-0}" = 1 ] && return; for f in "$@"; do rm -f "$(hot "$f")" "$(hot "$f").ok"; done; }
mean() { grep -E "^Mean    KLD" "$1" 2> /dev/null | awk '{print $3}'; }
failed=""

log "copies first"
HOT=$HOT bash "$here/hot_copy.sh" "$CODER" || { log "copy of $CODER failed"; echo copy-coder > "$OUT/score.done"; exit 1; }
set -- $CONTROL_FILES
control=$1 control_shards=""
while [ $# -ge 2 ]; do
    HOT=$HOT bash "$here/hot_copy.sh" "$1" "$2" || { log "copy of $1 failed"; echo copy-control > "$OUT/score.done"; exit 1; }
    control_shards="$control_shards $1"; shift 2
done

n=0
while [ -n "${START_WHEN:-}" ] && ! eval "$START_WHEN"; do
    [ $((n % 15)) -eq 0 ] && log "waiting: $START_WHEN"; n=$((n + 1)); sleep 60
done
while :; do
    busy=""
    for g in ${GPUS//,/ }; do
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        [ "$used" -lt "${FREE_MIB:-2000}" ] || busy="$busy $used"
    done
    [ -z "$busy" ] && break
    [ $((n % 15)) -eq 0 ] && log "waiting for the GPUs to empty: MiB in use$busy"; n=$((n + 1)); sleep 60
done
log "start on GPUs $GPUS"

for step in ${STEPS:-control coder m coder-vulkan}; do
    case $step in
        control)
            BACKEND=cuda bash "$here/kld.sh" "$(hot "$control")" control-ista-coder || failed="$failed $step"
            verdict=ok
            for ref in $CONTROL_REF; do
                IFS=: read -r t old se <<< "$ref"
                fl=$(printf '%s\n' $FLOOR | awk -F: -v t="$t" '$1 == t {print $2}')
                new=$(mean "$OUT/control-ista-coder.$t.log")
                line=$(awk -v t="$t" -v n="${new:-nan}" -v o="$old" -v s="$se" -v f="${fl:-0}" 'BEGIN {
                    tol = (f > 2 * s) ? f : 2 * s; d = n - o; if (d < 0) d = -d
                    printf "%s new %s old %s diff %.4f tolerance %.4f %s", t, n, o, d, tol, (n == "nan" || d > tol) ? "MISS" : "within" }')
                log "control $line"
                echo "$line" >> "$OUT/control.verdict"
                case $line in *MISS) verdict=miss ;; esac
            done
            unhot $control_shards
            if [ "$verdict" != ok ]; then
                log "the control misses its old score: stopping before the Gyro files"
                unhot "$CODER"; echo control-miss > "$OUT/score.done"; exit 1
            fi
            HOT=$HOT bash "$here/hot_copy.sh" "$GYRO_M" > "$OUT/hot_copy_m.log" 2>&1 &
            copy_m=$! ;;
        coder) BACKEND=cuda bash "$here/kld.sh" "$(hot "$CODER")" gyro-coder || failed="$failed $step" ;;
        m) if [ -z "${copy_m:-}" ]; then HOT=$HOT bash "$here/hot_copy.sh" "$GYRO_M" > "$OUT/hot_copy_m.log" 2>&1 & copy_m=$!; fi
           wait $copy_m || { log "copy of $GYRO_M failed: $(tail -n 1 "$OUT/hot_copy_m.log")"; failed="$failed copy-m"; continue; }
           BACKEND=cuda bash "$here/kld.sh" "$(hot "$GYRO_M")" gyro-m || failed="$failed $step"
           unhot "$GYRO_M" ;;
        coder-vulkan) BACKEND=vulkan TEXTS=code bash "$here/kld.sh" "$(hot "$CODER")" gyro-coder-vulkan || failed="$failed $step" ;;
        *) log "unknown step $step"; failed="$failed $step" ;;
    esac
done
wait
unhot "$CODER"
log "end${failed:+, failed:$failed}"
echo "${failed:-ok}" > "$OUT/score.done"
