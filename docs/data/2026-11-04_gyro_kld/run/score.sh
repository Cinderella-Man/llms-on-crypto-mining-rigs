#!/usr/bin/env bash
# The whole scoring run on one set of GPUs. STEPS is a list of <step>:<texts> in order (default below):
#   control       a file the reference chain already scored (ISTA's GSQ-RCO Coder), now on the fork's CUDA build: each
#                 text must land within noise of the old score, max(floor, 2 standard errors) per text, or the chain
#                 stops here ($OUT/control.verdict says which)
#   coder, m      Gyro-Coder and Gyro-M (CUDA build)
#   coder-vulkan  Gyro-Coder on the Vulkan build
# A text whose log already holds its score is not run again, so the chain can be restarted at any point.
# Files are read from a copy on a fast disk (hot_copy.sh), made when a step still has a text to run, or in place for
# the steps named in IN_PLACE (e.g. "m": a fast disk that is busy with writes serves reads slowly). For those, WARM=1
# reads the n-gram table once, sequentially, at the start, so its random reads later come from the page cache. The
# control's copy is removed after its check, the others at the end unless KEEP_HOT=1.
# START_WHEN (a shell condition, checked every minute) holds the start until the GPUs' previous job has ended; then
# each GPU must also hold less than FREE_MIB (default 2000) MiB. Logs $OUT/<name>.<text>.log, the chain's own
# $OUT/score.log, $OUT/score.done at the end ("ok" or what failed). A file $OUT/STOP ends the chain after its step.
#   (env: BIN BASES OUT GPUS HOT CODER GYRO_M CONTROL_FILES CONTROL_REF FLOOR [MOUNTS] [START_WHEN] [FREE_MIB] [STEPS]
#    [IN_PLACE] [WARM] [KEEP_HOT] [MIN_FREE_GB]) bash score.sh
# CONTROL_FILES "<shard> <sha256> ..." (first shard first); CONTROL_REF "<text>:<kld>:<se> ..."; FLOOR "<text>:<kld> ..."
set -uo pipefail
: "${BIN:?}" "${BASES:?}" "${OUT:?}" "${GPUS:?}" "${HOT:?}" "${CODER:?}" "${GYRO_M:?}" "${CONTROL_FILES:?}"
: "${CONTROL_REF:?}" "${FLOOR:?}"
STEPS=${STEPS:-control:agent,code coder:agent,code m:agent,code coder:wiki m:wiki coder-vulkan:code}
here=$(dirname "$(realpath "$0")")
mkdir -p "$OUT"
exec >> "$OUT/score.log" 2>&1
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
export BIN BASES OUT GPUS MOUNTS="${MOUNTS:-} $HOT"
hot() { echo "$HOT/$(basename "$1")"; }
unhot() { [ "${KEEP_HOT:-0}" = 1 ] && return; for f in "$@"; do rm -f "$(hot "$f")" "$(hot "$f").ok"; done; }
mean() { grep -E "^Mean    KLD" "$1" 2> /dev/null | awk '{print $3}'; }
in_place() { case " ${IN_PLACE:-} " in *" $1 "*) return 0 ;; esac; return 1; }
todo() { local t; for t in $2; do grep -q "^Mean    KLD" "$OUT/$1.$t.log" 2> /dev/null || return 0; done; return 1; }
# model <step> <file> [sha256]: the path the step reads, copied first unless the step reads in place
model() {
    if in_place "$1"; then echo "$2"; return; fi
    HOT=$HOT bash "$here/hot_copy.sh" "$2" ${3:-} >&2 || return 1
    hot "$2"
}
failed=""

if [ "${WARM:-0}" = 1 ]; then
    for s in $IN_PLACE; do
        case $s in coder|coder-vulkan) f=$CODER ;; m) f=$GYRO_M ;; *) continue ;; esac
        ( read -r start len <<< "$(python3 -I "$here/header.py" --range per_layer_token_embd.weight "$f")"
          log "warm: $(basename "$f")'s n-gram table, $len bytes"
          ionice -c2 -n7 dd if="$f" of=/dev/null bs=64M iflag=skip_bytes,count_bytes skip="$start" count="$len" status=none
          log "warm: $(basename "$f") done" ) &
    done
fi

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
log "start on GPUs $GPUS: $STEPS${IN_PLACE:+ (in place: $IN_PLACE)}"

for item in $STEPS; do
    step=${item%%:*} texts=${item#*:}
    [ "$texts" = "$item" ] && texts="wiki agent code"
    texts=${texts//,/ }
    case $step in
        control)
            shards=$(printf '%s\n' $CONTROL_FILES | awk 'NR % 2')
            if todo control-ista-coder "$texts"; then
                set -- $CONTROL_FILES
                control=""
                while [ $# -ge 2 ]; do
                    p=$(model control "$1" "$2") || { log "copy of $1 failed"; echo copy-control > "$OUT/score.done"; exit 1; }
                    control=${control:-$p}; shift 2
                done
                BACKEND=cuda TEXTS=$texts bash "$here/kld.sh" "$control" control-ista-coder || failed="$failed $item"
            fi
            in_place control || unhot $shards
            verdict=ok
            for t in $texts; do
                ref=$(printf '%s\n' $CONTROL_REF | awk -F: -v t="$t" '$1 == t')
                IFS=: read -r _ old se <<< "$ref"
                fl=$(printf '%s\n' $FLOOR | awk -F: -v t="$t" '$1 == t {print $2}')
                new=$(mean "$OUT/control-ista-coder.$t.log")
                line=$(awk -v t="$t" -v n="${new:-nan}" -v o="${old:-nan}" -v s="${se:-0}" -v f="${fl:-0}" 'BEGIN {
                    tol = (f > 2 * s) ? f : 2 * s; d = n - o; if (d < 0) d = -d
                    printf "%s new %s old %s diff %.4f tolerance %.4f %s", t, n, o, d, tol, (n == "nan" || o == "nan" || d > tol) ? "MISS" : "within" }')
                log "control $line"
                echo "$line" >> "$OUT/control.verdict"
                case $line in *MISS) verdict=miss ;; esac
            done
            if [ "$verdict" != ok ]; then
                log "the control misses its old score: stopping before the Gyro files"
                echo control-miss > "$OUT/score.done"; exit 1
            fi ;;
        coder|m|coder-vulkan)
            case $step in coder) f=$CODER name=gyro-coder b=cuda ;; m) f=$GYRO_M name=gyro-m b=cuda ;;
                coder-vulkan) f=$CODER name=gyro-coder-vulkan b=vulkan ;; esac
            todo $name "$texts" || continue
            p=$(model $step "$f") || { log "copy of $f failed"; failed="$failed copy-$step"; continue; }
            BACKEND=$b TEXTS=$texts bash "$here/kld.sh" "$p" $name || failed="$failed $item" ;;
        *) log "unknown step $step"; failed="$failed $item" ;;
    esac
    [ -e "$OUT/STOP" ] && { log "STOP file: ending after $item"; failed="$failed stopped"; break; }
done
wait
unhot "$CODER" "$GYRO_M"
log "end${failed:+, failed:$failed}"
echo "${failed:-ok}" > "$OUT/score.done"
