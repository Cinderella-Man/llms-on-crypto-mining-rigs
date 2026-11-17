#!/usr/bin/env bash
# run.sh (R186 Q1551, R188 Q1553 stage 2): two loads of Flint Pascal Q8 GGUF on its released preset, driven from a client
# machine over ssh, the requests sent through an ssh forward to the rig's loopback.
#   load A  the preset as released (release/flint-v4/settings.env; its keep-alive is off)
#     snap        host memory, resident processes, NVIDIA modules and interrupts, with the cards empty and loaded
#     warm-up     one 128-token decode
#     interrupts  the NVIDIA lines of /proc/interrupts around 20 s idle, one 256-token decode, 20 s idle, one more
#                 256-token decode; no other command on the rig inside a window
#     clocks      nvidia-smi every 500 ms for one minute: 15 s idle, one 512-token decode, then idle to the minute
#     modes       nvidia-smi every 100 ms; 256-token decodes with PowerMizer mode 0 (the default) and mode 1 on all
#                 nine cards, set through NVML between requests, in the order 0 1 1 0 0 1 1 0; mode 0 at the end
#   load B  the preset plus GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 (the keep-alive arm of the same day)
#     warm-up, then nvidia-smi every 100 ms and four 256-token decodes
# Environment: RIG (ssh alias), WORK (a folder on the rig, removed by hand afterwards), BIN (the build's bin folder on
# the rig), MODEL (the file on the rig), OUT (a local folder), PORT (8186, the rig's loopback), TPORT (18186, the local
# end of the forward). Needs: rig_r186.sh, pm_mode.py, decode_req.py beside it, scripts/ram_watchdog.sh in the repo.
set -uo pipefail
: "${RIG:?}" "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${OUT:?}"
PORT=${PORT:-8186}; TPORT=${TPORT:-18186}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
# shellcheck disable=SC1091
. "$REPO/release/flint-v4/settings.env"
mkdir -p "$OUT"
URL=http://127.0.0.1:$TPORT
log() { echo "$(date +%s.%N) $*" | tee -a "$OUT/run.log"; }
R() { ssh "$RIG" "WORK=$WORK PORT=$PORT $* bash $WORK/rig_r186.sh"; }
req() { python3 -I "$HERE/decode_req.py" "$URL" "$1" "$2" | tee -a "$OUT/requests.tsv"; }
pm() { ssh "$RIG" "CARDS=0,1,2,3,4,5,6,7,8 MODE=$1 python3 -I $WORK/pm_mode.py" >> "$OUT/pm.log"; }
irq() { R ACTION=irq TAG="$1" >> "$OUT/irq.txt"; }

ssh "$RIG" "mkdir -p $WORK"
for f in rig_r186.sh pm_mode.py; do ssh "$RIG" "cat > $WORK/$f" < "$HERE/$f"; done
ssh "$RIG" "cat > $WORK/ram_watchdog.sh" < "$REPO/scripts/ram_watchdog.sh"
log "clock offset: local $(date +%s.%N), rig $(ssh "$RIG" date +%s.%N), local $(date +%s.%N)"
R ACTION=snap TAG=empty >> "$OUT/snap.txt"
ssh "$RIG" "python3 -I $WORK/pm_mode.py" >> "$OUT/pm.log"

ssh -N -o ExitOnForwardFailure=yes -L "127.0.0.1:$TPORT:127.0.0.1:$PORT" "$RIG" &
TUN=$!
trap 'kill $TUN 2> /dev/null' EXIT

# ---- load A: the preset as released
log "start A"
R ACTION=start ARM=A BIN="$BIN" MODEL="$MODEL" SERVER_ENV="'$PRESET_ENV'" SERVER_ARGS="'$PRESET_ARGS'" | tee -a "$OUT/run.log" \
  || { log "load A failed"; exit 1; }
R ACTION=snap TAG=loadedA >> "$OUT/snap.txt"
req warmA 128

log "interrupts"
irq i0; sleep 20; irq i1
req irq1 256
irq i2; sleep 20; irq i3
req irq2 256
irq i4

log "clocks every 500 ms for one minute"
R ACTION=log_start TAG=r186a SMI_MS=500 | tee -a "$OUT/run.log"
LS=$(date +%s); sleep 15; req r186a 512
REST=$((LS + 60 - $(date +%s))); [ "$REST" -lt 20 ] && REST=20; sleep "$REST"
R ACTION=log_stop TAG=r186a | tee -a "$OUT/run.log"

log "modes 0 and 1, every 100 ms"
R ACTION=log_start TAG=modesA SMI_MS=100 | tee -a "$OUT/run.log"
for m in 0 1 1 0 0 1 1 0; do
  pm "$m"; sleep 3; req "A_mode$m" 256
done
pm 0
R ACTION=log_stop TAG=modesA | tee -a "$OUT/run.log"
R ACTION=snap TAG=loadedA_end >> "$OUT/snap.txt"
R ACTION=stop ARM=A | tee -a "$OUT/run.log"
sleep 5
R ACTION=snap TAG=stoppedA >> "$OUT/snap.txt"

# ---- load B: the keep-alive arm
log "start B"
R ACTION=start ARM=B BIN="$BIN" MODEL="$MODEL" \
  SERVER_ENV="'$PRESET_ENV GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1'" SERVER_ARGS="'$PRESET_ARGS'" \
  | tee -a "$OUT/run.log" || { log "load B failed"; exit 1; }
R ACTION=snap TAG=loadedB >> "$OUT/snap.txt"
req warmB 128
R ACTION=log_start TAG=keepaliveB SMI_MS=100 | tee -a "$OUT/run.log"
for i in 1 2 3 4; do sleep 3; req "B_keepalive" 256; done
R ACTION=log_stop TAG=keepaliveB | tee -a "$OUT/run.log"
R ACTION=stop ARM=B | tee -a "$OUT/run.log"
sleep 5
R ACTION=snap TAG=end >> "$OUT/snap.txt"
ssh "$RIG" "python3 -I $WORK/pm_mode.py" >> "$OUT/pm.log"
ssh "$RIG" "tar -C $WORK -cf - ." | tar -C "$OUT" -xf -
log "done"
touch "$OUT/DONE"
