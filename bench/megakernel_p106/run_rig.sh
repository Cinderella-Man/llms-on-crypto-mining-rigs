#!/usr/bin/env bash
# run_rig.sh: megakernel_p106's checks, timings and the card-to-card hop on a rig, logged into OUT, ending with
# OUT/DONE. Run it detached from the folder that holds the binary and this script (README, "Running on the rig").
#
# Env: GPU   card for the stage checks and timings (default 0)
#      HOP   two cards for the hop, as CUDA device numbers in PCI order (default "0,1"; empty skips the hop)
#      OUT   output folder (default ./out)
#      BIN   the binary (default ./megakernel_p106)
#      FORCE 1: start even with a compute process on the cards
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}
HOP=${HOP-0,1}
OUT=${OUT:-$here/out}
BIN=${BIN:-$here/megakernel_p106}
FORCE=${FORCE:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$BIN" > "$OUT/md5.txt"
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

log "1/4 smoke: every timing path once"
CUDA_VISIBLE_DEVICES=$GPU "$BIN" --smoke --no-check > "$OUT/smoke.txt" 2>&1
log "exit $?"
log "2/4 checks"
CUDA_VISIBLE_DEVICES=$GPU "$BIN" --check-only > "$OUT/check.txt" 2>&1
log "exit $?"
log "3/4 timings"
CUDA_VISIBLE_DEVICES=$GPU "$BIN" --no-check --csv "$OUT/timing.csv" > "$OUT/timing.txt" 2>&1
log "exit $?"
if [ -n "$HOP" ]; then
    log "4/4 hop between cards $HOP"
    CUDA_VISIBLE_DEVICES=$HOP "$BIN" --hop 0,1 --hop-only --csv "$OUT/hop.csv" > "$OUT/hop.txt" 2>&1
    log "exit $?"
fi
log "done"
touch "$OUT/DONE"
