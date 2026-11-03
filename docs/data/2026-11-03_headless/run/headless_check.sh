#!/usr/bin/env bash
# Q83: what stopping the desktop frees on a rig, and whether Flint Q8_0 then serves a 16k-token prompt at 262,144 tokens
# of context with -ub 256 (the release uses -ub 128 because of the 4 GB host). Runs p106_check.sh's long-context serve
# check three times: desktop with -ub 256, headless with -ub 128, headless with -ub 256. Takes a snapshot of host RAM,
# VRAM per card and the desktop's resident memory before each phase, then starts the desktop again.
# Run on the rig as root, detached, with nothing else running there:
#   BIN=<engine>/bin MODEL=<flint-Q8_0.gguf> CHECK=<scripts/p106_check.sh> OUT=/tmp/headless bash headless_check.sh
set -u
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${CHECK:?set CHECK}"
TS=${TS:-6,6,5,5,6,5,5,6,6}
OUT=${OUT:-/tmp/headless}
mkdir -p "$OUT"

snap() {
  {
    echo "== $1 $(date +%T)"
    free -m | awk '/Mem:/{print "MemAvailable_MB " $7 ", used_MB " $3}'
    echo "vram_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    echo "desktop_rss_MB $(ps -eo rss=,comm= | awk '$2 ~ /^(Xorg|gnome|mutter|gdm|gsd-|ibus|pipewire|wireplumber|at-spi|xdg-|evolution|tracker|gjs|goa-|nautilus)/ {t += $1} END {printf "%d", t / 1024}')"
    echo "running_services $(systemctl list-units --type=service --state=running --no-legend | wc -l)"
  } >> "$OUT/snapshots.txt"
}
run() {
  BIN=$BIN MODEL=$MODEL TS=$TS SERVE=1 CONFIGS=long9 UB=$2 bash "$CHECK" > "$OUT/check_$1.log" 2>&1
  echo "$1: exit $?" >> "$OUT/summary.txt"
}

echo "start $(date +%T)" > "$OUT/summary.txt"
snap desktop
run desktop_ub256 256
systemctl isolate multi-user.target; sleep 15
snap headless
run headless_ub128 128
run headless_ub256 256
systemctl isolate graphical.target; sleep 20
snap desktop_again
echo "default target $(systemctl get-default), display manager $(systemctl is-active display-manager)" >> "$OUT/summary.txt"
echo "DONE $(date +%T)" >> "$OUT/summary.txt"
