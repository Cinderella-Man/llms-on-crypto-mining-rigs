#!/usr/bin/env bash
# run_rig.sh: persistent_pipe's runs on a rig, logged into OUT, ending with OUT/DONE. Run it detached from the folder
# that holds the binary and this script (README, "Running on a rig"). ~15 minutes on nine P106 cards.
#
# Steps (STEPS, in this order by default):
#   smoke  every path once on the first two cards (--smoke)
#   f32    every card, the residual in F32, every variant, 1,000 timed passes
#   q8     the same with the residual in Q8_0 (the release line's LLAMA_HOP_CAST=q8_0)
#   two    the first two cards, F32: one boundary and the token's way back
#   k4     every card, Q8_0, 4 tokens a pass (a verify pass's width), 250 timed passes
#   polls  every card, F32, the two persistent variants with polls every 0.5 and every 8 us (default 2), 300 passes
# The keep-alive steps (C533, not in the default STEPS): every card, Q8_0, the staged hop (the release line's), 1,000
# timed passes (250 at k = 4), each with its control run first:
#   kaoff      no keep-alive
#   karesident --keepalive resident: one thread a card polling a word in mapped host memory every 50 us
#   kapulse    --keepalive pulse: one thread a card busy 100 ms every 1,000 ms
#   kak4off    4 tokens a pass, no keep-alive
#   kak4       4 tokens a pass, --keepalive resident
#   kak4pulse  4 tokens a pass, --keepalive pulse
# The runs load every kernel when they start (CUDA_MODULE_LOADING=EAGER): with CUDA 12's lazy loading, the first launch
# of a kernel not loaded yet waits until every kernel on the card has ended, so a resident keep-alive held it until its
# own timeout and the timed passes ran without it (C533's first rig run).
#
# Env: OUT     output folder (default ./out)
#      BIN     the binary (default ./persistent_pipe)
#      CARDS   cards for the full runs (default: every visible card)
#      STEPS   the steps to run (default "smoke f32 q8 two k4 polls")
#      EXTRA   more options for every run (e.g. "--layers 5")
#      PAUSE   seconds of idle before each step (default 0), so that a step does not start at the clock the last one left
#      FORCE   1: start even with a compute process on the cards
#      CUDA_MODULE_LOADING  default EAGER (above)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$here/out}
BIN=${BIN:-$here/persistent_pipe}
CARDS=${CARDS:-}
STEPS=${STEPS:-smoke f32 q8 two k4 polls}
EXTRA=${EXTRA:-}
FORCE=${FORCE:-0}
PAUSE=${PAUSE:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-EAGER}
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.limit --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$BIN" > "$OUT/md5.txt"
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

cards=()
[ -n "$CARDS" ] && cards=(--cards "$CARDS")
fails=0
run() {  # name, options...; a step named twice in STEPS writes <name>-2, <name>-3, ...
    local step=$1 name=$1 n=2
    shift
    while [ -e "$OUT/$name.txt" ]; do name=$step-$n; n=$((n + 1)); done
    log "$name: $*"
    # shellcheck disable=SC2086
    "$BIN" "$@" $EXTRA --csv "$OUT/$name.csv" > "$OUT/$name.txt" 2>&1
    local rc=$?
    log "$name: exit $rc"
    [ "$rc" = 0 ] || fails=$((fails + 1))
}
for step in $STEPS; do
    [ "$PAUSE" != 0 ] && { log "pause $PAUSE s"; sleep "$PAUSE"; }
    case $step in
        smoke) run smoke --cards 2 --smoke ;;
        f32)   run f32 "${cards[@]}" --wire f32 ;;
        q8)    run q8 "${cards[@]}" --wire q8 ;;
        two)   run two --cards 2 --wire f32 ;;
        k4)    run k4 "${cards[@]}" --wire q8 --k 4 --passes 250 --warmup 10 ;;
        polls) for us in 0.5 8; do
                   run "poll$us" "${cards[@]}" --wire f32 --variants persist,persist-direct --poll-us "$us" --passes 300
               done ;;
        kaoff)      run kaoff "${cards[@]}" --wire q8 --variants staged ;;
        karesident) run karesident "${cards[@]}" --wire q8 --variants staged --keepalive resident ;;
        kapulse)    run kapulse "${cards[@]}" --wire q8 --variants staged --keepalive pulse ;;
        kak4off)    run kak4off "${cards[@]}" --wire q8 --variants staged --k 4 --passes 250 --warmup 10 ;;
        kak4)       run kak4 "${cards[@]}" --wire q8 --variants staged --k 4 --passes 250 --warmup 10 --keepalive resident ;;
        kak4pulse)  run kak4pulse "${cards[@]}" --wire q8 --variants staged --k 4 --passes 250 --warmup 10 --keepalive pulse ;;
        *)     log "unknown step $step" ;;
    esac
done
log "done, $fails failed"
touch "$OUT/DONE"
